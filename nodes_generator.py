"""YuE2 Neural Music Generator Node for ComfyUI."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import folder_paths

from .nodes_model_loader import (
    DEVICES,
    get_yue2_models,
    get_yue2_vaes,
    resolve_model_path,
    resolve_vae_path,
    ensure_model_downloaded,
    ensure_vae_downloaded,
    sanitize_device
)
from .nodes_llm import clean_yue2_lyrics

RUNNER_SCRIPT = Path(__file__).parent / "runner.py"

def normalize_text(text) -> str:
    if not text:
        return ""
    if not isinstance(text, str):
        if isinstance(text, dict):
            sections = []
            for k, v in text.items():
                if str(k).lower() in ("title", "style", "cot", "suggested_seed", "notes", "bpm"):
                    continue
                header = str(k).replace("_", " ").strip()
                if not header.startswith("["):
                    header = f"[{header.title()}]"
                if not header.endswith("]"):
                    header = f"{header}]"
                if isinstance(v, (list, tuple)):
                    body = "\n".join(str(x) for x in v)
                elif isinstance(v, dict):
                    body = "\n".join(f"{sk}: {sv}" for sk, sv in v.items())
                else:
                    body = str(v)
                sections.append(f"{header}\n{body}".strip())
            text = "\n\n".join(s for s in sections if s)
        elif isinstance(text, (list, tuple)):
            text = "\n".join(str(x) for x in text)
        else:
            text = str(text)

    text = str(text).replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "").replace("\\t", " ")

    replacements = {
        "\u2011": "-", "\u2010": "-", "\u2012": "-", "\u2013": "-",
        "\u2014": "--", "\u2015": "--", "\u00a0": " ", "\u200b": "",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2026": "..."
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()

_WORKER_PROCESS = None
_WORKER_CONFIG = None

def get_or_start_worker(model_path, vae_path, target_device, attention_backend, memory_budget_gib, ode_steps):
    global _WORKER_PROCESS, _WORKER_CONFIG
    desired_config = (str(model_path), str(vae_path), str(target_device), str(attention_backend), int(memory_budget_gib))
    
    if _WORKER_PROCESS is not None:
        if _WORKER_PROCESS.poll() is None and _WORKER_CONFIG == desired_config:
            return _WORKER_PROCESS
        unload_worker()

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        sys.executable,
        "-X", "utf8",
        str(RUNNER_SCRIPT),
        "--worker",
        "--model", str(model_path),
        "--vae", str(vae_path),
        "--device", str(target_device),
        "--memory-budget-gib", str(memory_budget_gib),
        "--ode-steps", str(ode_steps),
        "--attention-backend", str(attention_backend),
        "--offline"
    ]

    print(f"\n[YuE2 ComfyUI] Starting persistent YuE2 worker process on {target_device}...")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1
    )

    for line in iter(proc.stdout.readline, ""):
        line_str = line.strip()
        if line_str:
            print(f"[YuE2] {line_str}")
        if "[YuE2 Worker] READY" in line_str:
            break
        if proc.poll() is not None:
            raise RuntimeError(f"YuE2 worker process failed to start with exit code {proc.returncode}")

    _WORKER_PROCESS = proc
    _WORKER_CONFIG = desired_config
    return _WORKER_PROCESS

def unload_worker():
    global _WORKER_PROCESS, _WORKER_CONFIG
    if _WORKER_PROCESS is not None:
        try:
            if _WORKER_PROCESS.poll() is None:
                print("[YuE2 ComfyUI] Unloading resident YuE2 worker and releasing VRAM...")
                _WORKER_PROCESS.stdin.write("QUIT\n")
                _WORKER_PROCESS.stdin.flush()
                _WORKER_PROCESS.wait(timeout=5)
        except Exception:
            try:
                _WORKER_PROCESS.kill()
            except Exception:
                pass
        _WORKER_PROCESS = None
        _WORKER_CONFIG = None

class YuE2SongGenerator:
    """Synthesizes complete stereo songs from style and lyrics using the YuE2 foundation model."""

    @classmethod
    def INPUT_TYPES(cls):
        models = get_yue2_models()
        vaes = get_yue2_vaes()
        return {
            "required": {
                "style": ("STRING", {
                    "multiline": True,
                    "default": "English, high emotive male tenor, K-Pop Dance Pop, heavy 808 bass, sharp synth leads, driving four-on-the-floor beat, futuristic hyper-energy vibe, 102 BPM"
                }),
                "lyrics": ("STRING", {
                    "multiline": True,
                    "default": "[Verse 1]\nFlickering screen light in the dark room glow\nAnother late night where my feelings start to grow\n\n[Chorus]\nPixel heartbeat you're the rhythm in my soul\nMy electric muse that makes me feel whole\n[End]"
                }),
                "cot": (["full", "melody", "off"], {"default": "full"}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "The random seed used for music generation. Use the dropdown below to randomize, increment, decrement, or keep fixed."
                }),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.05}),
                "model": (models, {"default": models[0]}),
                "vae": (vaes, {"default": vaes[0]}),
                "device": (DEVICES, {"default": "auto"}),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model loaded in GPU VRAM between generations for instantaneous repeat generations (saves ~10-15s load time). If False, immediately unloads the model and frees all VRAM after generation."
                }),
            },
            "optional": {
                "track_id": ("STRING", {"default": "comfy_song"}),
                "ode_steps": ("INT", {"default": 32, "min": 12, "max": 64, "step": 4, "tooltip": "Diffusion ODE steps for audio synthesis (32=reference fidelity, 24=fast 1.3x, 16=ultra-fast 2x)"}),
                "attention_backend": (["auto", "cudnn", "sage", "sdpa"], {"default": "auto", "tooltip": "Attention acceleration engine: auto/cudnn is fastest on RTX 40/50 series, sage uses SageAttention INT8 QK, sdpa is standard PyTorch"}),
                "memory_budget_gib": ("INT", {"default": 30, "min": 12, "max": 64}),
                "save_intermediate_artifacts": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, saves intermediate debug files (latent.npy, semantic.npy, plan.json) in ComfyUI/output/YuE2/. Keep False for a clean output directory containing only the final audio."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("audio", "abc_score", "audio_path", "lyrics_used", "style_used", "seed")
    FUNCTION = "generate_song"
    CATEGORY = "YuE2/Generation"

    def generate_song(self, style, lyrics, cot="full", seed=0, cfg_scale=1.0, model="YuE2-3B", vae="YuE2-Vae", device="auto", keep_model_loaded=True, track_id="comfy_song", ode_steps=32, attention_backend="auto", memory_budget_gib=30, save_intermediate_artifacts=False):
        model_path = ensure_model_downloaded(model)
        vae_path = ensure_vae_downloaded(vae)
        target_device = sanitize_device(device)
        clean_style = normalize_text(style)
        clean_lyrics = clean_yue2_lyrics(normalize_text(lyrics))

        # Enforce English at start of style string
        if not clean_style.lower().startswith("english"):
            clean_style = f"English, {clean_style.lstrip(', ')}"

        if seed == 0:
            import random
            seed = random.randint(100000, 99999999)

        timestamp = int(time.time())
        safe_id = "".join(c if c.isalnum() or c == "_" else "_" for c in track_id).strip("_") or "song"
        folder_name = f"{safe_id}_{timestamp}"

        # Destination directory for synthesis scratch files
        if save_intermediate_artifacts:
            out_dir = Path(folder_paths.get_output_directory()) / "YuE2" / folder_name
        else:
            out_dir = Path(folder_paths.get_temp_directory()) / "yue2_scratch" / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write request file to ComfyUI temp directory
        temp_dir = Path(folder_paths.get_temp_directory()) / "yue2_requests"
        temp_dir.mkdir(parents=True, exist_ok=True)
        req_file = temp_dir / f"{folder_name}.json"
        req_data = {
            "id": safe_id,
            "style": clean_style,
            "lyrics": clean_lyrics,
            "cot": cot,
            "seed": seed,
            "cfg_scale": float(cfg_scale),
            "ode_steps": int(ode_steps),
            "attention_backend": str(attention_backend),
            "save_artifacts": bool(save_intermediate_artifacts)
        }
        req_file.write_text(json.dumps(req_data, indent=2, ensure_ascii=False), encoding="utf-8")

        if keep_model_loaded:
            worker = get_or_start_worker(model_path, vae_path, target_device, attention_backend, memory_budget_gib, ode_steps)
            print(f"\n[YuE2 ComfyUI] Submitting request to resident worker on {target_device} (Mode: {cot} | Seed: {seed} | CFG: {cfg_scale})...")
            task = {"request": str(req_file), "output": str(out_dir), "ode_steps": ode_steps}
            worker.stdin.write(json.dumps(task) + "\n")
            worker.stdin.flush()

            for line in iter(worker.stdout.readline, ""):
                line_str = line.strip()
                if line_str:
                    print(f"[YuE2] {line_str}")
                if "[YuE2 Worker] COMPLETED" in line_str:
                    break
                if "[YuE2 Worker] ERROR" in line_str:
                    raise RuntimeError(f"YuE2 generation failed: {line_str}")
                if worker.poll() is not None:
                    raise RuntimeError(f"YuE2 worker exited unexpectedly with code {worker.returncode}")
        else:
            unload_worker()
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"

            cmd = [
                sys.executable,
                "-X", "utf8",
                str(RUNNER_SCRIPT),
                "--request", str(req_file),
                "--output", str(out_dir),
                "--model", str(model_path),
                "--vae", str(vae_path),
                "--device", target_device,
                "--memory-budget-gib", str(memory_budget_gib),
                "--ode-steps", str(ode_steps),
                "--attention-backend", str(attention_backend),
                "--offline"
            ]

            print(f"\n[YuE2 ComfyUI] Starting YuE2 one-shot generation on {target_device}...")
            print(f"[YuE2 ComfyUI] Mode: {cot} | Seed: {seed} | CFG: {cfg_scale}")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                bufsize=1
            )

            for line in iter(proc.stdout.readline, ""):
                line_str = line.strip()
                if line_str:
                    print(f"[YuE2] {line_str}")

            proc.stdout.close()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"YuE2 generation process failed with exit code {rc}. See console logs.")

        # Find generated audio file
        audio_file = None
        for ext in ("audio.flac", "audio.wav", "song.flac", "song.wav"):
            cand = out_dir / ext
            if cand.exists():
                audio_file = cand
                break

        if not audio_file:
            raise FileNotFoundError(f"Generation finished, but no audio file was found in {out_dir}")

        # Load ABC score if available
        abc_score = ""
        score_file = out_dir / "score.abc"
        if score_file.exists():
            abc_score = score_file.read_text(encoding="utf-8", errors="replace")

        # Load audio into ComfyUI AUDIO dict
        audio_np, sr = sf.read(str(audio_file), dtype="float32")

        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, np.newaxis, :]
        elif audio_np.ndim == 2:
            audio_np = audio_np.T[np.newaxis, :, :]

        waveform_tensor = torch.from_numpy(audio_np).float()
        audio_dict = {
            "waveform": waveform_tensor,
            "sample_rate": sr
        }

        print(f"[YuE2 ComfyUI] Generation successful! Created: {audio_file.name} ({waveform_tensor.shape[-1] / sr:.1f}s, {sr}Hz)")

        return (audio_dict, abc_score, str(audio_file), clean_lyrics, clean_style, int(seed))


class YuE2UnloadModel:
    """Utility node to immediately unload any resident YuE2 foundation model and free 100% of GPU VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "any_trigger": ("*", {"tooltip": "Optional trigger input wire from another node."})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "YuE2/Utilities"

    def unload(self, any_trigger=None):
        unload_worker()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return ("YuE2 model unloaded and GPU VRAM freed successfully.",)
