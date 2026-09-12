"""YuE2 Neural Music Generator Node for ComfyUI (Native High-Speed Engine).

Leverages ComfyUI core's native YuE2 implementation for 4x faster in-process
execution with zero subprocess overhead, FixedKV CUDA graph memory stability,
native KSampler diffusion, and direct .safetensors checkpoint loading.
"""

import os
import random
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

import folder_paths
import comfy.sd
import comfy.model_management
import comfy.utils
from comfy.text_encoders.yue2 import FRAMES_PER_SECOND
from comfy_extras.nodes_audio import vae_decode_audio
import nodes

from .nodes_model_loader import get_yue2_models
from .nodes_llm import clean_yue2_lyrics

_CACHED_CKPT_NAME = None
_CACHED_MODEL = None
_CACHED_CLIP = None
_CACHED_VAE = None

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


def get_or_load_checkpoint(ckpt_name: str):
    """Loads and caches model, clip, and vae from a ComfyUI checkpoint safetensors file."""
    global _CACHED_CKPT_NAME, _CACHED_MODEL, _CACHED_CLIP, _CACHED_VAE

    # Resolve legacy aliases or empty values
    available = folder_paths.get_filename_list("checkpoints")
    if not ckpt_name or ckpt_name in ("YuE2-3B", "m-a-p/YuE2-3B", "YuE2-Vae"):
        yue2_ckpts = [c for c in available if "yue2" in c.lower()]
        ckpt_name = yue2_ckpts[0] if yue2_ckpts else (available[0] if available else "yue2_3b_bf16.safetensors")

    if (
        _CACHED_CKPT_NAME == ckpt_name
        and _CACHED_MODEL is not None
        and _CACHED_CLIP is not None
        and _CACHED_VAE is not None
    ):
        return _CACHED_MODEL, _CACHED_CLIP, _CACHED_VAE

    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
    print(f"\n[YuE2 Native] Loading checkpoint '{ckpt_name}' ({ckpt_path})...")

    out = comfy.sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    _CACHED_MODEL, _CACHED_CLIP, _CACHED_VAE = out[0], out[1], out[2]
    _CACHED_CKPT_NAME = ckpt_name
    return _CACHED_MODEL, _CACHED_CLIP, _CACHED_VAE


def unload_native_models():
    """Unloads cached models and empties PyTorch VRAM cache."""
    global _CACHED_CKPT_NAME, _CACHED_MODEL, _CACHED_CLIP, _CACHED_VAE
    print("[YuE2 Native] Freeing resident YuE2 models and releasing VRAM...")
    _CACHED_CKPT_NAME = None
    _CACHED_MODEL = None
    _CACHED_CLIP = None
    _CACHED_VAE = None
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()


class YuE2SongGenerator:
    """Synthesizes complete stereo songs from style and lyrics using native ComfyUI acceleration."""

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = get_yue2_models()
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
                "cot": (["full", "melody", "off"], {
                    "default": "full",
                    "tooltip": "full: generates chord annotations and melody; melody: melody only; off: skips ABC notation."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed for ABC and music token generation."
                }),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.05}),
                "checkpoint": (checkpoints, {
                    "default": checkpoints[0],
                    "tooltip": "Select YuE2 safetensors checkpoint from ComfyUI/models/checkpoints/ (e.g. yue2_3b_bf16.safetensors or yue2_convrot_int8.safetensors)."
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model weights in RAM/VRAM cache between generations for instantaneous reruns."
                }),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Optional upstream MODEL connection (from CheckpointLoaderSimple)"}),
                "clip": ("CLIP", {"tooltip": "Optional upstream CLIP connection (from CheckpointLoaderSimple)"}),
                "vae": ("VAE", {"tooltip": "Optional upstream VAE connection (from CheckpointLoaderSimple)"}),
                "abc": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional custom ABC notation (e.g. from SheetSage2AudioToABC for covers or custom chord sheets)"}),
                "track_id": ("STRING", {"default": "comfy_song"}),
                "ode_steps": ("INT", {"default": 24, "min": 12, "max": 64, "step": 4, "tooltip": "Diffusion steps for audio synthesis (24=fast default, 32=reference fidelity, 16=draft)"}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpm_2"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "sgm_uniform"}),
                "max_duration": ("FLOAT", {"default": 360.0, "min": 10.0, "max": 360.0, "step": 5.0}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 3.0, "step": 0.05}),
                "save_intermediate_artifacts": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, saves a copy of the synthesized audio file to ComfyUI/output/YuE2/"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("audio", "abc_score", "audio_path", "lyrics_used", "style_used", "seed")
    FUNCTION = "generate_song"
    CATEGORY = "YuE2/Generation"

    def generate_song(
        self,
        style,
        lyrics,
        cot="full",
        seed=0,
        cfg_scale=1.0,
        checkpoint=None,
        keep_model_loaded=True,
        model=None,
        clip=None,
        vae=None,
        abc="",
        track_id="comfy_song",
        ode_steps=24,
        steps=None,
        sampler_name="dpm_2",
        scheduler="sgm_uniform",
        max_duration=360.0,
        temperature=1.0,
        top_p=0.95,
        top_k=100,
        repetition_penalty=1.2,
        save_intermediate_artifacts=False,
        **kwargs
    ):
        # Resolve models
        if model is None or clip is None or vae is None:
            ckpt_to_load = checkpoint or kwargs.get("model") or "yue2_3b_bf16.safetensors"
            loaded_model, loaded_clip, loaded_vae = get_or_load_checkpoint(ckpt_to_load)
            model = model or loaded_model
            clip = clip or loaded_clip
            vae = vae or loaded_vae

        clean_style = normalize_text(style)
        clean_lyrics = clean_yue2_lyrics(normalize_text(lyrics))

        if not clean_style.lower().startswith("english"):
            clean_style = f"English, {clean_style.lstrip(', ')}"

        if seed == 0 or seed is None:
            seed = random.randint(100000, 99999999)

        actual_steps = int(steps if steps is not None else ode_steps)

        # Step 1: ABC notation generation (if applicable)
        abc_text = normalize_text(abc) if abc else ""
        if not abc_text and cot in ("full", "melody"):
            print(f"\n[YuE2 Native] 1/3: Generating ABC score (Mode: {cot}, Seed: {seed})...")
            tokens = clip.tokenize(clean_style, lyrics=clean_lyrics, cot=cot, seed=seed, max_tokens=8192)
            ids = clip.generate(tokens, max_length=8192, temperature=0.7, top_p=0.9, top_k=30, repetition_penalty=1.005, seed=seed)
            abc_text = clip.decode(ids)
            print(f"[YuE2 Native] ABC score generated ({len(abc_text)} characters)")

        # Step 2: Music token generation & acoustic conditioning
        music_mode = "off" if not abc_text.strip() else cot
        max_frames = max(1, round(float(max_duration) * FRAMES_PER_SECOND))
        print(f"\n[YuE2 Native] 2/3: Generating music tokens & conditioning (Duration budget: {max_duration:.1f}s)...")
        tokens = clip.tokenize(
            clean_style,
            lyrics=clean_lyrics,
            cot=music_mode,
            seed=seed,
            abc=abc_text,
            max_tokens=max_frames,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        frames = conditioning[0][1]["yue2_frames"]
        seconds = frames / FRAMES_PER_SECOND
        print(f"[YuE2 Native] Conditioning ready: {frames} frames -> {seconds:.2f}s audio duration")

        # Step 3: Latent tensor allocation & KSampler diffusion
        print(f"\n[YuE2 Native] 3/3: Running acoustic diffusion ({actual_steps} steps, {sampler_name} / {scheduler})...")
        latent_tensor = torch.zeros(
            (1, 64, max(1, round(seconds * FRAMES_PER_SECOND))),
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        latent_dict = {"samples": latent_tensor, "type": "audio", "downscale_ratio_temporal": 1920}

        ksampler = nodes.KSampler()
        sampled_latents = ksampler.sample(
            model=model,
            seed=seed,
            steps=actual_steps,
            cfg=float(cfg_scale),
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=conditioning,
            negative=conditioning,
            latent_image=latent_dict,
            denoise=1.0,
        )[0]

        # Step 4: VAE Audio Decoding
        print("[YuE2 Native] Decoding acoustic latents into 44.1kHz stereo audio...")
        audio_output = vae_decode_audio(vae, sampled_latents, tile=1920, overlap=128)

        audio_path = ""
        if save_intermediate_artifacts:
            out_dir = Path(folder_paths.get_output_directory()) / "YuE2"
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            safe_id = "".join(c if c.isalnum() or c == "_" else "_" for c in track_id).strip("_") or "song"
            save_file = out_dir / f"{safe_id}_{timestamp}.wav"

            waveform = audio_output["waveform"].squeeze(0).cpu().numpy()
            if waveform.ndim == 2:
                waveform = waveform.T
            sf.write(str(save_file), waveform, audio_output["sample_rate"])
            audio_path = str(save_file)
            print(f"[YuE2 Native] Saved audio to: {audio_path}")

        if not keep_model_loaded:
            unload_native_models()

        print("[YuE2 Native] Generation finished successfully!\n")
        return (audio_output, abc_text, audio_path, clean_lyrics, clean_style, seed)


class YuE2UnloadModel:
    """Explicitly unloads resident YuE2 models and frees GPU VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_input": ("*", {"tooltip": "Pass through connection (e.g. connect output of generator here to run after generation)"})
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "free_vram"
    CATEGORY = "YuE2/Utilities"

    def free_vram(self, any_input):
        unload_native_models()
        return (any_input,)
