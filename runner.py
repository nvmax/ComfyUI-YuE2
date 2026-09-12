"""
Standalone generation worker for YuE2 in ComfyUI.
Executed in a separate subprocess using ComfyUI's python executable
to isolate CUDA memory and ensure clean VRAM release after generation.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# Silence harmless PyTorch/package deprecation messages in the console
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

import torch

def sanitize_device(dev: str) -> str:
    if not dev or dev == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if dev.startswith("cuda:"):
        if not torch.cuda.is_available():
            print("[YuE2 Runner] CUDA is not available. Falling back to cpu.")
            return "cpu"
        try:
            idx = int(dev.split(":")[1])
            count = torch.cuda.device_count()
            if idx >= count:
                print(f"[YuE2 Runner] Notice: Requested '{dev}', but only {count} CUDA device(s) visible (e.g. via CUDA_VISIBLE_DEVICES). Falling back to 'cuda:0' ({torch.cuda.get_device_name(0)}).")
                return "cuda:0"
        except ValueError:
            return "cuda:0"
    return dev

# Proactive patch to ensure Windows PyTorch builds without USE_FLASH_ATTENTION
# seamlessly fall back to PyTorch's native SDPA CUDA graph, while propagating
# the user's chosen attention backend (auto, cudnn, sdpa, sage) to GraphAR.
_CURRENT_ATTENTION_BACKEND = "auto"

try:
    import yue2.cuda_graph as cg
    _orig_graph_init = cg.GraphAR.__init__

    def _patched_graph_init(self, model, prefixes, max_tokens, attention_backend="auto", **kwargs):
        global _CURRENT_ATTENTION_BACKEND
        if attention_backend == "auto" and _CURRENT_ATTENTION_BACKEND != "auto":
            attention_backend = _CURRENT_ATTENTION_BACKEND

        # Map non-standard backends to sdpa if not directly supported by GraphAR
        if attention_backend not in {"auto", "flash", "cudnn", "sdpa"}:
            attention_backend = "sdpa"

        if attention_backend == "auto":
            try:
                _q = torch.zeros(1, 1, 1, 64, device=model.device, dtype=model.dtype)
                _cu = torch.tensor([0, 1], dtype=torch.int32, device=model.device)
                torch.ops.aten._flash_attention_forward(_q[:, 0], _q[0], _q[0], _cu, _cu, 1, 1, 0.0, False, False)
            except RuntimeError:
                attention_backend = "sdpa"
        _orig_graph_init(self, model, prefixes, max_tokens, attention_backend=attention_backend, **kwargs)

    cg.GraphAR.__init__ = _patched_graph_init
except Exception:
    pass

# Proactive patch for Stage 2 (NAR / Audio Synthesis) attention chunking:
# Without FlashAttention on Windows, PyTorch SDPA falls back to materializing
# the full [heads, seq, seq] attention matrix. For long sequences (e.g., 8,000+ tokens),
# an unchunked block demands 12+ GiB in a single allocation and causes OOM.
# By enforcing a default query_chunk_size (1024), we activate yue2's native
# chunked attention loop, keeping peak attention VRAM to ~1 GiB while preserving
# 100% mathematical and acoustic equivalence.
try:
    import yue2.nar as nar
    _orig_nar_attention = nar.attention

    def _patched_nar_attention(q, k, v, *, causal=False, backend="sdpa", query_chunk_size=None):
        if query_chunk_size is None and q.device.type == "cuda":
            query_chunk_size = 1024
        return _orig_nar_attention(q, k, v, causal=causal, backend=backend, query_chunk_size=query_chunk_size)

    nar.attention = _patched_nar_attention

    if hasattr(nar, "CachedNAR"):
        _orig_cached_nar_init = nar.CachedNAR.__init__

        def _patched_cached_nar_init(self, model, chunk, attention="sdpa", query_chunk_size=None):
            if query_chunk_size is None:
                query_chunk_size = 1024
            _orig_cached_nar_init(self, model, chunk, attention=attention, query_chunk_size=query_chunk_size)

        nar.CachedNAR.__init__ = _patched_cached_nar_init
except Exception:
    pass

# Inter-stage VRAM garbage collection and cache flushing:
# When transitioning from Stage 1 (AR token generation) to Stage 2 (NAR audio synthesis),
# flush Python garbage and PyTorch CUDA allocator cache to ensure clean, maximized VRAM
# before audio synthesis prefill and flow-matching ODE steps begin.
try:
    import gc
    from yue2 import YuE2Pipeline
    _orig_pipeline_synthesize = YuE2Pipeline.synthesize

    def _patched_pipeline_synthesize(self, semantic, *, cancelled=None):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return _orig_pipeline_synthesize(self, semantic, cancelled=cancelled)

    YuE2Pipeline.synthesize = _patched_pipeline_synthesize
except Exception:
    pass

# Adaptive memory fraction safety:
# In an isolated runner subprocess, prevent torch.cuda.set_per_process_memory_fraction
# from prematurely choking the PyTorch allocator below available device headroom.
try:
    _orig_set_memory_fraction = torch.cuda.set_per_process_memory_fraction

    def _safe_set_memory_fraction(fraction, device=None):
        # Allow fraction to utilize available VRAM without artificial sub-device capping
        if fraction >= 0.80:
            fraction = min(float(fraction) + 0.10, 1.0)
        try:
            _orig_set_memory_fraction(fraction, device=device)
        except Exception:
            pass

    torch.cuda.set_per_process_memory_fraction = _safe_set_memory_fraction
except Exception:
    pass

def run_generation(args):
    global _CURRENT_ATTENTION_BACKEND

    # Load request JSON
    req_path = Path(args.request)
    if not req_path.exists():
        raise FileNotFoundError(f"Request file not found: {req_path}")
    
    data = json.loads(req_path.read_text(encoding="utf-8"))
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    target_device = sanitize_device(args.device)

    from yue2 import YuE2Pipeline
    from yue2.protocol import SongRequest, GenerationConfig

    ode_steps = int(data.get("ode_steps", args.ode_steps))
    attn_backend = str(data.get("attention_backend", args.attention_backend))
    _CURRENT_ATTENTION_BACKEND = attn_backend
    gen_config = GenerationConfig(ode_steps=ode_steps)

    loader = {
        "model": args.model,
        "vae": args.vae,
        "local_files_only": args.offline,
        "device": target_device,
        "memory_budget_gib": args.memory_budget_gib,
        "progress": True,
        "generation_config": gen_config,
    }

    print(f"[YuE2 Runner] Loading pipeline with model={args.model}, vae={args.vae}, device={target_device}, attention={attn_backend}, ode_steps={ode_steps}...")
    request = SongRequest(
        id=data.get("id", "song"),
        style=data.get("style", ""),
        lyrics=data.get("lyrics", ""),
        cot=data.get("cot", "full"),
        seed=int(data.get("seed", 42)),
        cfg_scale=float(data.get("cfg_scale", 1.0))
    )

    loader.pop('attention_backend', None)
    with YuE2Pipeline.from_pretrained(**loader) as pipe:
        print(f"[YuE2 Runner] Starting generation (mode: {request.cot}, seed: {request.seed}, ode_steps: {ode_steps})...")
        song = pipe(**request.to_dict())
        save_artifacts = bool(data.get("save_artifacts", False))
        if save_artifacts:
            receipt = song.save_artifacts(out_dir)
            print(f"[YuE2 Runner] Saved all artifacts: identity={receipt.get('identity')}, duration={receipt.get('audio_seconds', 0):.1f}s")
        else:
            audio_path = out_dir / "audio.flac"
            song.save(audio_path)
            if song.abc:
                (out_dir / "score.abc").write_text(song.abc, encoding="utf-8")
            duration = len(song.audio) / song.sample_rate
            print(f"[YuE2 Runner] Audio synthesized successfully: {audio_path.name} ({duration:.1f}s)")

def run_worker(args):
    global _CURRENT_ATTENTION_BACKEND
    _CURRENT_ATTENTION_BACKEND = str(args.attention_backend)
    target_device = sanitize_device(args.device)
    from yue2 import YuE2Pipeline
    from yue2.protocol import SongRequest, GenerationConfig

    loader = {
        "model": args.model,
        "vae": args.vae,
        "local_files_only": args.offline,
        "device": target_device,
        "memory_budget_gib": args.memory_budget_gib,
        "progress": True,
        "generation_config": GenerationConfig(ode_steps=args.ode_steps),
    }

    print(f"[YuE2 Worker] Loading pipeline into VRAM on {target_device} (attention={args.attention_backend})...", flush=True)
    loader.pop('attention_backend', None)
    with YuE2Pipeline.from_pretrained(**loader) as pipe:
        print("[YuE2 Worker] READY", flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line in ("QUIT", "EXIT", "UNLOAD"):
                print("[YuE2 Worker] Received exit command. Releasing VRAM...", flush=True)
                break
            try:
                task = json.loads(line)
                req_path = Path(task["request"])
                out_dir = Path(task["output"])
                out_dir.mkdir(parents=True, exist_ok=True)

                data = json.loads(req_path.read_text(encoding="utf-8"))
                if "attention_backend" in data:
                    _CURRENT_ATTENTION_BACKEND = str(data["attention_backend"])
                ode_steps = int(data.get("ode_steps", args.ode_steps))
                pipe.generation_config = GenerationConfig(ode_steps=ode_steps)

                request = SongRequest(
                    id=data.get("id", "song"),
                    style=data.get("style", ""),
                    lyrics=data.get("lyrics", ""),
                    cot=data.get("cot", "full"),
                    seed=int(data.get("seed", 42)),
                    cfg_scale=float(data.get("cfg_scale", 1.0))
                )

                print(f"[YuE2 Worker] Starting generation (mode: {request.cot}, seed: {request.seed}, ode_steps: {ode_steps})...", flush=True)
                song = pipe(**request.to_dict())
                save_artifacts = bool(data.get("save_artifacts", False))
                if save_artifacts:
                    receipt = song.save_artifacts(out_dir)
                    print(f"[YuE2 Worker] Saved all artifacts: identity={receipt.get('identity')}, duration={receipt.get('audio_seconds', 0):.1f}s", flush=True)
                else:
                    audio_path = out_dir / "audio.flac"
                    song.save(audio_path)
                    if song.abc:
                        (out_dir / "score.abc").write_text(song.abc, encoding="utf-8")
                    duration = len(song.audio) / song.sample_rate
                    print(f"[YuE2 Worker] Audio synthesized successfully: {audio_path.name} ({duration:.1f}s)", flush=True)
                print("[YuE2 Worker] COMPLETED", flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[YuE2 Worker] ERROR: {e}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YuE2 ComfyUI Generation Runner")
    parser.add_argument("--worker", action="store_true", help="Run as persistent background worker")
    parser.add_argument("--request", help="Path to request JSON (one-shot mode)")
    parser.add_argument("--output", help="Output directory (one-shot mode)")
    parser.add_argument("--model", required=True, help="Path or name of model")
    parser.add_argument("--vae", required=True, help="Path or name of VAE")
    parser.add_argument("--device", default="auto", help="Device (e.g. auto, cuda:0, cuda:1)")
    parser.add_argument("--memory-budget-gib", type=int, default=30, help="Memory budget in GiB")
    parser.add_argument("--offline", action="store_true", help="Only use local files")
    parser.add_argument("--ode-steps", type=int, default=32, help="NAR flow matching ODE steps (default: 32)")
    parser.add_argument("--attention-backend", default="auto", choices=["auto", "cudnn", "sage", "sdpa"], help="Attention backend")
    
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
    else:
        if not args.request or not args.output:
            parser.error("--request and --output are required when not running in --worker mode")
        run_generation(args)
