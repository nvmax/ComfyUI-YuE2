"""YuE2 Model Loader Node for ComfyUI."""

import os
from pathlib import Path
import torch
import folder_paths

# Register model folder paths with ComfyUI
yue2_path = os.path.join(folder_paths.models_dir, "yue2")
folder_paths.add_model_folder_path("yue2", yue2_path)

def get_yue2_models():
    models = []
    for folder_type in ["yue2", "checkpoints"]:
        for base_dir in folder_paths.get_folder_paths(folder_type):
            if not os.path.isdir(base_dir):
                continue
            for item in os.listdir(base_dir):
                full = os.path.join(base_dir, item)
                if os.path.isdir(full) and "vae" not in item.lower():
                    if os.path.exists(os.path.join(full, "model.safetensors")) or os.path.exists(os.path.join(full, "config.json")):
                        models.append(item)
    if not models:
        models = ["YuE2-3B", "m-a-p/YuE2-3B"]
    return sorted(list(set(models)))

def get_yue2_vaes():
    vaes = []
    for folder_type in ["vae", "yue2"]:
        for base_dir in folder_paths.get_folder_paths(folder_type):
            if not os.path.isdir(base_dir):
                continue
            for item in os.listdir(base_dir):
                full = os.path.join(base_dir, item)
                if os.path.isdir(full) and "vae" in item.lower():
                    if os.path.exists(os.path.join(full, "model.safetensors")) or os.path.exists(os.path.join(full, "config.json")):
                        vaes.append(item)
    if not vaes:
        vaes = ["YuE2-Vae", "m-a-p/YuE2-Vae"]
    return sorted(list(set(vaes)))

def resolve_model_path(model_name: str) -> str:
    if os.path.exists(model_name):
        return os.path.abspath(model_name)
    for folder_type in ["yue2", "checkpoints"]:
        for base in folder_paths.get_folder_paths(folder_type):
            p = os.path.join(base, model_name)
            if os.path.exists(p):
                return os.path.abspath(p)
    return model_name

def resolve_vae_path(vae_name: str) -> str:
    if os.path.exists(vae_name):
        p = os.path.abspath(vae_name)
        return os.path.dirname(p) if os.path.isfile(p) else p
    for folder_type in ["vae", "yue2"]:
        for base in folder_paths.get_folder_paths(folder_type):
            p = os.path.join(base, vae_name)
            if os.path.exists(p):
                return os.path.dirname(p) if os.path.isfile(p) else os.path.abspath(p)
    return vae_name

def ensure_model_downloaded(model_name: str) -> str:
    """Returns local path to model, automatically downloading from Hugging Face if missing."""
    path = resolve_model_path(model_name)
    if os.path.exists(path) and (
        os.path.exists(os.path.join(path, "model.safetensors")) or os.path.exists(os.path.join(path, "config.json"))
    ):
        return path

    repo_id = model_name if "/" in model_name else f"m-a-p/{model_name}"
    folder_name = model_name.split("/")[-1]
    target_dir = os.path.join(folder_paths.models_dir, "yue2", folder_name)

    if os.path.exists(os.path.join(target_dir, "model.safetensors")):
        return os.path.abspath(target_dir)

    print(f"\n[YuE2 ComfyUI] Model '{model_name}' not found locally.")
    print(f"[YuE2 ComfyUI] Downloading foundation model weights from Hugging Face ({repo_id}) to {target_dir}...")
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(target_dir, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            ignore_patterns=["assets/*", "examples/*", "*.whl", "*.png", "*.mp3"]
        )
        print(f"[YuE2 ComfyUI] Model download completed successfully: {target_dir}")
        return os.path.abspath(target_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to automatically download YuE2 model '{repo_id}': {e}")

def ensure_vae_downloaded(vae_name: str) -> str:
    """Returns local path to VAE, automatically downloading from Hugging Face if missing."""
    path = resolve_vae_path(vae_name)
    if os.path.exists(path) and (
        os.path.exists(os.path.join(path, "model.safetensors")) or os.path.exists(os.path.join(path, "config.json"))
    ):
        return path

    repo_id = vae_name if "/" in vae_name else f"m-a-p/{vae_name}"
    folder_name = vae_name.split("/")[-1]
    target_dir = os.path.join(folder_paths.models_dir, "vae", folder_name)

    if os.path.exists(os.path.join(target_dir, "model.safetensors")):
        return os.path.abspath(target_dir)

    print(f"\n[YuE2 ComfyUI] VAE '{vae_name}' not found locally.")
    print(f"[YuE2 ComfyUI] Downloading VAE weights from Hugging Face ({repo_id}) to {target_dir}...")
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(target_dir, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            ignore_patterns=["assets/*", "*.png"]
        )
        print(f"[YuE2 ComfyUI] VAE download completed successfully: {target_dir}")
        return os.path.abspath(target_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to automatically download YuE2 VAE '{repo_id}': {e}")

def sanitize_device(dev: str) -> str:
    if not dev or dev == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if dev.startswith("cuda:"):
        if not torch.cuda.is_available():
            return "cpu"
        try:
            idx = int(dev.split(":")[1])
            count = torch.cuda.device_count()
            if idx >= count:
                print(f"[YuE2] Notice: Requested '{dev}', but only {count} CUDA device(s) visible (e.g. via CUDA_VISIBLE_DEVICES). Falling back to 'cuda:0'.")
                return "cuda:0"
        except ValueError:
            return "cuda:0"
    return dev

DEVICES = [
    "auto",
    "cuda:0",
    "cuda:1",
    "cpu"
]

ATTENTION_BACKENDS = [
    "auto",
    "cudnn",
    "sage",
    "sdpa"
]

class YuE2ModelLoader:
    """Discovers and configures YuE2 foundation models and VAEs stored in ComfyUI's model directories."""

    @classmethod
    def INPUT_TYPES(cls):
        models = get_yue2_models()
        vaes = get_yue2_vaes()
        return {
            "required": {
                "model": (models, {"default": models[0]}),
                "vae": (vaes, {"default": vaes[0]}),
                "device": (DEVICES, {"default": "auto"}),
                "memory_budget_gib": ("INT", {"default": 30, "min": 12, "max": 64, "step": 1}),
            },
            "optional": {
                "attention_backend": (ATTENTION_BACKENDS, {"default": "auto"}),
                "ode_steps": ("INT", {"default": 32, "min": 12, "max": 64, "step": 4}),
            }
        }

    RETURN_TYPES = ("YUE2_MODEL", "STRING", "STRING")
    RETURN_NAMES = ("yue2_model", "model_path", "vae_path")
    FUNCTION = "load_model_config"
    CATEGORY = "YuE2/Loaders"

    def load_model_config(self, model, vae, device="auto", memory_budget_gib=30, attention_backend="auto", ode_steps=32):
        resolved_model = resolve_model_path(model)
        resolved_vae = resolve_vae_path(vae)
        target_device = sanitize_device(device)

        model_config = {
            "model_name": model,
            "vae_name": vae,
            "model_path": resolved_model,
            "vae_path": resolved_vae,
            "device": target_device,
            "memory_budget_gib": int(memory_budget_gib),
            "attention_backend": str(attention_backend),
            "ode_steps": int(ode_steps)
        }

        print(f"[YuE2 Model Loader] Selected Model: {model} -> {resolved_model}")
        print(f"[YuE2 Model Loader] Selected VAE:   {vae} -> {resolved_vae}")
        print(f"[YuE2 Model Loader] Target Device:  {target_device} | Memory Budget: {memory_budget_gib} GiB | Attention: {attention_backend} | ODE Steps: {ode_steps}")

        return (model_config, resolved_model, resolved_vae)
