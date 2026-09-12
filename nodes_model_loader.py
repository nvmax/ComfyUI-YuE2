"""YuE2 Model Discovery & Utilities for ComfyUI."""

import os
from pathlib import Path
import torch
import folder_paths

# Ensure audio_encoders and checkpoints paths are recognized
if "audio_encoders" not in folder_paths.folder_names_and_paths:
    audio_encoders_dir = os.path.join(folder_paths.models_dir, "audio_encoders")
    folder_paths.add_model_folder_path("audio_encoders", audio_encoders_dir)

def get_yue2_models():
    """Returns available checkpoints from ComfyUI/models/checkpoints/, prioritizing YuE2 safetensors."""
    all_ckpts = folder_paths.get_filename_list("checkpoints")
    yue2_ckpts = [c for c in all_ckpts if "yue2" in c.lower()]
    other_ckpts = [c for c in all_ckpts if "yue2" not in c.lower()]
    result = yue2_ckpts + other_ckpts
    if not result:
        result = ["yue2_3b_bf16.safetensors"]
    return result

def get_yue2_vaes():
    """Returns available VAEs from ComfyUI/models/vae/."""
    vaes = folder_paths.get_filename_list("vae")
    return vaes if vaes else ["embedded_in_checkpoint"]

def get_audio_encoders():
    """Returns available audio encoders from ComfyUI/models/audio_encoders/."""
    encoders = folder_paths.get_filename_list("audio_encoders")
    return encoders if encoders else ["sheetsage2_bf16.safetensors"]

def resolve_model_path(model_name: str) -> str:
    if not model_name:
        return ""
    if os.path.exists(model_name):
        return os.path.abspath(model_name)
    for folder_type in ["checkpoints", "yue2"]:
        for base in folder_paths.get_folder_paths(folder_type):
            p = os.path.join(base, model_name)
            if os.path.exists(p):
                return os.path.abspath(p)
    return model_name

def resolve_vae_path(vae_name: str) -> str:
    if not vae_name:
        return ""
    if os.path.exists(vae_name):
        p = os.path.abspath(vae_name)
        return os.path.dirname(p) if os.path.isfile(p) else p
    for folder_type in ["vae", "checkpoints", "yue2"]:
        for base in folder_paths.get_folder_paths(folder_type):
            p = os.path.join(base, vae_name)
            if os.path.exists(p):
                return os.path.dirname(p) if os.path.isfile(p) else os.path.abspath(p)
    return vae_name

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
                print(f"[YuE2] Notice: Requested '{dev}', but only {count} CUDA device(s) visible. Falling back to 'cuda:0'.")
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
    "sdpa"
]

class YuE2ModelLoader:
    """Discovers and configures YuE2 foundation checkpoints stored in ComfyUI's model directories."""

    @classmethod
    def INPUT_TYPES(cls):
        models = get_yue2_models()
        return {
            "required": {
                "checkpoint": (models, {"default": models[0]}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("checkpoint_name", "checkpoint_path")
    FUNCTION = "load_model_config"
    CATEGORY = "YuE2/Loaders"

    def load_model_config(self, checkpoint):
        resolved_model = resolve_model_path(checkpoint)
        print(f"[YuE2 Model Loader] Selected Checkpoint: {checkpoint} -> {resolved_model}")
        return (checkpoint, resolved_model)
