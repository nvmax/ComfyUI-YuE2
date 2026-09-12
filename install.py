"""
Setup and dependency installer for ComfyUI-YuE2.
Ensures yue2-infer wheel and required dependencies are installed in the ComfyUI environment.
"""

import subprocess
import sys
from pathlib import Path

def install():
    try:
        import yue2
        print("[YuE2 Setup] yue2 is already installed.")
        return
    except ImportError:
        pass

    wheel_dir = Path(__file__).parent / "wheels"
    wheels = list(wheel_dir.glob("yue2_infer*.whl"))
    if wheels:
        wheel_path = str(wheels[0])
        print(f"[YuE2 Setup] Installing {wheel_path} into ComfyUI Python environment...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", wheel_path])
        print("[YuE2 Setup] Successfully installed yue2-infer.")
    else:
        print("[YuE2 Setup] No local wheel found. Attempting to install from PyPI...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "yue2-infer"])
        print("[YuE2 Setup] Successfully installed yue2-infer.")

if __name__ == "__main__":
    install()
