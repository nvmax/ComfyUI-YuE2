# ComfyUI-YuE2: Advanced Neural Music Generation Suite 🎵🎸

A professional custom node suite integrating the **YuE2 (m-a-p)** foundation music engine and **LM Studio** AI Co-Producer directly into **ComfyUI**.

⚡ **Now Powered by ComfyUI Native YuE2 Acceleration (~4x Faster!)**
- **100% In-Process Execution**: Zero subprocesses, zero disk I/O latency, no temporary `.wav` files.
- **FixedKV Cache & CUDA Memory Graphs**: Eliminates tensor allocation overhead during token generation.
- **Native KSampler & Diffusion**: Leverages ComfyUI's native `dpm_2` / `sgm_uniform` diffusion engine with PyTorch SDPA/cuDNN.
- **Standard Checkpoints**: Loads single `.safetensors` files directly from `ComfyUI/models/checkpoints/` (including FP16/BF16 and INT8 quantized checkpoints).
- **Audio Cover Mode**: Seamless integration with `sheetsage2_bf16.safetensors` from `ComfyUI/models/audio_encoders/` to transcribe reference songs to ABC notation.

---

## 📦 Quick Installation

1. **Clone into `custom_nodes`**:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-repo/ComfyUI-YuE2.git
   ```

2. **Install standard dependencies**:
   ```bash
   ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
   ```
   *(No wheels to build, no external desktop dependencies, no CUDACXX errors).*

---

## 📂 Model Locations

Place your YuE2 weights in the standard ComfyUI model directories:

| Model | File Name | Destination Folder |
| :--- | :--- | :--- |
| **YuE2 Checkpoint** | `yue2_3b_bf16.safetensors` *(or `yue2_convrot_int8.safetensors`)* | `ComfyUI/models/checkpoints/` |
| **SheetSage2 Encoder (Covers)** | `sheetsage2_bf16.safetensors` | `ComfyUI/models/audio_encoders/` |

---

## 🚀 Workflows

This repository provides two workflows tailored for different user preferences:

### 1. 🎛️ Modular Studio Workflow (`yue2_full_producer_studio_workflow.json`)
Connects our comprehensive Producer Suite (`YuE2StyleAndLyricsStudio`, `YuE2LLMProducer`, `YuE2Seed`, `YuE2AudioSavePreview`) directly into ComfyUI's official modular nodes:
- **`CheckpointLoaderSimple`**: Loads `yue2_3b_bf16.safetensors` (DiT Model + LLaMA CLIP + Audio VAE).
- **`YuE2GenerateABC`**: Generates chord & melody ABC notation from style & lyrics.
- **`YuE2GenerateMusic`**: Produces acoustic conditioning and audio frame duration.
- **`EmptyYuE2LatentAudio`**: Allocates the 64-channel latent tensor.
- **`KSampler`**: High-speed diffusion (`dpm_2` / `sgm_uniform` at 32 steps).
- **`VAEDecodeAudioTiled`**: Decodes 64-channel latent directly into 44.1kHz stereo audio.
- **`SheetSage2AudioToABC` & `AudioEncoderLoader`**: Converts reference audio into ABC score for cover song creation.
- **`YuE2AudioSavePreview`**: Interactive browser playback and saving to MP3 320k, FLAC 24-bit, or WAV.

### 2. ⚡ All-In-One Studio Workflow (`example_workflows/02_yue2_full_producer_studio_workflow.json`)
Uses our upgraded **`YuE2 Neural Song Generator`** (`YuE2SongGenerator`) node:
- Executes the entire native ComfyUI pipeline inside a single node.
- Automatically detects and loads `yue2_3b_bf16.safetensors` from `checkpoints/`.
- 100% backward-compatible with older workflows while running at full 4x native speed!

---

## 🎛️ Node Suite Reference

### 1. 🎵 `YuE2 Neural Song Generator` (`YuE2SongGenerator`)
- **Native In-Process Engine**: Runs ComfyUI's native YuE2 engine in memory without spawning external worker processes.
- **Dropdown Checkpoint Discovery**: Automatically lists checkpoints in `models/checkpoints/`, prioritizing YuE2 safetensors.
- **Modular Connections (Optional)**: Accepts upstream `model`, `clip`, `vae`, and `abc` inputs if you want to connect external nodes or SheetSage2 scores.
- **Options**: Supports `cot` ("full", "melody", "off"), `steps` (default 32), `cfg_scale`, `sampler_name`, `scheduler`, and `max_duration`.

### 2. 🎨 `YuE2 Style & Lyrics Studio` (`YuE2StyleAndLyricsStudio`)
- Visual UI with curated genre presets, vocal profiles (Male/Female/Emotive/Tenor/Baritone), BPM sliders, intro styles, instruments, and structured section tags (`[Verse 1]`, `[Chorus]`, etc.).

### 3. 🤖 `YuE2 LLM Co-Producer & Polisher` (`YuE2LLMProducer`)
- Connects directly to local **LM Studio**, **Ollama**, or cloud APIs (**OpenAI**, **Anthropic**, **Gemini**, etc.).
- Formats lyrics, refines rhymes, adds vocal taxonomy, and tailors song concepts specifically for the YuE2 foundation model.

### 4. 💾 `YuE2 Audio Preview & Saver` (`YuE2AudioSavePreview`)
- In-graph interactive waveform/audio player with direct saving to MP3 (up to 320k), FLAC (16/24-bit), and WAV.
- Clean folder organization under `ComfyUI/output/YuE2/` with customizable track naming.

### 5. 🎲 `YuE2 Seed Generator` (`YuE2Seed`)
- Master random seed generator with `fixed`, `increment`, `decrement`, and `randomize` controls.

---

## 📄 License
Apache-2.0. YuE2 model weights are released by M-A-P under their respective model license.
