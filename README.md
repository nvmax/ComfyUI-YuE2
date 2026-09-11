# ComfyUI-YuE2: Neural Music Generation Suite 🎵🎸

A professional, redistributable custom node suite bringing the **YuE2 (m-a-p)** foundation music engine and **LM Studio** AI Co-Producer directly into **ComfyUI**.

Designed for zero external desktop dependencies, native ComfyUI model folder management, and lossless 48kHz stereo output with interactive in-graph audio playback.

---

## 📦 Installation

1. **Via ComfyUI-Manager**:
   - Search for `ComfyUI-YuE2` in ComfyUI Manager and click **Install**. ComfyUI-Manager will automatically install all dependencies listed in `requirements.txt`.

2. **Manual Git Clone**:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-repo/ComfyUI-YuE2.git
   ```
   Install the required dependencies using your ComfyUI Python environment:
   ```bash
   # In ComfyUI portable or venv:
   python_embeded\python.exe -m pip install -r custom_nodes/ComfyUI-YuE2/requirements.txt
   ```
   *(Or run `python custom_nodes/ComfyUI-YuE2/install.py` to auto-install the bundled offline wheel)*

---

## 📂 Standard Model Directory Structure

To use this node suite, place your model weights in the standard ComfyUI `models/` directory:

```text
ComfyUI/
└── models/
    ├── yue2/
    │   └── YuE2-3B/                  <-- YuE2 Foundation Model
    │       ├── model.safetensors     (7.26 GB weights)
    │       ├── config.json
    │       ├── generation_config.json
    │       ├── yue2_generation_config.json
    │       ├── modeling_yue2.py
    │       ├── qwen.tiktoken
    │       └── weights_manifest.json
    │
    └── vae/
        └── YuE2-Vae/                 <-- YuE2 Audio VAE
            ├── model.safetensors     (530 MB weights)
            ├── config.json
            ├── modeling_vae.py
            └── weights_manifest.json
```

> **Automatic Model Downloader**:
> - If `YuE2-3B` or `YuE2-Vae` are not found locally, the node automatically downloads the weights from Hugging Face (`m-a-p/YuE2-3B` and `m-a-p/YuE2-Vae`) directly into `ComfyUI/models/yue2/` and `ComfyUI/models/vae/` with live progress bars.
> - Any custom checkpoints placed in `models/yue2/` or `models/checkpoints/` will automatically appear in the node's dropdown selectors.

---

## 🎛️ Node Suite Reference

### 1. 🎵 `YuE2 Neural Song Generator` (`YuE2SongGenerator`)
- **Integrated Model & VAE Selection**: Directly selects models from `ComfyUI/models/yue2/` and VAEs from `models/vae/` with automatic downloading if missing.
- **Clean Output Architecture**:
  - Temporary synthesis scratch files are kept in ComfyUI's temp directory.
  - Generates zero disk clutter: eliminates redundant `latent.npy`, `semantic.npy`, `abc_tokens.npy`, and `config.json` dumps.
  - `save_intermediate_artifacts` (optional boolean, default `False`): Set to `True` only if you specifically want raw research latents saved in `output/YuE2/`.
- **Defensive Text Normalization & Sanitization**:
  - Automatically unescapes literal `\r\n` and `\n` line breaks and runs `clean_yue2_lyrics` on all incoming lyrics and style prompts.
  - Guarantees clean, beautifully formatted multiline paragraphs even when bypassing upstream studio nodes or typing directly.
- **Persistent Model Daemon (`keep_model_loaded`)**:
  - `True` (Default): Keeps YuE2 loaded in GPU VRAM between runs via an isolated background worker. Repeat generations start instantly without re-loading ~8 GB of weights (saves 10–15s every run!).
  - `False`: Automatically terminates the worker and releases 100% of GPU VRAM immediately after synthesis.
- **Attention Engine (`attention_backend`)**:
  - `auto`: Defaults to the fastest available backend (`cudnn` on RTX 40/50 series).
  - `cudnn`: Hardware-accelerated Blackwell / Ada Tensor Core SDPA (~0.068 ms/layer).
  - `sage`: SageAttention INT8 QK attention with custom Triton kernels (~0.22 ms/layer).
  - `sdpa`: PyTorch native Scaled Dot-Product Attention fallback.
- **Audio Diffusion Steps (`ode_steps`)**:
  - `32` (Default): Reference studio fidelity.
  - `24`: Fast mode (saves ~3.5s, 1.3x faster NAR synthesis).
  - `16`: Ultra-fast draft mode (saves ~7s, 2x faster NAR synthesis).
- **Seed Control**: Built-in seed integer widget with full `control_after_generate` support (**randomize**, **increment**, **decrement**, or **fixed** after each queue) and outputs the exact seed integer used.
- **CoT Generation Modes**: Chain-of-Thought planning (`full`, `melody`, `off`).
- **Outputs**: Native ComfyUI `AUDIO` tensor dictionary (`{"waveform": [1, 2, num_samples], "sample_rate": 48000}`), ABC score text, local audio file path, lyrics used, style prompt used, and seed.

### 2. 🎸 `YuE2 Style and Lyrics Studio` (`YuE2StyleAndLyricsStudio`)
- **Producer Layout**: Organized workflow order (`genre_preset`, `vocal_profile`, `bpm`, `intro_style`, `custom_style`, `lyrics`, `custom_instruments`, `extra_tags`).
- **Operating Modes**:
  - **`Custom / Keep Typed Style`**: Direct pass-through mode. Preserves your custom style prompt and typed lyrics exactly as written without inserting preset tags.
  - **`Custom / Keep Only Lyrics`**: Preserves your custom style prompt and keeps your sung lyric lines **100% untouched**, while automatically structuring section headers and vocal assignments based on the chosen vocal profile.
  - **34 Curated Genre Presets**: Instantly sets production-grade style prompts, instruments, and enriched section headers across Rock, Pop, Country, Hip-Hop/R&B, Metal, Electronic, and Orchestral styles.
- **Vocal Profile**: 11 vocal configurations (male/female tenors, baritones, basses, sopranos, altos, duets, trios, harmony groups, and pure instrumental). Automatically configures voice conditioning and vocal assignments.
- **BPM & Intro Controls**: Injects synchronized tempo and intro styles (`Instrumental Intro`, `Ambient Nature Intro`, `Immediate Vocal Entry`, `None`).
- **Outputs**: `style_prompt` (STRING), `lyrics` (STRING), `bpm` (INT).

### 3. ⚡ `YuE2 Quick Song Starters` (`YuE2InspirationPresets`)
- 8 instant, radio-ready song templates with synchronized acoustic styles, fully written producer-annotated lyrics, BPM, and CoT mode for 1-click testing.

### 4. 🤖 `YuE2 LLM Co-Producer & Polisher` (`YuE2LLMProducer`)
- **Multi-Provider Engine**:
  - **Local**: **LM Studio** (`http://localhost:1234/v1`), **Ollama** (`http://localhost:11434`).
  - **Cloud Providers**: **OpenAI** (GPT-4o, GPT-4.1, etc.), **Anthropic** (Claude 3.5/3.7 Sonnet, Haiku, Opus), **Google Gemini** (Gemini 2.5 Flash/Pro), **xAI Grok** (Grok 2/4.6), **DeepSeek** (deepseek-chat, deepseek-reasoner), and **OpenRouter** (universal proxy).
  - Automatically reads API keys from UI widget or environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROK_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`).
  - `custom_model`: Type any unlisted or local model tag to override the dropdown (e.g. `gemma-2-9b-it`, `qwen2.5:14b`, `claude-3-7-sonnet-20250219`).
  - `base_url`: Optional host/port override for LAN or custom server setups.
  - `bpm`: Integer tempo input (connectable from Style Studio or manual) automatically injected into LLM prompt cues, style strings, and lyric tempo headers.
- **Producer Actions**:
  - **`Generate Full Song Concept`**: Generates complete lyrics, style conditioning, suggested title, seed, and BPM from a concept. If input lyrics are connected, automatically preserves user lyrics **100% verbatim**.
  - **`Polish & Arrange Lyrics`**: Formats and structures input lyrics into performance-ready YuE2 scripts. Dynamically reads `System Prompt.txt` from the custom node directory if present.
  - **`Generate Style Prompt Only`**: Generates acoustic and instrumentation conditioning prompts matching your topic.
  - **`Generate Lyrics Only`**: Writes structured lyrics matching an existing style prompt.
- **Built-in Sanitizer**:
  - Automatically cleans literal `\n` newlines, converts raw field labels into clean bracketed cues, preserves all sub-tags, and enforces clean paragraph spacing.
- **Outputs**: `lyrics_out` (STRING), `style_out` (STRING), `title` (STRING), `suggested_seed` (INT), `bpm` (INT).

### 5. 💾 `YuE2 Audio Preview & Saver` (`YuE2AudioSavePreview`)
- **Interactive In-Graph Player**: HTML5 audio player widget embedded directly on the ComfyUI canvas for instant playback of generated songs.
- **Multiple Formats**:
  - **MP3**: Broadcast-quality MP3 encoding (`320k`, `256k`, `192k`, `128k`) via high-performance SoundFile / TorchAudio.
  - **FLAC**: Studio lossless (`PCM_24`, `PCM_16`).
  - **WAV**: Uncompressed (`FLOAT` 32-bit, `PCM_24`, `PCM_16`).
- Saves directly to `ComfyUI/output/YuE2/` or temporary preview directory.

### 6. 🎲 `YuE2 Seed Generator` (`YuE2Seed`)
- Dedicated seed utility node equipped with `control_after_generate` (**randomize**, **increment**, **decrement**, **fixed**).
- Can be wired directly into `YuE2SongGenerator` or shared across multiple nodes.

### 7. 🧹 `YuE2 Unload Model / Free VRAM` (`YuE2UnloadModel`)
- One-click utility node to immediately terminate any resident YuE2 background worker and free 100% of GPU VRAM on demand.
- Can be run standalone or triggered after generation workflows.

---

## 🚀 Hardware & Performance Guide

### GPU & VRAM Requirements
- **Weights Footprint**: YuE2-3B (~7.3 GB) + YuE2-Vae (~0.5 GB).
- **Minimum VRAM**: **10 GB – 12 GB** (e.g. RTX 3060 12GB, RTX 4070 12GB) allows standard song generation.
- **Recommended VRAM**: **16 GB+** (NVIDIA RTX 3090, 4080, 4090, 5080, 5090) provides full context headroom and instant generation.
- **Persistent Model Daemon (`keep_model_loaded: True`)**: Eliminates the ~10–15 second weight reloading delay between generations by maintaining an isolated GPU worker.

### Attention Backend by GPU Family (`attention_backend`)
- **`auto`** *(Default)*: Automatically selects the fastest backend available on your system.
- **`cudnn`**: **Recommended for RTX 40 & 50 Series** (Ada Lovelace & Blackwell, SM 89/90/120). NVIDIA cuDNN SDPA delivers attention decode in **0.068 ms** per layer (over 4x faster than standard math fallback).
- **`sage`**: Built-in support for `sageattention` INT8 QK quantization inside CUDA graphs (~0.22 ms/layer). Excellent for RTX 30 and 40 series.
- **`sdpa`**: Standard PyTorch Scaled Dot-Product Attention fallback for all other CUDA GPUs.

### Optimal Synthesis Settings
- **Diffusion Steps (`ode_steps`)**:
  - `32` (Default): Studio reference fidelity.
  - `24`: Recommended sweet spot (saves ~25%–30% synthesis time with virtually identical audio quality).
  - `16`: Ultra-fast draft preview mode (2x faster NAR synthesis).
- **CFG Scale (`cfg_scale`)**:
  - `1.0` (Default): Recommended. Runs a single autoregressive pass. Values `> 1.0` execute two parallel branches (conditional + negative prompt), doubling AR generation time.
- **TensorFloat-32 (TF32)**: Automatically enabled on Ampere, Ada, and Blackwell GPUs for 3x–5x faster FP32 operations during flow matching and VAE decoding.

---

## 📁 Included Workflows

1. **`example_workflows/01_yue2_quick_preset_workflow.json`**:
   `Quick Song Starters` ➔ `Song Generator` (with persistent VRAM cache) ➔ `Audio Saver & Preview` (MP3 320k). Instant 1-click test.
2. **`example_workflows/02_yue2_full_producer_studio_workflow.json`**:
   `Style and Lyrics Studio` + `LLM Co-Producer` (grouped with `Fast Groups Bypasser`) ➔ `Style & Lyrics Switch / Merger` (`Any Switch`) with two `CR Text` direct manual override boxes ➔ `Song Generator` (with Seed Generator) ➔ `Audio Saver & Preview` + `Unload Model / Free VRAM`. Allows seamless 1-click toggling between Studio AI mode and direct manual prompt/lyrics entry without any cable rewiring!
