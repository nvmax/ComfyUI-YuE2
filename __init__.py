"""
ComfyUI-YuE2: Advanced Neural Music Generation Suite for ComfyUI.
High-Speed In-Process Native YuE2 Acceleration with LM Studio & Local LLM Co-Producer.
"""

from .nodes_style import YuE2StyleAndLyricsStudio, YuE2AcousticStyle
from .nodes_lyrics import YuE2LyricsStudio
from .nodes_presets import YuE2InspirationPresets
from .nodes_llm import YuE2LLMProducer, init_server_routes
from .nodes_generator import YuE2SongGenerator, YuE2UnloadModel
from .nodes_audio import YuE2AudioSavePreview
from .nodes_seed import YuE2Seed

# Register custom API routes (e.g. /yue2/models) with ComfyUI's server
init_server_routes()

NODE_CLASS_MAPPINGS = {
    "YuE2SongGenerator": YuE2SongGenerator,
    "YuE2StyleAndLyricsStudio": YuE2StyleAndLyricsStudio,
    "YuE2AcousticStyle": YuE2AcousticStyle,  # Legacy alias
    "YuE2LyricsStudio": YuE2LyricsStudio,    # Legacy alias
    "YuE2InspirationPresets": YuE2InspirationPresets,
    "YuE2LLMProducer": YuE2LLMProducer,
    "YuE2AudioSavePreview": YuE2AudioSavePreview,
    "YuE2Seed": YuE2Seed,
    "YuE2UnloadModel": YuE2UnloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2SongGenerator": "YuE2 Neural Song Generator (Native)",
    "YuE2StyleAndLyricsStudio": "YuE2 Style and Lyrics Studio",
    "YuE2InspirationPresets": "YuE2 Quick Song Starters",
    "YuE2LLMProducer": "YuE2 LLM Co-Producer & Polisher",
    "YuE2AudioSavePreview": "YuE2 Audio Preview & Saver",
    "YuE2Seed": "YuE2 Seed Generator",
    "YuE2UnloadModel": "YuE2 Unload Model / Free VRAM",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
