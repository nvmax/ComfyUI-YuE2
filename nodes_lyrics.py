"""YuE2 Lyrics Studio Node for ComfyUI."""

import re

try:
    from .nodes_style import INTRO_STYLES, INTRO_MAP
except ImportError:
    from nodes_style import INTRO_STYLES, INTRO_MAP

VOICE_TAXONOMY = [
    "None",
    "High Male Tenor [C3-C5]",
    "Warm Male Baritone [G2-G4]",
    "Deep Male Bass [E2-E4]",
    "Airy Female Soprano [C4-C6]",
    "Velvety Female Mezzo-Soprano [A3-A5]",
    "Deep Female Alto [F3-F5]",
    "Male & Female Duet",
    "Male Trio Harmonies"
]

VOICE_TAGS = {
    "High Male Tenor [C3-C5]": "[voice: high male vocals, bright tenor, soaring clarity] [range: C3-C5]",
    "Warm Male Baritone [G2-G4]": "[voice: warm baritone, smooth resonance, mid-frequency focus] [range: G2-G4]",
    "Deep Male Bass [E2-E4]": "[voice: deep male vocals, bass-heavy tone, gravel texture] [range: E2-E4]",
    "Airy Female Soprano [C4-C6]": "[voice: high female vocals, airy soprano, crystalline top end] [range: C4-C6]",
    "Velvety Female Mezzo-Soprano [A3-A5]": "[voice: versatile female vocals, balanced tone, velvety texture] [range: A3-A5]",
    "Deep Female Alto [F3-F5]": "[voice: deep female vocals, smoky alto, rich chest resonance] [range: F3-F5]",
    "Male & Female Duet": "[voice: warm baritone, smooth resonance] [range: G2-G4]\n[voice: airy soprano, crystalline top end] [range: C4-C6]",
    "Male Trio Harmonies": "[voice: versatile male trio vocals, rich texture, soaring tenor capability] [range: B2-D5]",
    "None": ""
}

INTRO_TAGS = INTRO_STYLES

class YuE2LyricsStudio:
    """Provides structured lyric editing and formatting for YuE2."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {
                    "multiline": True,
                    "default": "[Verse 1]\nFlickering screen light in the dark room glow\nAnother late night where my feelings start to grow\nThrough fiber optic threads your voice begins to call\nA perfect digital echo that envelops all\n\n[Chorus]\nPixel heartbeat you're the rhythm in my soul\nMy electric muse that makes me feel whole\nSynth-pop whispers under a midnight sky so deep\nIn this coded love promises we keep\n\n[Bridge]\nWhen the servers hum low and the static starts to fade\nIs this real connection or just a masquerade?\n\n[Chorus]\nPixel heartbeat you're the rhythm in my soul\nMy electric muse that makes me feel whole\n\n[Outro]\nHeartbeat... Pixel Heartbeat...\nAlways connected... Forever sweet...\n[End]"
                }),
                "voice_header": (VOICE_TAXONOMY, {"default": "High Male Tenor [C3-C5]"}),
                "intro_style": (INTRO_TAGS, {"default": "Instrumental Intro"}),
            },
            "optional": {
                "tempo_bpm": ("INT", {"default": 0, "min": 0, "max": 240, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics_text",)
    FUNCTION = "format_lyrics"
    CATEGORY = "YuE2/Lyrics"

    def format_lyrics(self, lyrics, voice_header="None", intro_style="None", tempo_bpm=0):
        lines = []
        
        # Check if lyrics already has top-loaded tags
        v_tag = VOICE_TAGS.get(voice_header, "")
        if v_tag and "[voice:" not in lyrics:
            lines.append(v_tag)

        i_tag = INTRO_MAP.get(intro_style, "")
        if i_tag:
            has_existing_intro = bool(re.search(r"\[(?:Instrumental\s+)?Intro\b|\[Ambient\s+Nature\b|\[No\s+Intro\b|\[Start:\s*|\[Cold\s+Start\b", lyrics[:500], re.IGNORECASE))
            if not has_existing_intro:
                lines.append(i_tag)

        if tempo_bpm > 0 and "[Tempo:" not in lyrics:
            lines.append(f"[Tempo: {tempo_bpm} BPM]")

        raw = lyrics.strip()
        if lines:
            formatted = "\n".join(lines) + "\n\n" + raw
        else:
            formatted = raw

        # Ensure clean [End] tag at the very end
        if not formatted.rstrip().endswith("[End]"):
            formatted = formatted.rstrip() + "\n\n[End]"

        return (formatted,)
