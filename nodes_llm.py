"""YuE2 Multi-Provider LLM Co-Producer & Lyric Polisher Node for ComfyUI."""

import os
import json
import re
import urllib.request
import urllib.parse
import urllib.error
from typing import Optional

def clean_json_text(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1].strip()
    return text

def clean_yue2_lyrics(text: str) -> str:
    """Sanitizes lyrics: unescapes literal newlines and quotes, converts raw field labels to clean musical cues, and formats clean paragraph spacing."""
    if not text:
        return ""

    # 1. Unescape literal \r\n, \n, \r, \t, and quotation artifacts
    text = str(text).replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "").replace("\\t", " ").replace('\\"', '"')

    # 2. Convert awkward field prefixes like 'Instrumentation details:' to clean bracketed musical cues
    text = re.sub(r"\[Instrumentation(?:\s+details)?:\s*", "[", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Vocal\s+delivery(?:\s+style)?:\s*", "[", text, flags=re.IGNORECASE)
    text = re.sub(r"\[(?:Production|Arrangement)(?:\s+details)?:\s*", "[", text, flags=re.IGNORECASE)

    # 3. Clean up line formatting and ensure paragraph spacing
    raw_lines = [l.strip() for l in text.splitlines()]
    processed_lines = []

    for line in raw_lines:
        if not line:
            if processed_lines and processed_lines[-1] != "":
                processed_lines.append("")
            continue

        # If this is a main section header (e.g. [Verse 1...], [Chorus...], [Pre-Chorus...], [Bridge...], [Outro...]),
        # ensure there is an empty line before it (unless it's top technical anchors)
        is_major_section = bool(re.match(r"^\[(Verse|Pre-Chorus|Chorus|Hook|Bridge|Outro)", line, re.IGNORECASE))
        if is_major_section and processed_lines and processed_lines[-1] != "":
            processed_lines.append("")

        processed_lines.append(line)

    result = "\n".join(processed_lines).strip()
    if not result.endswith("[End]"):
        result += "\n\n[End]"
    return result

def format_lyrics_payload(val) -> str:
    """Recursively parses and normalizes lyrics payloads (strings, dicts of sections, lists)."""
    if not val:
        return ""
    if isinstance(val, str):
        s = val.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "").replace("\\t", " ").replace('\\"', '"')
        return s.strip()
    if isinstance(val, dict):
        sections = []
        for k, v in val.items():
            if str(k).lower() in ("title", "style", "cot", "suggested_seed", "notes", "bpm"):
                continue
            if isinstance(v, dict) and ("text" in v or "lyrics" in v or "content" in v):
                header = v.get("section") or v.get("name") or v.get("header") or k
                body_text = v.get("text") or v.get("lyrics") or v.get("content") or ""
                header = str(header).replace("_", " ").strip()
                if not header.startswith("["):
                    header = f"[{header.title()}]"
                if not header.endswith("]"):
                    header = f"{header}]"
                sections.append(f"{header}\n{format_lyrics_payload(body_text)}".strip())
            else:
                header = str(k).replace("_", " ").strip()
                if not header.startswith("["):
                    header = f"[{header.title()}]"
                if not header.endswith("]"):
                    header = f"{header}]"
                body = format_lyrics_payload(v)
                sections.append(f"{header}\n{body}".strip())
        return "\n\n".join(s for s in sections if s)
    if isinstance(val, (list, tuple)):
        items = []
        for x in val:
            if isinstance(x, dict) and ("section" in x or "part" in x or "header" in x):
                header = x.get("section") or x.get("part") or x.get("header") or "Section"
                body_text = x.get("lyrics") or x.get("text") or x.get("content") or ""
                header = str(header).replace("_", " ").strip()
                if not header.startswith("["):
                    header = f"[{header.title()}]"
                if not header.endswith("]"):
                    header = f"{header}]"
                items.append(f"{header}\n{format_lyrics_payload(body_text)}".strip())
            else:
                items.append(format_lyrics_payload(x))
        return "\n\n".join(i for i in items if i)
    return str(val).strip()

def enforce_english_style(style) -> str:
    if not style:
        return "English, modern melodic, 100 BPM"
    if isinstance(style, dict):
        parts = [str(v).strip() for v in style.values() if v]
        style = ", ".join(parts)
    elif isinstance(style, (list, tuple)):
        parts = [str(v).strip() for v in style if v]
        style = ", ".join(parts)
    s = str(style).strip().replace("\\n", " ")
    # Strip any Korean / Korean/English / Japanese / etc language tags at the start
    s = re.sub(r"^(?:Korean\s*/\s*English|English\s*/\s*Korean|Korean|Japanese|Mandarin|Spanish|French|German)\s*,\s*", "", s, flags=re.IGNORECASE)
    # Replace any standalone occurrences of Korean/English or Korean
    s = re.sub(r"\bKorean\s*/\s*English\b", "English", s, flags=re.IGNORECASE)
    s = re.sub(r"\bKorean\b", "English", s, flags=re.IGNORECASE)
    if not s.lower().startswith("english"):
        s = f"English, {s.lstrip(', ')}"
    return s

def extract_clean_lyrics(resp: str) -> str:
    text = str(resp).strip()
    m_fence = re.search(r"```(?:text|lyrics|markdown)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m_fence and not m_fence.group(1).strip().startswith("{"):
        text = m_fence.group(1).strip()

    if text.startswith("{") or '"polished_lyrics"' in text or '"lyrics"' in text:
        try:
            fixed = re.sub(r'("polished_lyrics"\s*:\s*)\{([^}]+)\}', r'\1[\2]', text)
            m_json = re.search(r"(\{.*?\})", fixed, re.DOTALL)
            candidate = m_json.group(1) if m_json else fixed
            data = json.loads(candidate)
            if isinstance(data, dict):
                val = data.get("polished_lyrics") or data.get("lyrics")
                if val is not None:
                    text = format_lyrics_payload(val)
        except Exception:
            m_block = re.search(r'"(?:polished_lyrics|lyrics)"\s*:\s*"(.*)', text, re.DOTALL)
            if m_block:
                remainder = m_block.group(1)
                m_end = re.search(r'"\s*(?:,\s*"(?:bpm|suggested_seed|cot|notes|title|style)"|\s*\})', remainder)
                raw_lyr = remainder[:m_end.start()] if m_end else remainder
                text = raw_lyr.replace('\\"', '"').replace('\\n', '\n').replace('\\t', ' ')

    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)

    first_bracket = text.find("[")
    if first_bracket > 0:
        preamble = text[:first_bracket].strip()
        if any(w in preamble.lower() for w in ["here is", "here's", "sure", "certainly", "below is", "script", "lyrics"]):
            text = text[first_bracket:]

    end_tag = "[End]"
    end_idx = text.rfind(end_tag)
    if end_idx != -1:
        text = text[:end_idx + len(end_tag)]

    return clean_yue2_lyrics(text)

ACTIONS = [
    "Polish & Arrange Lyrics",
    "Generate Full Song Concept",
    "Optimize Acoustic Style String"
]

PROVIDERS = [
    "LMStudio",
    "Ollama",
    "openai",
    "anthropic",
    "google",
    "grok",
    "deepseek",
    "openrouter",
]

DEFAULT_BASE_URLS = {
    "LMStudio": "http://localhost:1234/v1",
    "Ollama": "http://localhost:11434",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "google": "https://generativelanguage.googleapis.com",
    "grok": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
    "openrouter": "https://openrouter.ai/api/v1",
}

DEFAULT_MODELS = {
    "LMStudio": "gemma-4-e4b-it",
    "Ollama": "qwen2.5:7b",
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
    "google": "gemini-2.5-flash",
    "grok": "grok-2-latest",
    "deepseek": "deepseek-chat",
    "openrouter": "openai/gpt-4o-mini",
}

PROVIDER_MODELS = {
    "LMStudio": [
        "gemma-4-e4b-it",
        "qwen2.5-7b-instruct",
        "qwen2.5-14b-instruct",
        "llama-3.1-8b-instruct",
        "mistral-7b-instruct",
    ],
    "Ollama": [
        "qwen2.5:7b",
        "qwen2.5:14b",
        "llama3.1:8b",
        "mistral",
        "gemma2:9b",
        "deepseek-r1:8b",
        "deepseek-r1:14b",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "o4-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-sonnet-4-6",
        "claude-opus-4-8",
    ],
    "google": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-3.5-flash",
        "gemini-3.1-flash",
    ],
    "grok": [
        "grok-2-latest",
        "grok-beta",
        "grok-4.6",
        "grok-4.5",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    "openrouter": [
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.5-flash",
        "deepseek/deepseek-chat",
        "meta-llama/llama-3.1-70b-instruct",
        "qwen/qwen-2.5-72b-instruct",
    ],
}

ALL_MODELS = []
for p in PROVIDERS:
    for m in PROVIDER_MODELS.get(p, []):
        if m not in ALL_MODELS:
            ALL_MODELS.append(m)

def resolve_api_key(provider: str, api_key: str) -> str:
    key = str(api_key or "").strip()
    if key:
        return key

    prov = provider.lower()
    env_map = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "grok": ["GROK_API_KEY", "XAI_API_KEY"],
        "deepseek": ["DEEPSEEK_API_KEY"],
        "openrouter": ["OPENROUTER_API_KEY"],
    }
    for env_var in env_map.get(prov, []):
        val = os.environ.get(env_var, "").strip()
        if val:
            return val
    return ""

def resolve_base_url(provider: str, base_url: str) -> str:
    u = str(base_url or "").strip().rstrip("/")
    if u:
        return u
    for k, v in DEFAULT_BASE_URLS.items():
        if k.lower() == provider.lower():
            return v
    return "http://localhost:1234/v1"

def resolve_model(provider: str, selected_model: str, custom_model: str) -> str:
    cust = str(custom_model or "").strip()
    if cust:
        return cust
    prov_key = None
    for k in PROVIDERS:
        if k.lower() == provider.lower():
            prov_key = k
            break
    if not prov_key:
        return selected_model or "gpt-4o"
    allowed = PROVIDER_MODELS.get(prov_key, [])
    if selected_model in allowed:
        return selected_model
    return DEFAULT_MODELS.get(prov_key, allowed[0] if allowed else selected_model)


def resolve_bpm(bpm, style_context: str = "", input_text: str = "") -> int:
    try:
        if bpm is not None and int(bpm) > 0:
            return int(bpm)
    except Exception:
        pass
    m = re.search(r"\b(\d{2,3})\s*BPM\b", str(style_context or ""), re.IGNORECASE)
    if m:
        return int(m.group(1))
    m2 = re.search(r"\[Tempo:\s*(\d{2,3})\s*BPM\]", str(input_text or ""), re.IGNORECASE)
    if m2:
        return int(m2.group(1))
    return 102

def ensure_tempo_tag(lyrics: str, bpm: int) -> str:
    if not lyrics or bpm <= 0:
        return lyrics
    if re.search(r"\[Tempo:\s*\d{2,3}\s*BPM\]", lyrics, re.IGNORECASE):
        return lyrics
    lines = lyrics.splitlines()
    insert_idx = 0
    for i, line in enumerate(lines[:6]):
        s = line.strip().lower()
        if s.startswith("[voice:") or s.startswith("[range:") or s.startswith("[instrumental intro") or s.startswith("[ambient nature") or s.startswith("[intro"):
            insert_idx = i + 1
    tempo_line = f"[Tempo: {bpm} BPM]"
    lines.insert(insert_idx, tempo_line)
    return "\n".join(lines)

def ensure_end_tag(lyrics: str) -> str:
    if not lyrics:
        return "[End]"
    s = lyrics.rstrip()
    if not s.endswith("[End]"):
        return s + "\n\n[End]"
    return s

SYSTEM_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "System Prompt.txt")

def get_polish_sys_prompt() -> str:
    """Loads the master System Prompt.txt from disk if present, ensuring complete fidelity to producer instructions."""
    if os.path.exists(SYSTEM_PROMPT_FILE):
        try:
            with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
                c = f.read().strip()
                if c:
                    return c
        except Exception as e:
            print(f"[YuE2] Notice: Could not read System Prompt.txt ({e})")
    return POLISH_SYS_PROMPT

POLISH_SYS_PROMPT = """You are the world's most advanced Music Prompt Engineer and Audio Architect for YuE2.
Your mission is to format user lyrics into a precise, radio-ready technical performance script.

══ CARDINAL RULE: 100% VERBATIM LYRIC PRESERVATION ══
Keep all existing sung lyrics 100% VERBATIM. Do NOT alter, rewrite, drop, or summarize words.
Your job is to architect the structural tags, descriptive sub-tags, and performance cues around their exact words.

══ MANDATORY SCRIPT ARCHITECTURE (2026 BEST PRACTICES) ══

1. TOP-LOADED ANCHORS (At the very top of lyrics):
- [voice: taxonomic description, tone, texture] [range: X#-Y#]
- [Instrumental Intro: Detailed description of starting instruments, mood, texture, and SFX] [Energy: Low]
- [Tempo: XX BPM]

2. MANDATORY SECTION TAGGING (PRECISION ARCHITECTURE):
Every structural section MUST include descriptive sub-tags immediately below the section header.
Format:
[Section Name - Vocal Assignment]
[Instrumentation details] [Vocal delivery style] [Energy level]

Example:
[Verse 1 - Male Vocal]
[Sparse acoustic guitar fingerpicking] [Intimate, close-mic vocal delivery] [Energy: Low]
The clock on the wall is a liar tonight
Spinning too fast while we chase the light
I'm holding your hand like a lifeline in June
The only thing quiet in a world out of tune

[Pre-Chorus 1 - Male Vocal]
[Subtle drum machine pulse enters] [Vocal: Softly, building intensity] [Add Tension]
(softly) Every second feels like it's caught on hold
A story written that refuses to grow old

[Chorus 1 - Male Vocal]
[Full orchestral swell with driving drums] {layered vocals: 'grace'} [Energy: Maximum, Euphoric Release]
Oh I'm slowing it down for the one that I love
Like the earth's standing still while the stars crash above
It's a beautiful weight it's a terrifying grace
I'm finally home in the light of your grace
(In your grace)

3. SYNTAX HIERARCHY:
- [BRACKETS] = Non-sung elements: structural section tags, instrumentation details, vocal delivery notes, production instructions, energy markers.
- (PARENTHESES) = Actual sung vocal delivery nuances: (softly), (urgent), (adlibs: 'yeah!'), (whispered).
- {BRACES} = Harmonies & backup layers: {layered vocals: 'word'}, {backup vocals: 'word'}, {harmony: 'word'}.
- Separate every section with an empty line. Never bunch lines into a wall of text.

4. OUTRO & LANDING:
- [OUTRO - REVERB TAIL FADE OUT, long decay, spacious hall reverb, fading echoes] [Energy: Decrescendo]
- (Slowing it down... softly)
- (For the one I love... fading)
- [End] (Must ALWAYS be on its own line)

══ OUTPUT FORMAT ══
Output ONLY the raw content of the technical lyrics script. ZERO Markdown formatting (no **, no ###, no backticks)."""

CONCEPT_SYS_PROMPT = """You are the Lead Music Producer, Songwriter, and Audio Architect for YuE2.
Your mission is to generate a complete, production-ready radio song concept with full performance script tagging.

══ MANDATORY SCRIPT ARCHITECTURE (2026 BEST PRACTICES) ══

1. TOP-LOADED ANCHORS (At the very top of lyrics):
- [voice: taxonomic description, tone, texture] [range: X#-Y#]
- [Instrumental Intro: Detailed description of starting instruments, mood, texture, and SFX] [Energy: Low]
- [Tempo: XX BPM]

2. MANDATORY SECTION TAGGING (PRECISION ARCHITECTURE):
Every structural section MUST include descriptive sub-tags immediately below the section header.
Format:
[Section Name - Vocal Assignment]
[Instrumentation details] [Vocal delivery style] [Energy level]

Example:
[Verse 1 - Male Vocal]
[Sparse acoustic guitar fingerpicking] [Intimate, close-mic vocal delivery] [Energy: Low]
The clock on the wall is a liar tonight
Spinning too fast while we chase the light
I'm holding your hand like a lifeline in June
The only thing quiet in a world out of tune

[Pre-Chorus 1 - Male Vocal]
[Subtle drum machine pulse enters] [Vocal: Softly, building intensity] [Add Tension]
(softly) Every second feels like it's caught on hold
A story written that refuses to grow old

[Chorus 1 - Male Vocal]
[Full orchestral swell with driving drums] {layered vocals: 'grace'} [Energy: Maximum, Euphoric Release]
Oh I'm slowing it down for the one that I love
Like the earth's standing still while the stars crash above
It's a beautiful weight it's a terrifying grace
I'm finally home in the light of your grace
(In your grace)

3. SYNTAX HIERARCHY:
- [BRACKETS] = Non-sung elements: structural section tags, instrumentation details, vocal delivery notes, production instructions, energy markers.
- (PARENTHESES) = Actual sung vocal delivery nuances: (softly), (urgent), (adlibs: 'yeah!'), (whispered).
- {BRACES} = Harmonies & backup layers: {layered vocals: 'word'}, {backup vocals: 'word'}, {harmony: 'word'}.
- Always use single quotes (') inside braces and brackets so JSON remains valid.
- Separate every section with an empty line. Never bunch lines into a wall of text.

4. OUTRO & LANDING:
- [OUTRO - REVERB TAIL FADE OUT, long decay, spacious hall reverb, fading echoes] [Energy: Decrescendo]
- (Slowing it down... softly)
- (For the one I love... fading)
- [End] (Must ALWAYS be on its own line)

5. VERBATIM PRESERVATION:
If user lyrics are provided in the input, you MUST preserve all existing sung words 100% VERBATIM! Keep their exact words and architect the rich structural headers, sub-tags, and cues around them.

══ OUTPUT FORMAT ══
Output ONLY valid raw JSON (no markdown fences, no conversational text) with these keys:
{
  "title": "Creative Track Title",
  "style": "English, [Vocal Character], [Genre & Sub-genres], [Key Instruments & Drums], [Sonic Space / Master], [BPM] BPM",
  "lyrics": "The full technical performance script following the architecture above with top anchors, section headers, sub-tags, and sung lyrics separated by blank lines and ending with [End]",
  "bpm": 121,
  "suggested_seed": 42
}"""

STYLE_SYS_PROMPT = """You are the Lead Audio Mastering Engineer and Acoustic Prompt Architect for YuE2.
Your mission is to craft an optimal conditioning style string for YuE2 music synthesis.

══ RULES ══
1. Must strictly begin with 'English, '.
2. Format: English, [Vocal Character & Range], [Genre & Sub-genres], [Key Instruments & Drums], [Sonic Space / Production Aesthetic], [BPM] BPM
3. Use concrete, specific audio terminology (e.g., analog tape warmth, 808 sub-bass, Roland Juno synths, cathedral reverb, radio-ready sparkle).

══ OUTPUT FORMAT ══
Output ONLY raw JSON: {"style_string": "English, ..."}. No markdown fences, no conversational text."""

def robust_parse_concept_json(text: str, default_style: str = "", default_bpm: int = 102) -> dict:
    """Robustly parses LLM JSON responses, tolerating unescaped inner double quotes or markdown fences."""
    clean_text = clean_json_text(text)
    try:
        return json.loads(clean_text)
    except Exception:
        pass

    out = {}
    m_title = re.search(r'"title"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', clean_text)
    if m_title:
        out["title"] = m_title.group(1).replace('\\"', '"')
    else:
        out["title"] = "Generated Track"

    m_style = re.search(r'"style"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', clean_text)
    if m_style:
        out["style"] = m_style.group(1).replace('\\"', '"')
    else:
        out["style"] = default_style or "English, pop, 102 BPM"

    m_bpm = re.search(r'"bpm"\s*:\s*(\d+)', clean_text)
    if m_bpm:
        out["bpm"] = int(m_bpm.group(1))

    m_seed = re.search(r'"suggested_seed"\s*:\s*(\d+)', clean_text)
    if m_seed:
        out["suggested_seed"] = int(m_seed.group(1))
    else:
        out["suggested_seed"] = 42

    m_lyrics = re.search(r'"lyrics"\s*:\s*"(.*)', clean_text, re.DOTALL)
    if m_lyrics:
        remainder = m_lyrics.group(1)
        m_end = re.search(r'"\s*(?:,\s*"(?:bpm|suggested_seed|cot|notes|title|style)"|\s*\})', remainder)
        raw_lyr = remainder[:m_end.start()] if m_end else remainder
        raw_lyr = raw_lyr.replace('\\"', '"').replace('\\n', '\n').replace('\\t', ' ')
        out["lyrics"] = raw_lyr

    return out

class YuE2LLMProducer:
    """Multi-Provider AI Music Co-Producer: generates complete song concepts, polishes verbatim lyrics, and crafts acoustic style strings."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (ACTIONS, {"default": "Polish & Arrange Lyrics"}),
                "provider": (PROVIDERS, {"default": "LMStudio"}),
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": "A high-energy synth-pop song about escaping into the night",
                    "placeholder": "Enter lyrics to polish OR enter a song concept/theme"
                }),
                "model": (ALL_MODELS, {"default": "gemma-4-e4b-it"}),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API Key for cloud providers (OpenAI, Anthropic, Google, Grok, DeepSeek, OpenRouter). Leave empty for LM Studio/Ollama or if set in environment variables."
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "Type any custom or unlisted model identifier (e.g. qwen2.5:14b, claude-3-7-sonnet, deepseek-r1) to override the dropdown."
                }),
                "base_url": ("STRING", {
                    "default": "",
                    "placeholder": "Default: auto per provider. Override here if using custom port or remote host."
                }),
                "style_context": ("STRING", {
                    "multiline": False,
                    "default": "80s Retro Synthwave, 118 BPM",
                    "placeholder": "Optional music genre/style context"
                }),
                "bpm": ("INT", {
                    "default": 102,
                    "min": 0,
                    "max": 240,
                    "step": 1,
                    "tooltip": "Target tempo in BPM (0 = auto-extract from style context). Passed directly into the LLM prompt, style prompt, and lyric tempo headers."
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 128, "max": 8192, "step": 128}),
                "cot_mode": (["full", "melody", "off"], {"default": "full"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("lyrics_out", "style_out", "title", "suggested_seed", "bpm")
    FUNCTION = "execute_llm_action"
    CATEGORY = "YuE2/AI Producer"

    def execute_llm_action(
        self,
        action,
        provider="LMStudio",
        input_text="",
        model="gemma-4-e4b-it",
        api_key="",
        custom_model="",
        base_url="",
        style_context="",
        bpm=102,
        temperature=0.7,
        max_tokens=2048,
        cot_mode="full",
        llm_base_url="",
        llm_model="",
        **kwargs
    ):
        # Backwards compatibility for legacy saved workflows
        if not base_url and llm_base_url:
            base_url = llm_base_url
        if not custom_model and llm_model and llm_model != "gemma-4-e4b-it":
            custom_model = llm_model
        if provider.startswith("http"):
            base_url = provider
            provider = "LMStudio"

        provider_name = str(provider).strip()
        final_base_url = resolve_base_url(provider_name, base_url)
        final_model = resolve_model(provider_name, model, custom_model)
        final_api_key = resolve_api_key(provider_name, api_key)

        effective_bpm = resolve_bpm(bpm, style_context, input_text)

        print(f"\n[YuE2 AI Producer] Provider: {provider_name} | Model: {final_model} | Action: {action} | BPM: {effective_bpm}")

        if action == "Polish & Arrange Lyrics":
            raw_input_text = format_lyrics_payload(input_text)
            user_msg = (
                f"Target Music Style: {style_context or 'Modern Melodic'}\n"
                f"Target Tempo: {effective_bpm} BPM\n\n"
                f"Lyrics to polish and arrange (keep all sung lyrics 100% verbatim):\n{raw_input_text}"
            )
            resp_text = self._query_llm(provider_name, final_model, get_polish_sys_prompt(), user_msg, final_base_url, final_api_key, temperature, max_tokens)
            clean_lyrics = extract_clean_lyrics(resp_text)
            if not clean_lyrics:
                clean_lyrics = raw_input_text
            clean_lyrics = clean_yue2_lyrics(clean_lyrics)
            clean_lyrics = ensure_tempo_tag(clean_lyrics, effective_bpm)
            clean_lyrics = ensure_end_tag(clean_lyrics)

            style_str = enforce_english_style(style_context)
            if effective_bpm > 0 and f"{effective_bpm} bpm" not in style_str.lower():
                style_str = f"{style_str.rstrip(', ')}, {effective_bpm} BPM"

            return (clean_lyrics, style_str, "Polished Track", 42, effective_bpm)

        elif action == "Generate Full Song Concept":
            has_existing_lyrics = any(k in input_text for k in ["[Verse", "[Chorus", "\n\n"]) or len(input_text.splitlines()) > 4
            if has_existing_lyrics:
                user_msg = (
                    f"User Lyrics to Arrange & Architect into Full Song Concept (KEEP ALL SUNG LYRICS 100% VERBATIM):\n{format_lyrics_payload(input_text)}\n\n"
                    f"Preferred genre / style: {style_context or 'Pop / Dance Pop'}\n"
                    f"Target Tempo: {effective_bpm} BPM\n"
                    f"Language: English only\n\n"
                    f"Instruction: Generate the complete song concept. Keep all user sung lyrics 100% verbatim, and architect rich technical sub-tags, dynamic cues, vocal assignments, and top-loaded anchors."
                )
            else:
                user_msg = (
                    f"Song concept / theme: {input_text}\n"
                    f"Preferred genre / style: {style_context or 'Pop / Dance Pop'}\n"
                    f"Target Tempo: {effective_bpm} BPM\n"
                    f"Language: English only"
                )
            resp_text = self._query_llm(provider_name, final_model, CONCEPT_SYS_PROMPT, user_msg, final_base_url, final_api_key, temperature, max_tokens)
            try:
                data = robust_parse_concept_json(resp_text, default_style=style_context, default_bpm=effective_bpm)
                title = str(data.get("title", "Generated Track"))
                style_val = data.get("style", f"English, {style_context or 'pop'}, {effective_bpm} BPM")
                style = enforce_english_style(style_val)
                if effective_bpm > 0 and f"{effective_bpm} bpm" not in style.lower():
                    style = f"{style.rstrip(', ')}, {effective_bpm} BPM"
                lyrics_raw = data.get("lyrics")
                if not lyrics_raw or not str(lyrics_raw).strip():
                    lyrics_raw = extract_clean_lyrics(resp_text) or format_lyrics_payload(input_text)
                lyrics = clean_yue2_lyrics(lyrics_raw)
                lyrics = ensure_tempo_tag(lyrics, effective_bpm)
                lyrics = ensure_end_tag(lyrics)
                try:
                    seed = int(data.get("suggested_seed", 42))
                except Exception:
                    seed = 42
                return (lyrics, style, title, seed, effective_bpm)
            except Exception:
                extracted = extract_clean_lyrics(resp_text)
                clean_lyrics = extracted or format_lyrics_payload(input_text)
                clean_lyrics = clean_yue2_lyrics(clean_lyrics)
                clean_lyrics = ensure_tempo_tag(clean_lyrics, effective_bpm)
                clean_lyrics = ensure_end_tag(clean_lyrics)
                style_str = enforce_english_style(style_context)
                if effective_bpm > 0 and f"{effective_bpm} bpm" not in style_str.lower():
                    style_str = f"{style_str.rstrip(', ')}, {effective_bpm} BPM"
                return (clean_lyrics, style_str, "Generated Track", 42, effective_bpm)

        else:  # Optimize Acoustic Style String
            user_msg = (
                f"Create the ideal YuE2 style string for: {input_text}\n"
                f"Preferred Style Context: {style_context}\n"
                f"Target Tempo: {effective_bpm} BPM\n"
                f"Language: English always"
            )
            resp_text = self._query_llm(provider_name, final_model, STYLE_SYS_PROMPT, user_msg, final_base_url, final_api_key, temperature, max_tokens)
            clean_json = clean_json_text(resp_text)
            try:
                data = json.loads(clean_json)
                style_str = enforce_english_style(data.get("style_string", data.get("style", resp_text)))
            except Exception:
                style_str = enforce_english_style(resp_text)
            if effective_bpm > 0 and f"{effective_bpm} bpm" not in style_str.lower():
                style_str = f"{style_str.rstrip(', ')}, {effective_bpm} BPM"
            clean_input_lyrics = clean_yue2_lyrics(format_lyrics_payload(input_text))
            return (clean_input_lyrics, style_str, "Acoustic Style", 42, effective_bpm)

    def _query_llm(
        self,
        provider: str,
        model: str,
        sys_prompt: str,
        user_msg: str,
        base_url: str,
        api_key: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        p = provider.strip().lower()

        if p == "lmstudio":
            endpoint = f"{base_url}/chat/completions"
            auth = f"Bearer {api_key}" if api_key else "Bearer lm-studio"
            return self._call_openai_compatible(endpoint, auth, model, sys_prompt, user_msg, temperature, max_tokens)

        elif p == "ollama":
            return self._call_ollama(base_url, model, sys_prompt, user_msg, temperature, max_tokens)

        elif p == "openai":
            if not api_key:
                raise RuntimeError("OpenAI API key missing. Please enter your API key in 'api_key' or set the OPENAI_API_KEY environment variable.")
            endpoint = f"{base_url}/chat/completions"
            auth = f"Bearer {api_key}"
            return self._call_openai_compatible(endpoint, auth, model, sys_prompt, user_msg, temperature, max_tokens)

        elif p == "anthropic":
            if not api_key:
                raise RuntimeError("Anthropic API key missing. Please enter your API key in 'api_key' or set the ANTHROPIC_API_KEY environment variable.")
            return self._call_anthropic(base_url, api_key, model, sys_prompt, user_msg, temperature, max_tokens)

        elif p == "google":
            if not api_key:
                raise RuntimeError("Google Gemini API key missing. Please enter your API key in 'api_key' or set the GEMINI_API_KEY environment variable.")
            return self._call_google(base_url, api_key, model, sys_prompt, user_msg, temperature, max_tokens)

        elif p == "grok":
            if not api_key:
                raise RuntimeError("xAI Grok API key missing. Please enter your API key in 'api_key' or set the GROK_API_KEY environment variable.")
            endpoint = f"{base_url}/chat/completions"
            auth = f"Bearer {api_key}"
            return self._call_openai_compatible(endpoint, auth, model, sys_prompt, user_msg, temperature, max_tokens)

        elif p == "deepseek":
            if not api_key:
                raise RuntimeError("DeepSeek API key missing. Please enter your API key in 'api_key' or set the DEEPSEEK_API_KEY environment variable.")
            endpoint = f"{base_url}/chat/completions"
            auth = f"Bearer {api_key}"
            return self._call_openai_compatible(endpoint, auth, model, sys_prompt, user_msg, temperature, max_tokens)

        elif p == "openrouter":
            if not api_key:
                raise RuntimeError("OpenRouter API key missing. Please enter your API key in 'api_key' or set the OPENROUTER_API_KEY environment variable.")
            endpoint = f"{base_url}/chat/completions"
            auth = f"Bearer {api_key}"
            headers = {
                "HTTP-Referer": "https://github.com/comfyui",
                "X-Title": "ComfyUI-YuE2"
            }
            return self._call_openai_compatible(endpoint, auth, model, sys_prompt, user_msg, temperature, max_tokens, extra_headers=headers)

        else:
            endpoint = f"{base_url}/chat/completions" if not base_url.endswith("/chat/completions") else base_url
            auth = f"Bearer {api_key}" if api_key else "Bearer local"
            return self._call_openai_compatible(endpoint, auth, model, sys_prompt, user_msg, temperature, max_tokens)

    def _call_openai_compatible(
        self,
        url: str,
        auth_header: str,
        model: str,
        sys_prompt: str,
        user_msg: str,
        temperature: float,
        max_tokens: int,
        extra_headers: Optional[dict] = None
    ) -> str:
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user_msg})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens)
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": auth_header,
            "Accept": "application/json",
            "User-Agent": "ComfyUI-YuE2/1.0"
        }
        if extra_headers:
            headers.update(extra_headers)

        data = self._post_json(url, headers, payload)
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenAI-compatible endpoint returned no choices: {data}")
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") in ("text", "output_text")]
            content = "".join(parts)
        text = str(content).strip()
        if not text:
            raise RuntimeError(f"Empty model text response: {data}")
        return text

    def _call_ollama(
        self,
        base_url: str,
        model: str,
        sys_prompt: str,
        user_msg: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        url = f"{base_url.rstrip('/')}/api/chat"
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user_msg})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens)
            }
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ComfyUI-YuE2/1.0"
        }
        try:
            data = self._post_json(url, headers, payload)
            msg = data.get("message", {})
            text = str(msg.get("content", "")).strip()
            if text:
                return text
        except Exception:
            # Fallback to Ollama's OpenAI-compatible /v1/chat/completions endpoint
            v1_url = f"{base_url.rstrip('/')}/v1/chat/completions"
            return self._call_openai_compatible(v1_url, "Bearer ollama", model, sys_prompt, user_msg, temperature, max_tokens)
        raise RuntimeError(f"Empty response from Ollama at {url}")

    def _call_anthropic(
        self,
        base_url: str,
        api_key: str,
        model: str,
        sys_prompt: str,
        user_msg: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        url = f"{base_url.rstrip('/')}/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ComfyUI-YuE2/1.0"
        }
        payload = {
            "model": model,
            "max_tokens": int(max_tokens),
            "messages": [{"role": "user", "content": user_msg}],
            "temperature": float(temperature)
        }
        if sys_prompt:
            payload["system"] = sys_prompt

        data = self._post_json(url, headers, payload)
        content = data.get("content", [])
        texts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        out = "".join(texts).strip()
        if not out:
            raise RuntimeError(f"Empty response from Anthropic API: {data}")
        return out

    def _call_google(
        self,
        base_url: str,
        api_key: str,
        model: str,
        sys_prompt: str,
        user_msg: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        safe_model = urllib.parse.quote(str(model).strip(), safe="-_.~")
        safe_key = urllib.parse.quote(str(api_key).strip(), safe="")
        url = f"{base_url.rstrip('/')}/v1beta/models/{safe_model}:generateContent?key={safe_key}"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ComfyUI-YuE2/1.0"
        }
        payload = {
            "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens)
            }
        }
        if sys_prompt:
            payload["system_instruction"] = {"parts": [{"text": sys_prompt}]}

        data = self._post_json(url, headers, payload)
        if isinstance(data, dict) and "error" in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise RuntimeError(f"Google Gemini Error: {msg}")

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Google Gemini returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join([p.get("text", "") for p in parts if isinstance(p, dict)]).strip()
        if not text:
            raise RuntimeError(f"Empty response from Google Gemini: {data}")
        return text

    def _post_json(self, url: str, headers: dict, payload: dict, timeout: int = 120) -> dict:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} error ({url}): {err_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network connection error ({url}): {str(e.reason)}")
