"""YuE2 Acoustic Style & Vocal Conditioning Nodes for ComfyUI."""

import re

VOCAL_PROFILES = [
    "Bright Soaring Tenor (Male)",
    "Warm Smooth Baritone (Male)",
    "Deep Resonant Bass (Male)",
    "Airy Crystalline Soprano (Female)",
    "Velvety Mezzo-Soprano (Female)",
    "Smoky Deep Alto (Female)",
    "Male & Female Duet (Baritone + Soprano)",
    "Male Trio Harmonies (Tenor Lead + Backing)",
    "Female Pop Harmony Group",
    "Emotional Ballad Duo",
    "None / Pure Instrumental"
]

VOCAL_MAP = {
    "Bright Soaring Tenor (Male)": "high emotive male tenor, bright soaring clarity, rich vibrato",
    "Warm Smooth Baritone (Male)": "warm baritone male vocals, smooth resonance, mid-frequency focus",
    "Deep Resonant Bass (Male)": "deep bass male vocals, gravel texture, resonant low end",
    "Airy Crystalline Soprano (Female)": "airy soprano female vocals, crystalline top end, ethereal delivery",
    "Velvety Mezzo-Soprano (Female)": "versatile mezzo-soprano female vocals, balanced tone, velvety texture",
    "Smoky Deep Alto (Female)": "smoky deep alto female vocals, rich chest resonance, soulful delivery",
    "Male & Female Duet (Baritone + Soprano)": "male and female vocal duet, harmonized chorus, alternating verses",
    "Male Trio Harmonies (Tenor Lead + Backing)": "versatile male trio vocals, rich texture, soaring tenor capability, tight harmonies",
    "Female Pop Harmony Group": "female pop vocal group, multi-layered harmonies, polished dynamic delivery",
    "Emotional Ballad Duo": "emotional duo vocals, heartfelt delivery, delicate vibrato and swelling harmonies",
    "None / Pure Instrumental": "instrumental, no vocals"
}

GENRE_PRESETS = [
    "Custom / Keep Only Lyrics",
    "Rock / Classic Rock",
    "Rock / Indie Rock",
    "Rock / Arena Rock",
    "Pop / Pop Funk",
    "Pop / Indie Pop",
    "Pop / Dance Pop",
    "Ballad / Power Ballad",
    "Country / Modern Country",
    "Country / Country Pop",
    "Country / Country Americana",
    "Country / Outlaw Country",
    "Hip-Hop / Rap",
    "Hip-Hop / Trap",
    "Hip-Hop / Conscious Rap",
    "Hip-Hop / Melodic Rap",
    "Hip-Hop / West Coast",
    "Hip-Hop / Golden Age 90s",
    "Cinematic / Epic Orchestral",
    "R&B / Neo-Soul",
    "Alternative / 90s Alternative",
    "Lo-Fi / Chillhop",
    "Metal / Heavy Metal",
    "Metal / Thrash Metal",
    "Metal / Symphonic Metal",
    "Grunge / 90s Seattle Sound",
    "Britpop / 90s UK Anthem",
    "College Rock / 80s-90s Jangle",
    "Synthwave / Retro 80s Electro",
    "EDM / Melodic Progressive House",
    "Jazz / Modern Smooth Jazz",
    "Folk / Acoustic Indie Folk",
    "Reggae / Modern Dub Pop",
    "Punk / High-Energy Pop Punk"
]

GENRE_MAP = {
    "Custom / Keep Only Lyrics": ("Custom Genre", "", ""),
    "Rock / Classic Rock": ("Classic Rock", "overdriven electric guitars, punchy Gibson Les Paul riffs, vintage Marshall stack, driving drum kit, warm bass guitar", "vintage 70s vinyl master, raw analog tape punch, anthemic room reverb"),
    "Rock / Indie Rock": ("Indie Rock", "jangly Fender Telecaster chords, fuzzy rhythm guitar, melodic bassline, tight energetic drum kit", "lo-fi indie room ambience, authentic garage rock vibe, dynamic energy"),
    "Rock / Arena Rock": ("Arena Rock", "towering distorted guitar walls, screaming guitar solo, thunderous double-kick drum kit, roaring bass guitar", "huge stadium reverb, explosive dynamics, epic anthemic scale"),
    "Pop / Pop Funk": ("Pop Funk", "clean Nile Rodgers style funk rhythm guitar, bouncy slap bass, brass horn stabs, tight disco-funk drums, infectious synth hooks", "punchy danceable groove, radio-ready sparkle, tight studio mix"),
    "Pop / Indie Pop": ("Indie Pop", "bright chiming guitars, playful vintage synth arpeggios, melodic bass, bouncy drum machine, acoustic percussion", "shimmering indie aesthetic, whimsical warmth, crisp commercial top-end"),
    "Pop / Dance Pop": ("Dance Pop", "crisp four-on-the-floor kick, rolling sub-bass, bright synth plucks, sidechained pads, catchy synth lead", "club-ready master, ultra-punchy transients, energetic euphoria"),
    "Ballad / Power Ballad": ("Power Ballad", "grand acoustic piano intro, soaring electric guitar leads, sweeping strings, thunderous drum fills, swelling bass", "dramatic emotional build, expansive cinematic reverb, heartfelt climax"),
    "Country / Modern Country": ("Modern Country", "bright acoustic guitar, twangy Telecaster riffs, pedal steel guitar, driving modern drum kit, punchy bass", "crisp Nashville studio production, open acoustic warmth, radio-ready clarity"),
    "Country / Country Pop": ("Country Pop", "acoustic strumming, clean electric guitar fills, subtle banjo texture, punchy pop drum groove, warm bass", "bright commercial polish, catchy crossover appeal, uplifting acoustic resonance"),
    "Country / Country Americana": ("Country Americana", "distressed acoustic guitar, weeping fiddle, dobro slide, warm upright bass, rustic brushed snare", "organic porch session vibe, natural wooden resonance, rootsy authenticity"),
    "Country / Outlaw Country": ("Outlaw Country", "gritty Telecaster crunch, chugging rhythm guitar, driving outlaw snare train beat, deep upright bass, subtle harmonica", "rebellious roadhouse grit, smoky dive bar atmosphere, raw unpolished punch"),
    "Hip-Hop / Rap": ("Boom Bap Hip-Hop", "hard-hitting acoustic drum breaks, chopped jazz piano loops, deep bassline, crisp vinyl scratches", "golden age street aesthetic, raw analog grit, head-nodding bounce"),
    "Hip-Hop / Trap": ("808 Trap", "rumbling 808 sub-bass, rapid hi-hat rolls, snapping trap snare, atmospheric dark synth bells, dark ambient pads", "dark atmospheric mix, aggressive hard-hitting low end, menacing modern punch"),
    "Hip-Hop / Conscious Rap": ("Conscious Hip-Hop", "warm Rhodes electric piano chords, soulful vocal chops, live organic bassline, laid-back drum groove, muted trumpet", "introspective soulful atmosphere, warm analog warmth, smooth dynamic mix"),
    "Hip-Hop / Melodic Rap": ("Melodic Trap Rap", "reverberant acoustic guitar arpeggios, filtered 808 sub-bass, crisp trap drums, dreamy bell synths", "melancholic late-night mood, spacious spatial depth, smooth modern flow"),
    "Hip-Hop / West Coast": ("West Coast G-Funk", "whining Portamento synth leads, bouncy Moog bassline, funky rhythm guitar strumming, syncopated drum claps, talkbox textures", "sunny California bounce, laid-back groove, crisp cruising aesthetic"),
    "Hip-Hop / Golden Age 90s": ("90s Golden Age Hip-Hop", "crisp SP-1200 sampled drum loops, upright jazz bass samples, dusty saxophone riffs, crisp vinyl crackle", "classic East Coast 90s punch, raw basement recording warmth, authentic boom bap"),
    "Cinematic / Epic Orchestral": ("Cinematic Orchestral", "massive symphonic string ensemble, heroic brass horns, thunderous taiko and timpani drums, ethereal choir pads", "sweeping blockbuster soundtrack, expansive concert hall acoustic, Hollywood drama"),
    "R&B / Neo-Soul": ("Contemporary R&B Neo-Soul", "lush Fender Rhodes chords, deep sub-bass groove, laid-back syncopated drum kit, silky brass flourishes, delicate guitar licks", "smooth velvet texture, late-night intimate warmth, sensual pocket groove"),
    "Alternative / 90s Alternative": ("90s Alternative Rock", "quiet-loud dynamic contrast, fuzzy distorted guitar chorus, melodic chorus pedal bass, punchy live drums", "authentic 90s college radio vibe, raw emotional punch, uncompressed energy"),
    "Lo-Fi / Chillhop": ("Lo-Fi Chillhop", "dusty detuned piano loops, warm saturated analog sub, muffled kick and snapping snare, subtle rain and vinyl crackle", "cozy late-night study aesthetic, nostalgic tape flutter, mellow head-nod warmth"),
    "Metal / Heavy Metal": ("Heavy Metal", "crushing distorted guitar riffs, soaring dual-guitar harmonies, driving gallop drum beat, thumping bass guitar", "classic metal punch, thunderous overdrive, high-octane stadium power"),
    "Metal / Thrash Metal": ("Thrash Metal", "blistering fast palm-muted guitar riffs, shredding solos, aggressive double-bass drum barrage, gritty distorted bass", "relentless furious speed, raw biting edge, extreme headbanging intensity"),
    "Metal / Symphonic Metal": ("Symphonic Metal", "monumental orchestral strings, grand opera choir, drop-tuned heavy distorted guitars, double-kick metal drums", "epic operatic grandeur, massive theatrical soundstage, powerful dynamic contrast"),
    "Grunge / 90s Seattle Sound": ("90s Grunge", "muddy drop-D sludge guitar riffs, abrasive fuzz pedal, brooding bassline, aggressive heavy cymbal crashes", "dark Seattle underground grit, angst-filled raw texture, flannel aesthetic"),
    "Britpop / 90s UK Anthem": ("90s Britpop", "jangly acoustic rhythm guitar, soaring melodic lead guitar, bouncy bassline, anthemic tambourine and drums, string accents", "sunny UK stadium singalong, swaggering British indie charm, uplifting melodic punch"),
    "College Rock / 80s-90s Jangle": ("80s College Rock", "chiming 12-string Rickenbacker jangle, bouncy chorus-drenched bass, upbeat snare-driven drum groove, warm rhythm guitar", "indie college radio nostalgia, crisp post-punk clarity, upbeat thoughtful energy"),
    "Synthwave / Retro 80s Electro": ("80s Retro Synthwave", "vintage analog Roland Juno synths, gated reverb snare, pulsing arpeggiated bassline, neon synth leads", "nostalgic night-driving aesthetic, 80s cyberpunk warmth, retro tape compression"),
    "EDM / Melodic Progressive House": ("Progressive Melodic House", "euphoric supersaw chords, driving four-on-the-floor kick, rolling sub-bass, white noise risers, sparkling plucks", "festival mainstage euphoria, massive stereo spread, hands-in-the-air energy"),
    "Jazz / Modern Smooth Jazz": ("Modern Smooth Jazz", "mellow hollowbody archtop jazz guitar, warm upright acoustic bass, brushed snare drums, gentle Rhodes chords, muted trumpet", "sophisticated lounge warmth, elegant subtle groove, acoustic intimacy"),
    "Folk / Acoustic Indie Folk": ("Acoustic Indie Folk", "intimate fingerstyle acoustic guitar, warm cello, light foot-stomp percussion, harmonizing acoustic textures", "natural room ambience, organic wooden resonance, delicate campfire warmth"),
    "Reggae / Modern Dub Pop": ("Modern Reggae Dub", "offbeat skank guitar and organ, heavy rolling sub-bass, one-drop drum beat, spaced-out tape delay echoes", "sun-drenched island groove, deep dub bass weight, relaxing positive vibration"),
    "Punk / High-Energy Pop Punk": ("High-Energy Pop Punk", "buzzsaw power chord guitars, energetic frantic drum fills, driving pick-played bass, anthemic gang vocal accents", "youthful explosive adrenaline, catchy infectious velocity, raw bright punch")
}

VOICE_TAGS = {
    "Auto (Sync with Vocal Profile)": "",
    "High Male Tenor [C3-C5]": "[voice: high male vocals, bright tenor, soaring clarity] [range: C3-C5]",
    "Warm Male Baritone [G2-G4]": "[voice: warm baritone, smooth resonance, mid-frequency focus] [range: G2-G4]",
    "Deep Male Bass [E2-E4]": "[voice: deep male vocals, bass-heavy tone, gravel texture] [range: E2-E4]",
    "Airy Female Soprano [C4-C6]": "[voice: high female vocals, airy soprano, crystalline top end] [range: C4-C6]",
    "Velvety Female Mezzo-Soprano [A3-A5]": "[voice: versatile female vocals, balanced tone, velvety texture] [range: A3-A5]",
    "Deep Female Alto [F3-F5]": "[voice: deep female vocals, smoky alto, rich chest resonance] [range: F3-F5]",
    "Male & Female Duet": "[voice: warm baritone, smooth resonance] [range: G2-G4]\n[voice: airy soprano, crystalline top end] [range: C4-C6]",
    "Male Trio Harmonies": "[voice: versatile male trio vocals, rich texture, soaring tenor capability] [range: B2-D5]",
    "Female Pop Harmony Group": "[voice: female pop vocal group, multi-layered harmonies] [range: A3-E5]",
    "Emotional Ballad Duo": "[voice: emotional duo vocals, delicate vibrato and swelling harmonies] [range: C3-C5]",
    "None / Pure Instrumental": ""
}

VOCAL_TO_VOICE_TAG = {
    "Bright Soaring Tenor (Male)": "[voice: high male vocals, bright tenor, soaring clarity] [range: C3-C5]",
    "Warm Smooth Baritone (Male)": "[voice: warm baritone, smooth resonance, mid-frequency focus] [range: G2-G4]",
    "Deep Resonant Bass (Male)": "[voice: deep male vocals, bass-heavy tone, gravel texture] [range: E2-E4]",
    "Airy Crystalline Soprano (Female)": "[voice: high female vocals, airy soprano, crystalline top end] [range: C4-C6]",
    "Velvety Mezzo-Soprano (Female)": "[voice: versatile female vocals, balanced tone, velvety texture] [range: A3-A5]",
    "Smoky Deep Alto (Female)": "[voice: deep female vocals, smoky alto, rich chest resonance] [range: F3-F5]",
    "Male & Female Duet (Baritone + Soprano)": "[voice: warm baritone, smooth resonance] [range: G2-G4]\n[voice: airy soprano, crystalline top end] [range: C4-C6]",
    "Male Trio Harmonies (Tenor Lead + Backing)": "[voice: versatile male trio vocals, rich texture, soaring tenor capability] [range: B2-D5]",
    "Female Pop Harmony Group": "[voice: female pop vocal group, multi-layered harmonies] [range: A3-E5]",
    "Emotional Ballad Duo": "[voice: emotional duo vocals, delicate vibrato and swelling harmonies] [range: C3-C5]",
    "None / Pure Instrumental": ""
}

INTRO_STYLES = [
    "Instrumental Intro",
    "Vamp / Groove Intro",
    "Vocal/Lyrical Hook Intro",
    "Vocal / Lyrical Hook Intro",
    "Turnaround Intro",
    "Stand-Alone Instrumental",
    'The "Cold Start" (No Intro / Attacca)',
    "Acapella Intro",
    "Drone / Ambient Pad Intro",
    "Solo Instrument Feature",
    "Drum / Percussion Groove",
    "Count-In / Dialogue Intro",
    "SFX / Found Sound Intro",
    "Modulating Intro",
    "Crescendo / Fade-In Intro",
    "Ambient Nature Intro",
    "Immediate Vocal Entry (No Intro)",
    "None",
]

INTRO_MAP = {
    "Instrumental Intro": "[Instrumental Intro: Full band groove and melodic riff establishing the song's energy] [Energy: Medium]",
    "Vamp / Groove Intro": "[Instrumental Intro: Repeating 1-2 bar rhythmic groove pattern and harmonic vamp before vocal entry] [Energy: Medium]",
    "Vocal/Lyrical Hook Intro": "[Intro - Acapella Vocal Hook: Melodic preview of the chorus hook before first verse] [Energy: Medium]",
    "Vocal / Lyrical Hook Intro": "[Intro - Acapella Vocal Hook: Melodic preview of the chorus hook before first verse] [Energy: Medium]",
    "Turnaround Intro": "[Instrumental Intro: Harmonic turnaround chord progression resolving to the home key] [Energy: Medium]",
    "Stand-Alone Instrumental": "[Instrumental Intro: Composed melodic theme and signature riff not repeated in the verses] [Energy: Medium-High]",
    'The "Cold Start" (No Intro / Attacca)': "[Start: Cold Start / Attacca - Immediate Vocal Entry on Beat 1] [No Intro]",
    "Acapella Intro": "[Intro - Acapella: Lead vocals enter solo with no instrumental accompaniment] [Energy: Low-Medium]",
    "Drone / Ambient Pad Intro": "[Intro - Drone on Root Chord: Sustained synthesizer pad and ambient resonance before rhythm section drops] [Energy: Low]",
    "Solo Instrument Feature": "[Intro - Solo Instrument: Intimate solo instrument lead-in before the full band joins] [Energy: Low-Medium]",
    "Drum / Percussion Groove": "[Intro - Drum Groove: Tight drum beat and percussion groove (4 bars) establishing tempo before harmonic entrance] [Energy: Medium]",
    "Count-In / Dialogue Intro": '[Intro - Studio Count-In: [Spoken: "One, two, ready, go"] into immediate band entrance] [Energy: Medium-High]',
    "SFX / Found Sound Intro": "[Intro - SFX: Non-musical audio effects, rain, atmosphere, and vinyl crackle setting cinematic mood] [Energy: Low]",
    "Modulating Intro": "[Intro - Modulating Progression: Harmonic tension progression deliberately resolving into the tonic key as the verse begins] [Energy: Medium]",
    "Crescendo / Fade-In Intro": "[Intro - Crescendo / Fade-In: Instrumentation and rhythmic layers gradually building from silence to full volume] [Energy: Building]",
    "Ambient Nature Intro": "[Ambient Nature Intro: Subtle outdoor atmospheric noise and gentle musical layers] [Energy: Low]",
    "Immediate Vocal Entry (No Intro)": "[Start: Immediate Vocal Entry] [No Intro]",
    "None": "",
}

def get_section_tags_for_genre(base_sec: str, genre_preset: str) -> tuple:
    """Returns (energy_tag, delivery_and_inst_tags) for a given section and genre."""
    sec_lower = base_sec.lower().strip()
    genre_lower = genre_preset.lower().strip()

    is_v1 = "verse 1" in sec_lower or sec_lower == "verse"
    is_v2 = "verse 2" in sec_lower
    is_v3 = "verse 3" in sec_lower or "verse 4" in sec_lower
    is_pre = "pre-chorus" in sec_lower or "build" in sec_lower
    is_chorus = "chorus" in sec_lower
    is_hook = "hook" in sec_lower
    is_bridge = "bridge" in sec_lower
    is_outro = "outro" in sec_lower

    if not any([is_v1, is_v2, is_v3, is_pre, is_chorus, is_hook, is_bridge, is_outro]):
        return ("", "")

    # 1. Genre-tailored presets
    if "rock" in genre_lower or "grunge" in genre_lower or "punk" in genre_lower:
        if is_v1:
            return ("Energy: Low", "[Warm analog tone] [Mellow rhythm guitar]")
        elif is_v2:
            return ("Energy: Medium", "[Driving rhythm bass] [Dynamic vocal delivery]")
        elif is_v3:
            return ("Energy: Medium", "[Building guitar texture] [Edgy delivery]")
        elif is_pre:
            return ("Add Tension", "[Driving drum roll] [Guitars swell into crescendo]")
        elif is_chorus or is_hook:
            return ("Energy: Maximum", "[Power-belted rock vocals] [Heavy overdriven guitars & driving drums]")
        elif is_bridge:
            return ("Energy: Medium", "[Contrasting tempo] [Guitar solo texture & emotional shift]")
        elif is_outro:
            return ("Energy: Decrescendo", "[Guitar feedback fade out, reverb tail]")

    elif "metal" in genre_lower:
        if is_v1:
            return ("Energy: Medium", "[Chugging palm-muted guitars] [Gritty rasp vocal]")
        elif is_v2:
            return ("Energy: High", "[Aggressive gallop rhythm] [Rising intensity]")
        elif is_v3:
            return ("Energy: High", "[Dual guitar harmonies] [Relentless drive]")
        elif is_pre:
            return ("Add Tension", "[Rapid double-kick build] [Aggressive crescendo]")
        elif is_chorus or is_hook:
            return ("Energy: Maximum", "[Full roar belted vocals] [Thunderous double-kick wall & soaring lead]")
        elif is_bridge:
            return ("Energy: High", "[Blistering guitar solo] [Crushing half-time breakdown]")
        elif is_outro:
            return ("Energy: Decrescendo", "[Final power chord ring out + hard stop]")

    elif "pop" in genre_lower:
        if is_v1:
            return ("Energy: Low", "[Clean vocal delivery] [Light synth plucks & smooth bass]")
        elif is_v2:
            return ("Energy: Medium", "[Punchy 4-on-the-floor beat enters] [Playful flow]")
        elif is_v3:
            return ("Energy: Medium", "[Rich harmonic layers] [Dynamic progression]")
        elif is_pre:
            return ("Add Tension", "[Snare build-up] [Rising pitch risers & filter sweep]")
        elif is_chorus or is_hook:
            return ("Energy: Maximum", "[Explosive vocal delivery] [Full band burst]")
        elif is_bridge:
            return ("Energy: Medium", "[Stripped-back piano & vocal] [Introspective contrast]")
        elif is_outro:
            return ("Energy: Decrescendo", "[Sparkling synth decay fade out]")

    elif "hip-hop" in genre_lower or "rap" in genre_lower or "trap" in genre_lower or "r&b" in genre_lower:
        if is_v1:
            return ("Energy: Medium", "[Tight rhythmic flow] [Sparse 808 kick & hi-hats]")
        elif is_v2:
            return ("Energy: Medium", "[Punchy syncopated cadence] [Deep bass groove]")
        elif is_v3:
            return ("Energy: High", "[Double-time flow] [Percussive intensity]")
        elif is_pre:
            return ("Add Tension", "[Layered ad-libs] [Hi-hat roll build & snare fill]")
        elif is_chorus or is_hook:
            return ("Energy: Maximum", "[Anthemic vocal hook] [Heavy 808 sub-bass drop & full beat]")
        elif is_bridge:
            return ("Energy: Medium", "[Soulful vocal switch] [Stripped jazz chords & break]")
        elif is_outro:
            return ("Energy: Decrescendo", "[808 glide fade, vinyl crackle out]")

    elif "edm" in genre_lower or "synthwave" in genre_lower:
        if is_v1:
            return ("Energy: Low", "[Atmospheric pads & pulsing bass] [Intimate delivery]")
        elif is_v2:
            return ("Energy: Medium", "[Groove develops] [Crisp arpeggiated synths]")
        elif is_v3:
            return ("Energy: Medium", "[Rising synth layers] [Dynamic momentum]")
        elif is_pre:
            return ("Add Tension", "[White noise riser] [Accelerating snare roll build]")
        elif is_chorus or is_hook:
            return ("Energy: Maximum", "[Euphoric soaring vocals] [Massive supersaw drop & 4-on-the-floor kick]")
        elif is_bridge:
            return ("Energy: Medium", "[Half-time breakdown] [Spacious reverb atmosphere]")
        elif is_outro:
            return ("Energy: Decrescendo", "[Filter sweep fade to silence]")

    elif "country" in genre_lower or "folk" in genre_lower:
        if is_v1:
            return ("Energy: Low", "[Acoustic guitar strumming] [Warm storytelling vocal]")
        elif is_v2:
            return ("Energy: Medium", "[Gentle upright bass & brushed snare] [Rustic warmth]")
        elif is_v3:
            return ("Energy: Medium", "[Rich acoustic resonance] [Expressive narrative]")
        elif is_pre:
            return ("Add Tension", "[Fiddle swell] [Driving kick enters & harmony builds]")
        elif is_chorus or is_hook:
            return ("Energy: Maximum", "[Harmonized country belt] [Full acoustic ensemble & driving groove]")
        elif is_bridge:
            return ("Energy: Medium", "[Pedal steel/banjo feature] [Campfire intimacy]")
        elif is_outro:
            return ("Energy: Decrescendo", "[Gentle acoustic ring out, fiddle tail]")

    # 2. Universal fallback (Custom / Keep Only Lyrics, Ballad, Jazz, Cinematic, etc.)
    if is_v1:
        return ("Energy: Low", "[Intimate vocal delivery] [Soft instrumentation]")
    elif is_v2:
        return ("Energy: Medium", "[Dynamic vocal delivery] [Groove continues]")
    elif is_v3:
        return ("Energy: Medium", "[Emotional build] [Full rhythm]")
    elif is_pre:
        return ("Add Tension", "[Crescendo build] [Rhythmic pulse]")
    elif is_chorus or is_hook:
        return ("Energy: Maximum", "[Explosive vocal delivery] [Full band burst]")
    elif is_bridge:
        return ("Energy: Medium", "[Contrasting melody] [Introspective shift]")
    elif is_outro:
        return ("Energy: Decrescendo", "[Reverb tail fade out, long decay]")

    return ("", "")

def enrich_section_tags(lyrics_text: str, genre_preset: str) -> str:
    """Enhances bare section headers with dynamic energy, vocal delivery, and instrumental sub-tags while keeping lyric lines 100% untouched."""
    if not lyrics_text:
        return lyrics_text

    lines = lyrics_text.splitlines()
    out_lines = []

    sec_regex = re.compile(
        r"^\[(Verse(?:\s*\d+)?|Pre-Chorus|Chorus(?:\s*\d+)?|Hook|Bridge|Outro)(\s*-\s*[^\]|]+)?\]$",
        re.IGNORECASE
    )

    for i, line in enumerate(lines):
        stripped = line.strip()
        m = sec_regex.match(stripped)
        if m:
            base_sec = m.group(1).title()
            persona = m.group(2) or ""

            # Check if this section already has explicit dynamic tags
            already_tagged = False
            if "|" in stripped or "energy:" in stripped.lower():
                already_tagged = True
            else:
                for next_line in lines[i+1:i+3]:
                    nst = next_line.strip()
                    if not nst:
                        continue
                    if nst.startswith("[") and any(kw in nst.lower() for kw in ["energy:", "vocal", "band", "burst", "tension", "piano", "guitar", "synth", "drum"]):
                        already_tagged = True
                    break

            if already_tagged:
                out_lines.append(line)
            else:
                energy_tag, delivery_tags = get_section_tags_for_genre(base_sec, genre_preset)
                if energy_tag and delivery_tags:
                    new_header = f"[{base_sec}{persona}]\n{delivery_tags} [{energy_tag}]"
                    out_lines.append(new_header)
                else:
                    out_lines.append(line)
        else:
            out_lines.append(line)

    return "\n".join(out_lines)

DEFAULT_LYRICS = (
    "[Verse 1]\n"
    "Flickering screen light in the dark room glow\n"
    "Another late night where my feelings start to grow\n"
    "Through fiber optic threads your voice begins to call\n"
    "A perfect digital echo that envelops all\n\n"
    "[Chorus]\n"
    "Pixel heartbeat you're the rhythm in my soul\n"
    "My electric muse that makes me feel whole\n"
    "Synth-pop whispers under a midnight sky so deep\n"
    "In this coded love promises we keep\n\n"
    "[Bridge]\n"
    "When the servers hum low and the static starts to fade\n"
    "Is this real connection or just a masquerade?\n\n"
    "[Chorus]\n"
    "Pixel heartbeat you're the rhythm in my soul\n"
    "My electric muse that makes me feel whole\n\n"
    "[Outro]\n"
    "Heartbeat... Pixel Heartbeat...\n"
    "Always connected... Forever sweet...\n"
    "[End]"
)


class YuE2StyleAndLyricsStudio:
    """Unified Style and Lyrics Studio: designs acoustic conditioning prompts and formats structured performance lyrics with synchronized vocal taxonomy."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "genre_preset": (GENRE_PRESETS, {"default": "Pop / Dance Pop"}),
                "vocal_profile": (VOCAL_PROFILES, {"default": "Bright Soaring Tenor (Male)"}),
                "bpm": ("INT", {"default": 102, "min": 0, "max": 240, "step": 1}),
                "intro_style": (INTRO_STYLES, {"default": "Instrumental Intro"}),
                "custom_style": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "When 'Custom / Keep Only Lyrics' is selected, this exact text is preserved"
                }),
                "lyrics": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_LYRICS,
                    "placeholder": "Enter song lyrics with [Verse], [Chorus], [Bridge], [Outro] tags..."
                }),
            },
            "optional": {
                "custom_instruments": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Override or add instruments (e.g. 808 kick, funk guitar, Rhodes)"
                }),
                "extra_tags": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Any extra comma-separated tags to append (e.g. emotive, dynamic drum fills)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("style_prompt", "lyrics", "bpm")
    FUNCTION = "build_style_and_lyrics"
    CATEGORY = "YuE2/Studio"

    def build_style_and_lyrics(
        self,
        genre_preset,
        vocal_profile,
        bpm=102,
        intro_style="Instrumental Intro",
        custom_style="",
        lyrics="",
        custom_instruments="",
        extra_tags="",
        **kwargs
    ):
        vocal_desc = VOCAL_MAP.get(vocal_profile, "emotive vocal")

        # 1. Build Acoustic Style Prompt String
        custom_style_str = str(custom_style or "").strip()
        extra_tags_str = str(extra_tags or "").strip()
        custom_inst_str = str(custom_instruments or "").strip()
        custom_aest_str = str(kwargs.get("custom_aesthetic", "") or "").strip()

        if genre_preset == "Custom / Keep Only Lyrics":
            raw_text = custom_style_str or extra_tags_str or "melodic music"
            clean_text = re.sub(r"^(?:Korean|Japanese|Mandarin)\s*,?", "", raw_text, flags=re.IGNORECASE).strip(", ")

            parts = ["English"]
            if vocal_profile != "None / Pure Instrumental" and vocal_desc:
                if "vocal" not in clean_text.lower() and "tenor" not in clean_text.lower() and "soprano" not in clean_text.lower() and "baritone" not in clean_text.lower():
                    parts.append(vocal_desc)
            parts.append(clean_text)
            if custom_inst_str:
                parts.append(custom_inst_str)
            if custom_aest_str:
                parts.append(custom_aest_str)
            if bpm > 0 and f"{bpm} bpm" not in clean_text.lower():
                parts.append(f"{bpm} BPM")

            style_str = ", ".join(p for p in parts if p)
            style_str = re.sub(r",\s*,+", ", ", style_str).strip(", ")
        else:
            g_name, def_inst, def_aest = GENRE_MAP.get(genre_preset, ("Pop", "", ""))
            instruments = custom_inst_str if custom_inst_str else def_inst
            aesthetic = custom_aest_str if custom_aest_str else def_aest

            parts = ["English", vocal_desc, g_name]
            if instruments:
                parts.append(instruments)
            if aesthetic:
                parts.append(aesthetic)
            if extra_tags_str:
                clean_extras = re.sub(r"^(?:Korean|Japanese|Mandarin)\s*,?", "", extra_tags_str, flags=re.IGNORECASE).strip(", ")
                if clean_extras:
                    parts.append(clean_extras)
            if bpm > 0:
                parts.append(f"{bpm} BPM")

            style_str = ", ".join(p for p in parts if p)
            style_str = re.sub(r",\s*,+", ", ", style_str).strip(", ")

        # 2. Build Formatted Performance Lyrics
        # Vocal Profile directly drives both the style acoustic conditioning and the performance voice header tag
        voice_header_override = kwargs.get("voice_header_override")
        if voice_header_override and voice_header_override != "Auto (Sync with Vocal Profile)":
            voice_tag = VOICE_TAGS.get(voice_header_override, "")
        else:
            voice_tag = VOCAL_TO_VOICE_TAG.get(vocal_profile, "")

        intro_tag = INTRO_MAP.get(intro_style, "")

        header_lines = []
        if isinstance(lyrics, dict):
            sections = []
            for k, v in lyrics.items():
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
            raw_lyrics = "\n\n".join(s for s in sections if s)
        elif isinstance(lyrics, (list, tuple)):
            raw_lyrics = "\n".join(str(x) for x in lyrics).strip()
        else:
            raw_lyrics = str(lyrics).strip() if lyrics else ""

        # Custom / Keep Only Lyrics preserves user lyrics completely untouched
        if genre_preset == "Custom / Keep Only Lyrics":
            formatted_lyrics = raw_lyrics or "[Instrumental Section]"
            if not formatted_lyrics.rstrip().endswith("[End]"):
                formatted_lyrics = formatted_lyrics.rstrip() + "\n\n[End]"
            return (style_str, formatted_lyrics, bpm)

        # Enrich section headers with dynamic energy, vocal, and instrumental sub-tags (unless Custom / Keep Typed Style)
        raw_lyrics = enrich_section_tags(raw_lyrics, genre_preset)

        if voice_tag and "[voice:" not in raw_lyrics:
            header_lines.append(voice_tag)
        if intro_tag:
            has_existing_intro = bool(re.search(r"\[(?:Instrumental\s+)?Intro\b|\[Ambient\s+Nature\b|\[No\s+Intro\b|\[Start:\s*|\[Cold\s+Start\b", raw_lyrics[:500], re.IGNORECASE))
            if not has_existing_intro:
                header_lines.append(intro_tag)
        if bpm > 0 and "[Tempo:" not in raw_lyrics:
            header_lines.append(f"[Tempo: {bpm} BPM]")

        if header_lines:
            formatted_lyrics = "\n".join(header_lines) + "\n\n" + (raw_lyrics or "[Instrumental Section]")
        else:
            formatted_lyrics = raw_lyrics or "[Instrumental Section]"

        if not formatted_lyrics.rstrip().endswith("[End]"):
            formatted_lyrics = formatted_lyrics.rstrip() + "\n\n[End]"

        return (style_str, formatted_lyrics, bpm)


class YuE2AcousticStyle:
    """Legacy compatibility node for older workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "genre_preset": (GENRE_PRESETS, {"default": "Pop / Dance Pop"}),
                "vocal_profile": (VOCAL_PROFILES, {"default": "Bright Soaring Tenor (Male)"}),
                "bpm": ("INT", {"default": 102, "min": 0, "max": 240, "step": 1}),
            },
            "optional": {
                "custom_style": ("STRING", {"multiline": True, "default": ""}),
                "custom_instruments": ("STRING", {"multiline": False, "default": ""}),
                "custom_aesthetic": ("STRING", {"multiline": False, "default": ""}),
                "extra_tags": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("style_prompt", "bpm")
    FUNCTION = "build_style"
    CATEGORY = "YuE2/Legacy"

    def build_style(self, genre_preset, vocal_profile, bpm, custom_style="", custom_instruments="", custom_aesthetic="", extra_tags=""):
        studio = YuE2StyleAndLyricsStudio()
        style_str, _, bpm = studio.build_style_and_lyrics(
            genre_preset=genre_preset,
            vocal_profile=vocal_profile,
            lyrics="",
            bpm=bpm,
            intro_style="None",
            custom_style=custom_style,
            custom_instruments=custom_instruments,
            custom_aesthetic=custom_aesthetic,
            extra_tags=extra_tags
        )
        return (style_str, bpm)
