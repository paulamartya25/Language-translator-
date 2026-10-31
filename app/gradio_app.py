"""
╔══════════════════════════════════════════════════════════════════════╗
║          MULTILINGUAL AI LANGUAGE TRANSLATOR — GRADIO UI             ║
║  Dark ocean theme · Floating multilingual letters · 49 languages    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import warnings
import tempfile
import json
import time

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import gradio as gr

# ── Lazy model imports ────────────────────────────────────────────────
try:
    from models.asr.whisper_asr import WhisperASR
    from models.translation.nllb_translator import NLLBTranslator
    from models.tts.gtts_engine import TTSEngine
    MODELS_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Model import warning: {e}")
    MODELS_AVAILABLE = False

# ── Language configuration (49 languages) ────────────────────────────
LANGUAGES = {
    # ── Indian Languages (22) ─────────────────────────────────────────
    "Hindi":      {"nllb": "hin_Deva", "gtts": "hi",    "flag": "🇮🇳", "sample": "नमस्ते"},
    "Bengali":    {"nllb": "ben_Beng", "gtts": "bn",    "flag": "🇧🇩", "sample": "নমস্কার"},
    "Tamil":      {"nllb": "tam_Taml", "gtts": "ta",    "flag": "🇮🇳", "sample": "வணக்கம்"},
    "Telugu":     {"nllb": "tel_Telu", "gtts": "te",    "flag": "🇮🇳", "sample": "నమస్కారం"},
    "Marathi":    {"nllb": "mar_Deva", "gtts": "mr",    "flag": "🇮🇳", "sample": "नमस्कार"},
    "Gujarati":   {"nllb": "guj_Gujr", "gtts": "gu",    "flag": "🇮🇳", "sample": "નમસ્તે"},
    "Kannada":    {"nllb": "kan_Knda", "gtts": "kn",    "flag": "🇮🇳", "sample": "ನಮಸ್ಕಾರ"},
    "Malayalam":  {"nllb": "mal_Mlym", "gtts": "ml",    "flag": "🇮🇳", "sample": "നമസ്കാരം"},
    "Punjabi":    {"nllb": "pan_Guru", "gtts": "pa",    "flag": "🇮🇳", "sample": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ"},
    "Odia":       {"nllb": "ory_Orya", "gtts": "or",    "flag": "🇮🇳", "sample": "ନମସ୍କାର"},
    "Assamese":   {"nllb": "asm_Beng", "gtts": "as",    "flag": "🇮🇳", "sample": "নমস্কাৰ"},
    "Urdu":       {"nllb": "urd_Arab", "gtts": "ur",    "flag": "🇵🇰", "sample": "السلام عليكم"},
    "Sanskrit":   {"nllb": "san_Deva", "gtts": "sa",    "flag": "🇮🇳", "sample": "नमस्ते"},
    "Maithili":   {"nllb": "mai_Deva", "gtts": "hi",    "flag": "🇮🇳", "sample": "प्रणाम"},
    "Sindhi":     {"nllb": "snd_Arab", "gtts": "sd",    "flag": "🇮🇳", "sample": "جي آيا"},
    "Nepali":     {"nllb": "npi_Deva", "gtts": "ne",    "flag": "🇳🇵", "sample": "नमस्कार"},
    "Konkani":    {"nllb": "kok_Deva", "gtts": "hi",    "flag": "🇮🇳", "sample": "नमस्कार"},
    "Manipuri":   {"nllb": "mni_Mtei", "gtts": "hi",    "flag": "🇮🇳", "sample": "ꯍꯥꯌ"},
    "Bodo":       {"nllb": "brx_Deva", "gtts": "hi",    "flag": "🇮🇳", "sample": "नमस्कार"},
    "Dogri":      {"nllb": "dgo_Deva", "gtts": "hi",    "flag": "🇮🇳", "sample": "नमस्ते"},
    "Kashmiri":   {"nllb": "kas_Arab", "gtts": "hi",    "flag": "🇮🇳", "sample": "السلام"},
    "Santali":    {"nllb": "sat_Olck", "gtts": "hi",    "flag": "🇮🇳", "sample": "ᱡᱚᱦᱟᱨ"},
    # ── International Languages (27) ──────────────────────────────────
    "Japanese":   {"nllb": "jpn_Jpan", "gtts": "ja",    "flag": "🇯🇵", "sample": "こんにちは"},
    "Chinese":    {"nllb": "zho_Hans", "gtts": "zh-CN", "flag": "🇨🇳", "sample": "你好"},
    "German":     {"nllb": "deu_Latn", "gtts": "de",    "flag": "🇩🇪", "sample": "Hallo"},
    "French":     {"nllb": "fra_Latn", "gtts": "fr",    "flag": "🇫🇷", "sample": "Bonjour"},
    "Dutch":      {"nllb": "nld_Latn", "gtts": "nl",    "flag": "🇳🇱", "sample": "Hallo"},
    "Spanish":    {"nllb": "spa_Latn", "gtts": "es",    "flag": "🇪🇸", "sample": "Hola"},
    "Italian":    {"nllb": "ita_Latn", "gtts": "it",    "flag": "🇮🇹", "sample": "Ciao"},
    "Portuguese": {"nllb": "por_Latn", "gtts": "pt",    "flag": "🇵🇹", "sample": "Olá"},
    "Russian":    {"nllb": "rus_Cyrl", "gtts": "ru",    "flag": "🇷🇺", "sample": "Привет"},
    "Arabic":     {"nllb": "arb_Arab", "gtts": "ar",    "flag": "🇸🇦", "sample": "مرحبا"},
    "Korean":     {"nllb": "kor_Hang", "gtts": "ko",    "flag": "🇰🇷", "sample": "안녕하세요"},
    "Turkish":    {"nllb": "tur_Latn", "gtts": "tr",    "flag": "🇹🇷", "sample": "Merhaba"},
    "Polish":     {"nllb": "pol_Latn", "gtts": "pl",    "flag": "🇵🇱", "sample": "Cześć"},
    "Swedish":    {"nllb": "swe_Latn", "gtts": "sv",    "flag": "🇸🇪", "sample": "Hej"},
    "Norwegian":  {"nllb": "nob_Latn", "gtts": "no",    "flag": "🇳🇴", "sample": "Hei"},
    "Danish":     {"nllb": "dan_Latn", "gtts": "da",    "flag": "🇩🇰", "sample": "Hej"},
    "Finnish":    {"nllb": "fin_Latn", "gtts": "fi",    "flag": "🇫🇮", "sample": "Hei"},
    "Greek":      {"nllb": "ell_Grek", "gtts": "el",    "flag": "🇬🇷", "sample": "Γεια"},
    "Hebrew":     {"nllb": "heb_Hebr", "gtts": "iw",    "flag": "🇮🇱", "sample": "שלום"},
    "Thai":       {"nllb": "tha_Thai", "gtts": "th",    "flag": "🇹🇭", "sample": "สวัสดี"},
    "Vietnamese": {"nllb": "vie_Latn", "gtts": "vi",    "flag": "🇻🇳", "sample": "Xin chào"},
    "Indonesian": {"nllb": "ind_Latn", "gtts": "id",    "flag": "🇮🇩", "sample": "Halo"},
    "Malay":      {"nllb": "zsm_Latn", "gtts": "ms",    "flag": "🇲🇾", "sample": "Helo"},
    "Filipino":   {"nllb": "tgl_Latn", "gtts": "tl",    "flag": "🇵🇭", "sample": "Kumusta"},
    "Swahili":    {"nllb": "swh_Latn", "gtts": "sw",    "flag": "🇰🇪", "sample": "Habari"},
    "Ukrainian":  {"nllb": "ukr_Cyrl", "gtts": "uk",    "flag": "🇺🇦", "sample": "Привіт"},
    "Czech":      {"nllb": "ces_Latn", "gtts": "cs",    "flag": "🇨🇿", "sample": "Ahoj"},
    "Romanian":   {"nllb": "ron_Latn", "gtts": "ro",    "flag": "🇷🇴", "sample": "Salut"},
}

LANG_NAMES   = list(LANGUAGES.keys())
INDIAN_LANGS = LANG_NAMES[:22]
INTL_LANGS   = LANG_NAMES[22:]

# ── Global model handles ──────────────────────────────────────────────
_asr   = None
_nmt   = None
_tts   = None
_ready = False

def _load_models():
    global _asr, _nmt, _tts, _ready
    if _ready:
        return
    try:
        print("Loading ASR …")
        _asr = WhisperASR()
        print("Loading NMT …")
        _nmt = NLLBTranslator()
        print("Loading TTS …")
        _tts = TTSEngine()
        _ready = True
        print("✅ All models loaded!")
    except Exception as e:
        print(f"⚠️  Model load error: {e}")

# ── Core functions ────────────────────────────────────────────────────
def transcribe(audio_path: str) -> str:
    if not audio_path:
        return ""
    _load_models()
    try:
        if _asr:
            return _asr.transcribe(audio_path)
    except Exception as e:
        return f"[ASR error: {e}]"
    return ""

def translate(text: str, tgt_nllb: str) -> str:
    if not text.strip():
        return ""
    _load_models()
    try:
        if _nmt:
            return _nmt.translate(text, tgt_lang=tgt_nllb)
    except Exception as e:
        return f"[NMT error: {e}]"
    return text

def speak(text: str, gtts_code: str) -> str | None:
    if not text.strip() or text.startswith("["):
        return None
    try:
        if _tts:
            path = _tts.synthesize(text, lang=gtts_code)
            return path
        from gtts import gTTS
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        gTTS(text=text, lang=gtts_code, slow=False).save(tmp.name)
        return tmp.name
    except Exception:
        return None

# ── Batch translate helper ────────────────────────────────────────────
def batch_translate(source_text: str, lang_list: list[str]) -> dict:
    results = {}
    for name in lang_list:
        info = LANGUAGES[name]
        translated = translate(source_text, info["nllb"])
        results[name] = {
            "flag":       info["flag"],
            "text":       translated,
            "gtts_code":  info["gtts"],
        }
    return results

# ═══════════════════════════════════════════════════════════════════════
#  CUSTOM CSS + FLOATING LETTERS JAVASCRIPT
# ═══════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
/* ── Google Fonts ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=Space+Grotesk:wght@400;600;700&display=swap');

/* ── CSS Variables ──────────────────────────────────────────── */
:root {
  --bg-deep:    #060d1f;
  --bg-card:    #0d1b35;
  --bg-card2:   #0f2040;
  --accent-1:   #00e5ff;
  --accent-2:   #7c3aed;
  --accent-3:   #f59e0b;
  --text-main:  #e8f4fd;
  --text-muted: #7b9cc4;
  --border:     rgba(0, 229, 255, 0.15);
  --glow:       0 0 24px rgba(0, 229, 255, 0.25);
  --radius:     14px;
}

/* ── Base Reset ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container, .gradio-container * {
  font-family: 'Inter', 'Space Grotesk', sans-serif !important;
  color: var(--text-main);
}

/* ── Full-page ocean background ─────────────────────────────── */
.gradio-container {
  background: var(--bg-deep) !important;
  min-height: 100vh;
  position: relative;
  overflow-x: hidden;
}

/* ── Floating letters canvas ────────────────────────────────── */
#lang-canvas {
  position: fixed;
  top: 0; left: 0;
  width: 100vw; height: 100vh;
  pointer-events: none;
  z-index: 0;
  opacity: 0.18;
}

/* ── Bring content above canvas ─────────────────────────────── */
.gradio-container > * {
  position: relative;
  z-index: 1;
}

/* ── Hero header ────────────────────────────────────────────── */
#hero-header {
  background: linear-gradient(135deg, #060d1f 0%, #0a1628 50%, #081020 100%);
  border-bottom: 1px solid var(--border);
  padding: 36px 24px 28px;
  text-align: center;
  position: relative;
  overflow: hidden;
}
#hero-header::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,229,255,0.08) 0%, transparent 70%);
}
.hero-badge {
  display: inline-block;
  background: rgba(0,229,255,0.1);
  border: 1px solid rgba(0,229,255,0.3);
  color: var(--accent-1);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 2px;
  text-transform: uppercase;
  padding: 4px 14px;
  border-radius: 20px;
  margin-bottom: 16px;
}
.hero-title {
  font-size: clamp(28px, 5vw, 52px);
  font-weight: 900;
  line-height: 1.15;
  margin: 0 0 12px;
  background: linear-gradient(90deg, #ffffff 0%, var(--accent-1) 50%, #a78bfa 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  font-size: 15px;
  color: var(--text-muted);
  max-width: 560px;
  margin: 0 auto 20px;
  line-height: 1.6;
}
.hero-stats {
  display: flex;
  justify-content: center;
  gap: 32px;
  flex-wrap: wrap;
}
.hero-stat {
  text-align: center;
}
.hero-stat .number {
  font-size: 24px;
  font-weight: 800;
  color: var(--accent-1);
}
.hero-stat .label {
  font-size: 11px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
}

/* ── Model status bar ───────────────────────────────────────── */
.model-bar {
  background: rgba(13,27,53,0.8);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 20px;
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
  align-items: center;
  backdrop-filter: blur(12px);
  margin: 16px 0;
}
.model-pill {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}
.model-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--accent-1);
  box-shadow: 0 0 6px var(--accent-1);
  animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%,100% { opacity: 1; } 50% { opacity: 0.4; }
}

/* ── Tabs ───────────────────────────────────────────────────── */
.tabs > .tab-nav {
  background: rgba(13,27,53,0.6) !important;
  border-bottom: 1px solid var(--border) !important;
  backdrop-filter: blur(12px);
  padding: 0 16px;
  gap: 4px;
}
.tabs > .tab-nav button {
  background: transparent !important;
  border: none !important;
  color: var(--text-muted) !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 12px 18px !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  transition: all 0.25s ease !important;
  letter-spacing: 0.3px;
}
.tabs > .tab-nav button:hover {
  color: var(--text-main) !important;
  border-bottom-color: rgba(0,229,255,0.4) !important;
}
.tabs > .tab-nav button.selected {
  color: var(--accent-1) !important;
  border-bottom-color: var(--accent-1) !important;
  background: rgba(0,229,255,0.05) !important;
}
.tabitem {
  background: transparent !important;
  padding: 20px 0 !important;
}

/* ── Cards ──────────────────────────────────────────────────── */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  backdrop-filter: blur(8px);
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
  border-color: rgba(0,229,255,0.35);
  box-shadow: var(--glow);
}
.card-title {
  font-size: 13px;
  font-weight: 700;
  color: var(--accent-1);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* ── Inputs ─────────────────────────────────────────────────── */
textarea, input[type="text"] {
  background: rgba(6,13,31,0.8) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text-main) !important;
  font-size: 15px !important;
  line-height: 1.6 !important;
  padding: 14px 16px !important;
  transition: border-color 0.25s, box-shadow 0.25s !important;
  resize: vertical !important;
}
textarea:focus, input[type="text"]:focus {
  border-color: var(--accent-1) !important;
  box-shadow: 0 0 0 3px rgba(0,229,255,0.12) !important;
  outline: none !important;
}
textarea::placeholder { color: var(--text-muted) !important; opacity: 0.7 !important; }

/* ── Labels ─────────────────────────────────────────────────── */
label > span, .gr-form label {
  color: var(--text-muted) !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 1.2px !important;
  margin-bottom: 8px !important;
}

/* ── Buttons ────────────────────────────────────────────────── */
.gr-button, button.gr-button {
  border-radius: 10px !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  letter-spacing: 0.5px !important;
  padding: 12px 24px !important;
  transition: all 0.25s ease !important;
  cursor: pointer !important;
}
.gr-button.primary, button[variant="primary"] {
  background: linear-gradient(135deg, var(--accent-1) 0%, #0891b2 100%) !important;
  border: none !important;
  color: #060d1f !important;
  box-shadow: 0 4px 15px rgba(0,229,255,0.3) !important;
}
.gr-button.primary:hover, button[variant="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 25px rgba(0,229,255,0.4) !important;
}
.gr-button.secondary {
  background: rgba(124,58,237,0.2) !important;
  border: 1px solid rgba(124,58,237,0.4) !important;
  color: #a78bfa !important;
}

/* ── Audio component ────────────────────────────────────────── */
.gr-audio, audio {
  background: rgba(6,13,31,0.8) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}

/* ── Checkboxes / Radio ─────────────────────────────────────── */
.gr-check-radio {
  background: rgba(13,27,53,0.6) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 10px 14px !important;
}
input[type="checkbox"]:checked { accent-color: var(--accent-1) !important; }

/* ── Dropdown ───────────────────────────────────────────────── */
select, .gr-dropdown {
  background: rgba(6,13,31,0.9) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text-main) !important;
}

/* ── Translation result cards ───────────────────────────────── */
.tr-card {
  background: var(--bg-card2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
  margin-bottom: 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  transition: all 0.25s ease;
}
.tr-card:hover {
  border-color: rgba(0,229,255,0.4);
  transform: translateX(3px);
}
.tr-lang {
  font-size: 12px;
  font-weight: 700;
  color: var(--accent-1);
  text-transform: uppercase;
  letter-spacing: 1px;
}
.tr-text {
  font-size: 16px;
  color: var(--text-main);
  line-height: 1.5;
}

/* ── Architecture diagram cards ─────────────────────────────── */
.arch-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 22px;
  margin-bottom: 16px;
  position: relative;
  overflow: hidden;
}
.arch-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
}
.arch-card.asr::before  { background: linear-gradient(90deg, var(--accent-1), #0891b2); }
.arch-card.nmt::before  { background: linear-gradient(90deg, var(--accent-2), #c084fc); }
.arch-card.tts::before  { background: linear-gradient(90deg, var(--accent-3), #f97316); }
.arch-title {
  font-size: 17px;
  font-weight: 800;
  margin-bottom: 8px;
}
.arch-param {
  font-size: 28px;
  font-weight: 900;
  color: var(--accent-1);
  margin-bottom: 4px;
}
.arch-sub {
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 14px;
}
.arch-list {
  list-style: none;
  padding: 0;
  margin: 0;
}
.arch-list li {
  font-size: 13px;
  color: var(--text-muted);
  padding: 4px 0;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  display: flex;
  align-items: center;
  gap: 8px;
}
.arch-list li::before {
  content: '›';
  color: var(--accent-1);
  font-weight: 700;
}

/* ── Result section header ───────────────────────────────────── */
.section-label {
  font-size: 11px;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── Scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: rgba(0,229,255,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,229,255,0.4); }

/* ── JSON output ─────────────────────────────────────────────── */
.gr-json {
  background: rgba(6,13,31,0.8) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  font-size: 13px !important;
}

/* ── Footer ──────────────────────────────────────────────────── */
.footer {
  text-align: center;
  padding: 24px;
  color: var(--text-muted);
  font-size: 12px;
  border-top: 1px solid var(--border);
  margin-top: 32px;
}
"""

FLOATING_LETTERS_JS = """
<canvas id="lang-canvas"></canvas>
<script>
(function() {
  const canvas = document.getElementById('lang-canvas');
  const ctx = canvas.getContext('2d');

  // Letters from many scripts
  const letters = [
    // Devanagari (Hindi/Sanskrit)
    'क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट','ठ','ड','ढ','ण','त','थ','द','ध','न','प','फ','ब','भ','म','य','र','ल','व','श','ष','स','ह',
    // Bengali
    'অ','আ','ই','ঈ','উ','ক','খ','গ','ঘ','চ','ছ','জ','ঝ','ট','ঠ','ড','ঢ','ত','থ','দ','ধ','ন','প','ফ','ব','ভ','ম','য','র','ল','শ','স','হ',
    // Tamil
    'அ','ஆ','இ','ஈ','உ','க','ங','ச','ஞ','ட','ண','த','ந','ப','ம','ய','ர','ல','வ','ழ','ள','ற','ன',
    // Telugu
    'అ','ఆ','ఇ','ఈ','ఉ','క','ఖ','గ','ఘ','చ','జ','ట','త','ద','న','ప','బ','మ','య','ర','ల','వ','శ','స','హ',
    // Gujarati
    'અ','આ','ઇ','ઈ','ઉ','ક','ખ','ગ','ઘ','ચ','છ','જ','ઝ','ટ','ત','દ','ન','પ','ફ','બ','ભ','મ','ય','ર','લ','વ','સ','હ',
    // Kannada
    'ಅ','ಆ','ಇ','ಈ','ಉ','ಕ','ಖ','ಗ','ಘ','ಚ','ಛ','ಜ','ಟ','ತ','ದ','ನ','ಪ','ಬ','ಮ','ಯ','ರ','ಲ','ವ','ಶ','ಸ','ಹ',
    // Malayalam
    'അ','ആ','ഇ','ഈ','ഉ','ക','ഖ','ഗ','ഘ','ച','ഛ','ജ','ഝ','ട','ഠ','ത','ദ','ന','പ','ബ','മ','യ','ര','ല','വ','ശ','സ','ഹ',
    // Gurmukhi (Punjabi)
    'ਅ','ਆ','ਇ','ਈ','ਉ','ਕ','ਖ','ਗ','ਘ','ਚ','ਛ','ਜ','ਝ','ਟ','ਤ','ਦ','ਨ','ਪ','ਫ','ਬ','ਭ','ਮ','ਯ','ਰ','ਲ','ਵ','ਸ','ਹ',
    // Japanese Hiragana
    'あ','い','う','え','お','か','き','く','け','こ','さ','し','す','せ','そ','た','ち','つ','て','と','な','に','ぬ','ね','の','は','ひ','ふ','へ','ほ','ま','み','む','め','も','や','ゆ','よ','ら','り','る','れ','ろ','わ','を','ん',
    // Japanese Katakana
    'ア','イ','ウ','エ','オ','カ','キ','ク','ケ','コ','サ','シ','ス','セ','ソ','タ','チ','ツ','テ','ト',
    // CJK Chinese
    '你','好','我','是','中','文','语','言','翻','译','器','世','界','和','平','美','丽','学','习','知','识',
    // Korean Hangul
    '가','나','다','라','마','바','사','아','자','차','카','타','파','하','안','녕','하','세','요',
    // Arabic
    'ا','ب','ت','ث','ج','ح','خ','د','ذ','ر','ز','س','ش','ص','ض','ط','ظ','ع','غ','ف','ق','ك','ل','م','ن','ه','و','ي',
    // Greek
    'α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ','ν','ξ','ο','π','ρ','σ','τ','υ','φ','χ','ψ','ω',
    // Cyrillic
    'А','Б','В','Г','Д','Е','Ж','З','И','К','Л','М','Н','О','П','Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Э','Ю','Я',
    // Hebrew
    'א','ב','ג','ד','ה','ו','ז','ח','ט','י','כ','ל','מ','נ','ס','ע','פ','צ','ק','ר','ש','ת',
    // Thai
    'ก','ข','ค','ง','จ','ช','ซ','ญ','ด','ต','ถ','ท','น','บ','ป','ผ','พ','ฟ','ภ','ม','ย','ร','ล','ว','ส','ห','อ','ฮ',
    // Georgian (bonus!)
    'ა','ბ','გ','დ','ე','ვ','ზ','თ','ი','კ','ლ','მ','ნ','ო','პ','ჟ','რ','ს','ტ','უ','ფ','ქ','ღ','ყ','შ','ჩ','ც','ძ','წ','ჭ','ხ','ჯ','ჰ',
  ];

  // Color palette matching the UI
  const colors = [
    'rgba(0, 229, 255, ALPHA)',   // cyan accent
    'rgba(167, 139, 250, ALPHA)', // violet
    'rgba(245, 158, 11, ALPHA)',  // amber
    'rgba(52, 211, 153, ALPHA)',  // emerald
    'rgba(248, 113, 113, ALPHA)', // red
    'rgba(255, 255, 255, ALPHA)', // white
  ];

  let particles = [];
  let animId;

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  function randomBetween(a, b) { return a + Math.random() * (b - a); }

  function createParticle() {
    const size = randomBetween(14, 48);
    return {
      x:     randomBetween(0, canvas.width),
      y:     randomBetween(-80, canvas.height + 80),
      vx:    randomBetween(-0.3, 0.3),
      vy:    randomBetween(0.15, 0.55),
      size:  size,
      alpha: randomBetween(0.3, 0.85),
      alphaDir: Math.random() > 0.5 ? 1 : -1,
      alphaDelta: randomBetween(0.002, 0.007),
      letter: letters[Math.floor(Math.random() * letters.length)],
      color:  colors[Math.floor(Math.random() * colors.length)],
      rotation: randomBetween(0, Math.PI * 2),
      rotSpeed: randomBetween(-0.005, 0.005),
    };
  }

  function initParticles(count) {
    particles = [];
    for (let i = 0; i < count; i++) {
      const p = createParticle();
      p.y = randomBetween(0, canvas.height); // distribute initially
      particles.push(p);
    }
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    particles.forEach(p => {
      // Move
      p.x += p.vx;
      p.y += p.vy;
      p.rotation += p.rotSpeed;
      p.alpha += p.alphaDir * p.alphaDelta;
      if (p.alpha >= 0.85 || p.alpha <= 0.15) p.alphaDir *= -1;

      // Wrap around
      if (p.y > canvas.height + 60)  { p.y = -60; p.x = randomBetween(0, canvas.width); }
      if (p.x < -60) p.x = canvas.width + 60;
      if (p.x > canvas.width + 60)   p.x = -60;

      // Draw
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rotation);
      ctx.font = `${p.size}px 'Noto Sans', 'Arial Unicode MS', sans-serif`;
      ctx.fillStyle = p.color.replace('ALPHA', p.alpha.toFixed(2));
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(p.letter, 0, 0);
      ctx.restore();
    });

    animId = requestAnimationFrame(draw);
  }

  function start() {
    resize();
    const count = Math.max(40, Math.floor((canvas.width * canvas.height) / 28000));
    initParticles(count);
    draw();
  }

  window.addEventListener('resize', () => {
    cancelAnimationFrame(animId);
    resize();
    const count = Math.max(40, Math.floor((canvas.width * canvas.height) / 28000));
    initParticles(count);
    draw();
  });

  // Wait for canvas to be in DOM
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
</script>
"""

HERO_HTML = """
<div id="hero-header">
  <div class="hero-badge">🌐 AI Language Translator</div>
  <div class="hero-title">Speak. Translate. Connect.</div>
  <div class="hero-sub">
    Real-time speech recognition &amp; translation powered by custom Conformer ASR,
    Seq2Seq NMT with RoPE, and FastSpeech2 TTS — across 49 languages.
  </div>
  <div class="hero-stats">
    <div class="hero-stat"><div class="number">49</div><div class="label">Languages</div></div>
    <div class="hero-stat"><div class="number">22</div><div class="label">Indian</div></div>
    <div class="hero-stat"><div class="number">27</div><div class="label">International</div></div>
    <div class="hero-stat"><div class="number">~80M</div><div class="label">Total Params</div></div>
  </div>
</div>
"""

MODEL_BAR_HTML = """
<div class="model-bar">
  <span style="font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;">Active Models:</span>
  <div class="model-pill"><span class="model-dot"></span><span>🎙️ MicroConformer ASR (~25M)</span></div>
  <div class="model-pill"><span class="model-dot" style="background:#a78bfa;box-shadow:0 0 6px #a78bfa"></span><span>🌐 Seq2Seq NMT + RoPE (~35M)</span></div>
  <div class="model-pill"><span class="model-dot" style="background:#f59e0b;box-shadow:0 0 6px #f59e0b"></span><span>🔊 FastSpeech2 TTS (~20M)</span></div>
</div>
"""

ARCH_HTML = """
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-top:8px;">
  <div class="arch-card asr">
    <div class="arch-param">~25M</div>
    <div class="arch-title">🎙️ MicroConformerASR</div>
    <div class="arch-sub">Audio-Spectrogram Transformer + Conformer Encoder</div>
    <ul class="arch-list">
      <li>Log-Mel Spectrogram (80 mels) + Conv Subsampling (4×)</li>
      <li>6× ConformerBlocks (RelPos MHSA + ConvModule)</li>
      <li>6× Transformer Decoder blocks</li>
      <li>Joint CTC (30%) + CrossEntropy (70%) loss</li>
      <li>Weight-tied LM head ↔ decoder embeddings</li>
      <li>Trained on: FLEURS (8 Indian langs) + CommonVoice EN</li>
    </ul>
  </div>
  <div class="arch-card nmt">
    <div class="arch-param" style="color:#a78bfa">~35M</div>
    <div class="arch-title" style="color:#c4b5fd">🌐 MultilingualSeq2Seq</div>
    <div class="arch-sub">Rotary Position Embeddings + Weight-Tied Embeddings</div>
    <ul class="arch-list">
      <li>6× Encoder blocks with RoPE self-attention + GELU FFN</li>
      <li>6× Decoder blocks with RoPE self-attn + cross-attn</li>
      <li>Rotary Position Embeddings (RoPE) on Q, K</li>
      <li>Weight-tied: embed = decoder embed = lm_head</li>
      <li>Label Smoothing loss (ε = 0.1)</li>
      <li>Trained on: Samanantar + OPUS-100 (13 lang pairs)</li>
    </ul>
  </div>
  <div class="arch-card tts">
    <div class="arch-param" style="color:#f59e0b">~20M</div>
    <div class="arch-title" style="color:#fcd34d">🔊 FastSpeech2TTS</div>
    <div class="arch-sub">Non-Autoregressive TTS + Griffin-Lim Vocoder</div>
    <ul class="arch-list">
      <li>4× FFT Blocks Encoder (Conv1D FFN, kernel=9)</li>
      <li>Duration + Pitch + Energy Variance Predictor</li>
      <li>Length Regulator → frame-level expansion</li>
      <li>4× FFT Blocks Decoder + 5-layer PostNet</li>
      <li>Griffin-Lim Vocoder (training-free)</li>
      <li>Trained on: LJSpeech-1.1 + IndicTTS</li>
    </ul>
  </div>
</div>
"""

# ═══════════════════════════════════════════════════════════════════════
#  GRADIO INTERFACE
# ═══════════════════════════════════════════════════════════════════════
def build_ui():

    # ── Translation handler functions ─────────────────────────────────
    def do_transcribe(audio_path):
        if not audio_path:
            return "⚠️ Please record or upload audio first."
        result = transcribe(audio_path)
        return result or "Could not transcribe audio."

    def do_single_translate(text, lang_name):
        if not text.strip():
            return "", None
        info = LANGUAGES.get(lang_name, {})
        translated = translate(text.strip(), info.get("nllb", "hin_Deva"))
        audio_path = speak(translated, info.get("gtts", "hi"))
        return translated, audio_path

    def do_multi_translate(text, selected_langs):
        if not text.strip() or not selected_langs:
            return {}
        out = {}
        for lang_name in selected_langs:
            info = LANGUAGES[lang_name]
            t = translate(text.strip(), info["nllb"])
            out[f"{info['flag']} {lang_name}"] = t
        return out

    def do_audio_pipeline(audio_path, selected_langs):
        if not audio_path:
            return "No audio provided.", {}
        text = transcribe(audio_path) or ""
        translations = do_multi_translate(text, selected_langs)
        return text, translations

    def do_all_langs(text):
        return do_multi_translate(text, LANG_NAMES)

    def do_indian_langs(text):
        return do_multi_translate(text, INDIAN_LANGS)

    def do_intl_langs(text):
        return do_multi_translate(text, INTL_LANGS)

    # ── Build Gradio blocks ───────────────────────────────────────────
    with gr.Blocks(
        title="🌐 Multilingual AI Translator",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="cyan",
            secondary_hue="violet",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        )
    ) as demo:

        # Floating letters canvas + Hero
        gr.HTML(FLOATING_LETTERS_JS)
        gr.HTML(HERO_HTML)
        gr.HTML(MODEL_BAR_HTML)

        with gr.Tabs():
            # ──────────────────────────────────────────────────────────
            # TAB 1 — LIVE RECORDING
            # ──────────────────────────────────────────────────────────
            with gr.Tab("🎙️ Live Recording"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="card-title">🎙️ Audio Input</div>')
                        audio_in = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Record or Upload Audio",
                        )
                        lang_check = gr.CheckboxGroup(
                            choices=LANG_NAMES,
                            value=["Hindi", "Bengali", "Tamil", "Japanese",
                                   "Chinese", "German", "French", "Arabic"],
                            label="Target Languages",
                        )
                        rec_btn = gr.Button("🔄 Transcribe & Translate", variant="primary")

                    with gr.Column(scale=1):
                        gr.HTML('<div class="card-title">📝 Transcription</div>')
                        transcribed_box = gr.Textbox(
                            label="Transcribed English Text",
                            placeholder="Transcription appears here…",
                            lines=3, interactive=False,
                        )
                        gr.HTML('<div class="card-title" style="margin-top:16px">🌍 Translations</div>')
                        trans_json = gr.JSON(label="Translation Results")

                rec_btn.click(
                    fn=do_audio_pipeline,
                    inputs=[audio_in, lang_check],
                    outputs=[transcribed_box, trans_json],
                )

            # ──────────────────────────────────────────────────────────
            # TAB 2 — TEXT INPUT + QUICK TRANSLATE
            # ──────────────────────────────────────────────────────────
            with gr.Tab("📝 Text → Translate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="card-title">✏️ Enter English Text</div>')
                        text_in = gr.Textbox(
                            label="English Text",
                            placeholder="Type something to translate…\ne.g. Hello, how are you today?",
                            lines=5,
                        )
                        single_lang = gr.Dropdown(
                            choices=LANG_NAMES,
                            value="Hindi",
                            label="Target Language",
                        )
                        translate_btn = gr.Button("🔊 Translate & Speak", variant="primary")

                    with gr.Column(scale=1):
                        gr.HTML('<div class="card-title">📖 Translation Output</div>')
                        single_out = gr.Textbox(
                            label="Translated Text",
                            lines=5, interactive=False,
                            placeholder="Translation appears here…",
                        )
                        gr.HTML('<div class="card-title" style="margin-top:16px">🔊 Voice Output</div>')
                        audio_out = gr.Audio(label="Synthesized Speech", type="filepath")

                translate_btn.click(
                    fn=do_single_translate,
                    inputs=[text_in, single_lang],
                    outputs=[single_out, audio_out],
                )

            # ──────────────────────────────────────────────────────────
            # TAB 3 — MULTI-LANGUAGE SELECT
            # ──────────────────────────────────────────────────────────
            with gr.Tab("🌍 Multi-Language"):
                gr.HTML('<div class="card-title">🌍 Translate to Multiple Languages at Once</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        multi_text = gr.Textbox(
                            label="English Text",
                            placeholder="Enter text to translate to multiple languages…",
                            lines=4,
                        )
                        multi_langs = gr.CheckboxGroup(
                            choices=LANG_NAMES,
                            value=INDIAN_LANGS[:8] + ["Japanese", "Chinese", "German", "French"],
                            label="Select Languages",
                        )
                        multi_btn = gr.Button("🌐 Translate to Selected", variant="primary")

                    with gr.Column(scale=1):
                        multi_out = gr.JSON(label="🌍 Translations")

                multi_btn.click(
                    fn=do_multi_translate,
                    inputs=[multi_text, multi_langs],
                    outputs=[multi_out],
                )

            # ──────────────────────────────────────────────────────────
            # TAB 4 — ALL 49 LANGUAGES
            # ──────────────────────────────────────────────────────────
            with gr.Tab("🗺️ All 49 Languages"):
                gr.HTML('<div class="card-title">🗺️ Translate to All 49 Languages</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        all_text = gr.Textbox(
                            label="English Text",
                            placeholder="Enter text to translate to all languages…",
                            lines=4,
                        )
                        all_btn = gr.Button("🌐 Translate to All 49 Languages", variant="primary")

                    with gr.Column(scale=1):
                        all_out = gr.JSON(label="All Translations")

                all_btn.click(fn=do_all_langs, inputs=[all_text], outputs=[all_out])

            # ──────────────────────────────────────────────────────────
            # TAB 5 — INDIAN LANGUAGES
            # ──────────────────────────────────────────────────────────
            with gr.Tab("🇮🇳 Indian Languages"):
                gr.HTML("""
                <div class="card-title">🇮🇳 All 22 Indian Languages</div>
                <div style="font-size:13px;color:var(--text-muted);margin-bottom:16px;">
                  Hindi · Bengali · Tamil · Telugu · Marathi · Gujarati · Kannada · Malayalam
                  · Punjabi · Odia · Assamese · Urdu · Sanskrit · Maithili · Sindhi · Nepali
                  · Konkani · Manipuri · Bodo · Dogri · Kashmiri · Santali
                </div>
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        ind_text = gr.Textbox(
                            label="English Text",
                            placeholder="Enter English text…",
                            lines=4,
                        )
                        ind_btn = gr.Button("🇮🇳 Translate to Indian Languages", variant="primary")
                    with gr.Column(scale=1):
                        ind_out = gr.JSON(label="🇮🇳 Indian Language Translations")
                ind_btn.click(fn=do_indian_langs, inputs=[ind_text], outputs=[ind_out])

            # ──────────────────────────────────────────────────────────
            # TAB 6 — INTERNATIONAL
            # ──────────────────────────────────────────────────────────
            with gr.Tab("🌏 International"):
                gr.HTML("""
                <div class="card-title">🌏 27 International Languages</div>
                <div style="font-size:13px;color:var(--text-muted);margin-bottom:16px;">
                  Japanese · Chinese · German · French · Dutch · Spanish · Italian · Portuguese
                  · Russian · Arabic · Korean · Turkish · Polish · Swedish · Norwegian · Danish
                  · Finnish · Greek · Hebrew · Thai · Vietnamese · Indonesian · Malay · Filipino
                  · Swahili · Ukrainian · Czech · Romanian
                </div>
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        intl_text = gr.Textbox(
                            label="English Text",
                            placeholder="Enter English text…",
                            lines=4,
                        )
                        intl_btn = gr.Button("🌏 Translate to International Languages", variant="primary")
                    with gr.Column(scale=1):
                        intl_out = gr.JSON(label="🌏 International Translations")
                intl_btn.click(fn=do_intl_langs, inputs=[intl_text], outputs=[intl_out])

            # ──────────────────────────────────────────────────────────
            # TAB 7 — ARCHITECTURE
            # ──────────────────────────────────────────────────────────
            with gr.Tab("🏗️ Architecture"):
                gr.HTML("""
                <div class="card-title" style="margin-bottom:20px;">
                  🏗️ Custom Model Architectures — Built From Scratch in PyTorch
                </div>
                """)
                gr.HTML(ARCH_HTML)
                gr.HTML("""
                <div style="margin-top:20px;padding:20px;background:var(--bg-card);border:1px solid var(--border);border-radius:14px;">
                  <div class="card-title">🗂️ Kaggle Training Pipeline</div>
                  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:12px;">
                    <div style="background:rgba(0,229,255,0.06);border:1px solid rgba(0,229,255,0.2);border-radius:10px;padding:14px;">
                      <div style="font-size:12px;font-weight:700;color:var(--accent-1);">📓 Notebook 1</div>
                      <div style="font-size:13px;margin-top:4px;">Train MicroConformerASR</div>
                      <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">FLEURS (8 Indian) + CommonVoice EN</div>
                      <div style="font-size:11px;color:var(--text-muted);">Output: <code>asr_model.pt</code></div>
                    </div>
                    <div style="background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.25);border-radius:10px;padding:14px;">
                      <div style="font-size:12px;font-weight:700;color:#a78bfa;">📓 Notebook 2</div>
                      <div style="font-size:13px;margin-top:4px;">Train Multilingual NMT</div>
                      <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">Samanantar + OPUS-100</div>
                      <div style="font-size:11px;color:var(--text-muted);">Output: <code>nmt_model.pt</code></div>
                    </div>
                    <div style="background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.25);border-radius:10px;padding:14px;">
                      <div style="font-size:12px;font-weight:700;color:var(--accent-3);">📓 Notebook 3</div>
                      <div style="font-size:13px;margin-top:4px;">Train FastSpeech2TTS</div>
                      <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">LJSpeech-1.1 + IndicTTS</div>
                      <div style="font-size:11px;color:var(--text-muted);">Output: <code>tts_model.pt</code></div>
                    </div>
                    <div style="background:rgba(52,211,153,0.07);border:1px solid rgba(52,211,153,0.25);border-radius:10px;padding:14px;">
                      <div style="font-size:12px;font-weight:700;color:#34d399;">📓 Notebook 4</div>
                      <div style="font-size:13px;margin-top:4px;">Live Gradio App</div>
                      <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">Load 3 .pt files → Gradio share=True</div>
                      <div style="font-size:11px;color:var(--text-muted);">Public URL for 72h</div>
                    </div>
                  </div>
                </div>
                """)

        # Footer
        gr.HTML("""
        <div class="footer">
          🌐 Multilingual AI Language Translator &nbsp;·&nbsp;
          Built with PyTorch · Gradio · HuggingFace &nbsp;·&nbsp;
          49 Languages · Custom Architectures from Scratch
        </div>
        """)

    return demo


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
    )
