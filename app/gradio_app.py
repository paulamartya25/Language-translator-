"""
🌐 Multilingual AI Translator — Gradio Application
====================================================
Entry point: python app/gradio_app.py

Replaces Streamlit with Gradio for:
  - Better audio recording widget support
  - Real-time streaming UI
  - More control over layout and components
  - No rerun loop — event-driven callbacks

Pipeline:
  Audio/Text → Whisper ASR → NLLB-200 Translation → gTTS TTS
"""

import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Suppress noisy warnings ────────────────────────────────────────────────────
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import gradio as gr

from config.languages import (
    INDIAN_LANGUAGES,
    INTERNATIONAL_LANGUAGES,
    ALL_LANGUAGES,
    get_nllb_code,
    whisper_to_nllb,
)
from config.settings import (
    ASR_MODEL,
    TRANSLATION_MODEL,
    DEVICE,
    SAMPLE_RATE,
    WHISPER_MODEL_OPTIONS,
    TTS_SLOW_MODE,
)
from utils.audio_utils import bytes_to_audio_array, audio_array_to_bytes, is_silent
from utils.text_utils import clean_text, detect_script

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Global model cache  (loaded once, reused across all requests)
# ══════════════════════════════════════════════════════════════════════════════

_MODELS: Dict = {}


def _get_whisper(model_id: str):
    key = f"whisper:{model_id}"
    if key not in _MODELS:
        from models.asr.whisper_asr import WhisperASR
        _MODELS[key] = WhisperASR(model_id=model_id, device=DEVICE)
    return _MODELS[key]


def _get_translator():
    if "nllb" not in _MODELS:
        from models.translation.nllb_translator import NLLBTranslator
        _MODELS["nllb"] = NLLBTranslator(model_id=TRANSLATION_MODEL, device=DEVICE)
    return _MODELS["nllb"]


def _get_tts():
    if "tts" not in _MODELS:
        from models.tts.gtts_engine import TTSEngine
        _MODELS["tts"] = TTSEngine(slow=TTS_SLOW_MODE)
    return _MODELS["tts"]


# ══════════════════════════════════════════════════════════════════════════════
# Core pipeline functions (called by Gradio event handlers)
# ══════════════════════════════════════════════════════════════════════════════

def _translate_text_core(
    text: str,
    source_nllb: str,
    target_langs: List[str],
    num_beams: int,
    enable_tts: bool,
    progress: gr.Progress,
) -> Tuple[str, List[List]]:
    """
    Translate text to target languages and optionally synthesise TTS.

    Returns:
        status_msg : str
        table_data : List[List[str]]  — rows for the output Dataframe
    """
    if not text or not text.strip():
        return "⚠️ No text to translate.", []

    translator = _get_translator()
    tts = _get_tts()

    table_rows = []
    total = len(target_langs)

    for i, lang_name in enumerate(progress.tqdm(target_langs, desc="Translating")):
        if lang_name not in ALL_LANGUAGES:
            continue
        lang_info = ALL_LANGUAGES[lang_name]
        nllb_code = lang_info["nllb_code"]
        gtts_code = lang_info["gtts_lang"]
        flag = lang_info.get("flag", "")

        translated = translator.translate(
            text=text,
            source_lang_nllb=source_nllb,
            target_lang_nllb=nllb_code,
            num_beams=num_beams,
        )

        table_rows.append([f"{flag} {lang_name}", nllb_code, translated])

    status = f"✅ Translated to {len(table_rows)} languages."
    return status, table_rows


def _audio_to_nllb_source(whisper_lang_code: str) -> str:
    return whisper_to_nllb(whisper_lang_code)


# ══════════════════════════════════════════════════════════════════════════════
# Gradio event handlers
# ══════════════════════════════════════════════════════════════════════════════

def handle_audio_translate(
    audio_input,         # (sample_rate, ndarray) from gr.Audio
    whisper_size: str,
    src_mode: str,
    manual_src: str,
    selected_langs: List[str],
    num_beams: int,
    enable_tts: bool,
    progress=gr.Progress(track_tqdm=True),
):
    """Full audio pipeline: ASR → Translation → TTS per language."""

    if audio_input is None:
        return (
            "⚠️ No audio provided. Record or upload audio first.",
            "",
            "",
            [],
            None,
        )

    sr, audio_arr = audio_input
    audio_arr = audio_arr.astype(np.float32)
    if audio_arr.ndim > 1:
        audio_arr = audio_arr.mean(axis=1)

    # Normalize
    max_val = np.abs(audio_arr).max()
    if max_val > 0:
        audio_arr /= max_val

    # Resample to 16 kHz if needed
    if sr != 16000:
        from utils.audio_utils import resample_audio
        audio_arr = resample_audio(audio_arr, sr, 16000)

    if is_silent(audio_arr):
        return "⚠️ Audio is silent. Please speak clearly.", "", "", [], None

    # ── ASR ──────────────────────────────────────────────────────────────────
    progress(0, desc="🎤 Loading ASR model…")
    model_id = WHISPER_MODEL_OPTIONS.get(whisper_size, ASR_MODEL)
    whisper = _get_whisper(model_id)

    lang_hint = None
    if src_mode == "Manual" and manual_src in ALL_LANGUAGES:
        lang_hint = ALL_LANGUAGES[manual_src]["gtts_lang"]

    progress(0.1, desc="🎤 Transcribing audio…")
    asr_result = whisper.transcribe(audio_arr, sample_rate=16000, language=lang_hint)

    transcription = asr_result.get("text", "").strip()
    detected_whisper = asr_result.get("language", lang_hint or "en")

    if not transcription:
        return "❌ No speech detected. Please try again.", "", "", [], None

    from config.languages import get_language_name_from_whisper
    src_lang_name = get_language_name_from_whisper(detected_whisper)
    nllb_source = _audio_to_nllb_source(detected_whisper)
    script = detect_script(transcription)

    src_info = f"**Detected:** {src_lang_name} (`{detected_whisper}`) | Script: {script}"

    # ── Translation ───────────────────────────────────────────────────────────
    if not selected_langs:
        return "⚠️ Select at least one target language.", transcription, src_info, [], None

    progress(0.2, desc="⏳ Loading translation model…")
    status, table_rows = _translate_text_core(
        transcription, nllb_source, selected_langs, num_beams, enable_tts, progress
    )

    # ── TTS for first selected language (preview audio) ───────────────────────
    preview_audio = None
    if enable_tts and table_rows:
        try:
            tts = _get_tts()
            first_lang = selected_langs[0]
            first_text = table_rows[0][2]
            gtts_code = ALL_LANGUAGES[first_lang]["gtts_lang"]
            audio_bytes = tts.synthesize(first_text, gtts_code)
            if audio_bytes:
                preview_audio = io.BytesIO(audio_bytes)
        except Exception as e:
            logger.warning(f"Preview TTS failed: {e}")

    progress(1.0, desc="✅ Done!")
    return status, transcription, src_info, table_rows, preview_audio


def handle_text_translate(
    text_input: str,
    src_lang_manual: str,
    selected_langs: List[str],
    num_beams: int,
    enable_tts: bool,
    progress=gr.Progress(track_tqdm=True),
):
    """Text-only pipeline: Translation → TTS."""

    text = clean_text(text_input or "")
    if not text:
        return "⚠️ Please enter some text.", [], None

    if src_lang_manual not in ALL_LANGUAGES:
        return "⚠️ Please select a valid source language.", [], None

    nllb_source = get_nllb_code(src_lang_manual) or "eng_Latn"

    progress(0.05, desc="⏳ Loading translation model…")
    status, table_rows = _translate_text_core(
        text, nllb_source, selected_langs, num_beams, enable_tts, progress
    )

    # TTS preview
    preview_audio = None
    if enable_tts and table_rows:
        try:
            tts = _get_tts()
            first_lang = selected_langs[0]
            first_text = table_rows[0][2]
            gtts_code = ALL_LANGUAGES[first_lang]["gtts_lang"]
            audio_bytes = tts.synthesize(first_text, gtts_code)
            if audio_bytes:
                preview_audio = io.BytesIO(audio_bytes)
        except Exception as e:
            logger.warning(f"Preview TTS failed: {e}")

    progress(1.0, desc="✅ Done!")
    return status, table_rows, preview_audio


def handle_tts_single(lang_name: str, text: str):
    """Generate TTS for a specific language on demand."""
    if not text or lang_name not in ALL_LANGUAGES:
        return None
    try:
        tts = _get_tts()
        gtts_code = ALL_LANGUAGES[lang_name]["gtts_lang"]
        audio_bytes = tts.synthesize(text, gtts_code)
        return io.BytesIO(audio_bytes) if audio_bytes else None
    except Exception as e:
        logger.error(f"On-demand TTS failed: {e}")
        return None


def handle_export_json(table_rows: List[List], transcription: str, src_info: str) -> str:
    """Export translations as formatted JSON string."""
    export = {
        "transcription": transcription,
        "source_info": src_info,
        "translations": [
            {"language": row[0], "nllb_code": row[1], "text": row[2]}
            for row in (table_rows or [])
        ],
    }
    return json.dumps(export, ensure_ascii=False, indent=2)


def get_lang_audio(lang_name: str, table_data: List[List]):
    """Get TTS audio for a selected language from the table."""
    if not table_data or not lang_name:
        return None
    for row in table_data:
        if lang_name in row[0]:
            return handle_tts_single(lang_name, row[2])
    return None


# ══════════════════════════════════════════════════════════════════════════════
# UI helpers
# ══════════════════════════════════════════════════════════════════════════════

ALL_LANG_NAMES = list(ALL_LANGUAGES.keys())
INDIAN_LANG_NAMES = list(INDIAN_LANGUAGES.keys())
INTL_LANG_NAMES = list(INTERNATIONAL_LANGUAGES.keys())

DEFAULT_SELECTION = [
    "Hindi", "Bengali", "Tamil", "Telugu", "Malayalam", "Kannada",
    "Gujarati", "Marathi", "Punjabi",
    "English", "French", "German", "Spanish", "Japanese",
    "Chinese (Simplified)", "Arabic", "Russian", "Korean",
]
DEFAULT_SELECTION = [l for l in DEFAULT_SELECTION if l in ALL_LANGUAGES]

CUSTOM_CSS = """
/* ── Global ──────────────────────────────────────────────────────── */
body, .gradio-container {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}

/* ── Header banner ───────────────────────────────────────────────── */
.header-banner {
    background: linear-gradient(135deg, #1a1f36 0%, #0f3460 50%, #16213e 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 28px 36px;
    text-align: center;
    margin-bottom: 24px;
}
.header-banner h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #58a6ff, #79c0ff, #a5f3fc, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.header-banner p {
    color: #8b949e;
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.6;
}

/* ── Tab bar ─────────────────────────────────────────────────────── */
.tab-nav button {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #8b949e !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected {
    background: #1f6feb !important;
    color: #ffffff !important;
    border-color: #1f6feb !important;
}

/* ── Cards ───────────────────────────────────────────────────────── */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
}

/* ── Buttons ─────────────────────────────────────────────────────── */
button.primary {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    font-size: 1rem !important;
    cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(31,111,235,0.4) !important;
}

/* ── Inputs ──────────────────────────────────────────────────────── */
input, textarea, select {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}
input:focus, textarea:focus {
    border-color: #1f6feb !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(31,111,235,0.2) !important;
}

/* ── Dataframe ───────────────────────────────────────────────────── */
.dataframe table {
    background: #161b22 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
.dataframe th {
    background: #1f6feb !important;
    color: white !important;
    font-weight: 700 !important;
    padding: 10px 16px !important;
}
.dataframe td {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    padding: 10px 16px !important;
    border-bottom: 1px solid #21262d !important;
}
.dataframe tr:hover td { background: #161b22 !important; }

/* ── Status badges ───────────────────────────────────────────────── */
.status-ok  { color: #3fb950 !important; font-weight: 700; }
.status-err { color: #f85149 !important; font-weight: 700; }
.status-warn { color: #d29922 !important; font-weight: 700; }
"""

# ══════════════════════════════════════════════════════════════════════════════
# Build Gradio App
# ══════════════════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="🌐 Multilingual AI Translator",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.indigo,
            neutral_hue=gr.themes.colors.gray,
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
        ),
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-banner">
            <h1>🌐 Multilingual AI Translator</h1>
            <p>
                Speech-to-Text &nbsp;·&nbsp; Neural Machine Translation &nbsp;·&nbsp; Text-to-Speech<br>
                <strong>Whisper AST</strong> &nbsp;·&nbsp; <strong>Conformer</strong>
                &nbsp;·&nbsp; <strong>NLLB-200 Seq2Seq</strong> &nbsp;·&nbsp; <strong>gTTS</strong>
            </p>
        </div>
        """)

        # ── Shared state ──────────────────────────────────────────────────────
        state_table   = gr.State([])   # translation table rows
        state_transcript = gr.State("")

        with gr.Row():
            # ════════════════════════════════════════════════════════════════
            # LEFT COLUMN — Settings
            # ════════════════════════════════════════════════════════════════
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### ⚙️ Settings")

                whisper_size = gr.Dropdown(
                    label="🎤 ASR Model (Whisper)",
                    choices=list(WHISPER_MODEL_OPTIONS.keys()),
                    value="Base (recommended)",
                    info="Larger = more accurate, slower to load",
                )

                src_mode = gr.Radio(
                    label="🗣️ Source Language",
                    choices=["Auto-detect", "Manual"],
                    value="Auto-detect",
                )
                manual_src = gr.Dropdown(
                    label="Manual source language",
                    choices=ALL_LANG_NAMES,
                    value="English",
                    visible=False,
                )
                src_mode.change(
                    fn=lambda m: gr.update(visible=(m == "Manual")),
                    inputs=src_mode,
                    outputs=manual_src,
                )

                gr.Markdown("---")
                gr.Markdown("### 🎯 Target Languages")

                lang_group = gr.Radio(
                    label="Language group",
                    choices=["All Indian", "All International", "Custom"],
                    value="Custom",
                )
                selected_langs = gr.CheckboxGroup(
                    label="Select languages",
                    choices=ALL_LANG_NAMES,
                    value=DEFAULT_SELECTION,
                )

                def update_lang_group(group):
                    if group == "All Indian":
                        return gr.update(value=INDIAN_LANG_NAMES)
                    elif group == "All International":
                        return gr.update(value=INTL_LANG_NAMES)
                    return gr.update()

                lang_group.change(update_lang_group, inputs=lang_group, outputs=selected_langs)

                gr.Markdown("---")
                gr.Markdown("### 🔬 Quality")
                num_beams = gr.Slider(
                    label="Beam search width",
                    minimum=1, maximum=8, value=4, step=1,
                    info="Higher = better quality, slower",
                )
                enable_tts = gr.Checkbox(label="🔊 Generate audio (TTS)", value=True)

                gr.Markdown("---")
                gr.Markdown(f"**Device:** `{DEVICE.upper()}`")
                gr.Markdown(f"**Total languages:** `{len(ALL_LANGUAGES)}`")

            # ════════════════════════════════════════════════════════════════
            # RIGHT COLUMN — Main content
            # ════════════════════════════════════════════════════════════════
            with gr.Column(scale=3):
                with gr.Tabs():

                    # ── Tab 1: Microphone ─────────────────────────────────
                    with gr.Tab("🎤 Live Recording"):
                        gr.Markdown(
                            "Record your voice using the microphone below. "
                            "Click **▶ Translate** after recording."
                        )
                        mic_audio = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            label="🎙️ Record Audio",
                            waveform_options={"waveform_color": "#1f6feb"},
                        )
                        mic_btn = gr.Button("🌐 Translate Recording", variant="primary", size="lg")

                        gr.Markdown("---")
                        mic_status = gr.Markdown("*Ready — record audio and click Translate*")
                        mic_transcript = gr.Textbox(
                            label="📝 Transcription", lines=3, interactive=False
                        )
                        mic_src_info = gr.Markdown("")
                        mic_preview = gr.Audio(label="🔊 TTS Preview (first language)", type="filepath")
                        mic_table = gr.Dataframe(
                            headers=["Language", "NLLB Code", "Translation"],
                            label="📋 All Translations",
                            wrap=True,
                            interactive=False,
                        )

                        mic_btn.click(
                            fn=handle_audio_translate,
                            inputs=[
                                mic_audio, whisper_size, src_mode,
                                manual_src, selected_langs, num_beams, enable_tts,
                            ],
                            outputs=[mic_status, mic_transcript, mic_src_info, mic_table, mic_preview],
                            show_progress=True,
                        )

                    # ── Tab 2: Upload Audio ───────────────────────────────
                    with gr.Tab("📁 Upload Audio"):
                        gr.Markdown("Upload a WAV, MP3, M4A, OGG or FLAC file.")
                        upload_audio = gr.Audio(
                            sources=["upload"],
                            type="numpy",
                            label="📂 Upload Audio File",
                        )
                        upload_btn = gr.Button("🌐 Translate File", variant="primary", size="lg")

                        gr.Markdown("---")
                        upload_status   = gr.Markdown("*Upload a file and click Translate*")
                        upload_transcript = gr.Textbox(label="📝 Transcription", lines=3, interactive=False)
                        upload_src_info   = gr.Markdown("")
                        upload_preview    = gr.Audio(label="🔊 TTS Preview", type="filepath")
                        upload_table      = gr.Dataframe(
                            headers=["Language", "NLLB Code", "Translation"],
                            label="📋 All Translations",
                            wrap=True,
                            interactive=False,
                        )

                        upload_btn.click(
                            fn=handle_audio_translate,
                            inputs=[
                                upload_audio, whisper_size, src_mode,
                                manual_src, selected_langs, num_beams, enable_tts,
                            ],
                            outputs=[upload_status, upload_transcript, upload_src_info, upload_table, upload_preview],
                            show_progress=True,
                        )

                    # ── Tab 3: Text Input ─────────────────────────────────
                    with gr.Tab("✍️ Text Input"):
                        gr.Markdown(
                            "Type or paste text directly. "
                            "Select **Manual** source language in the sidebar."
                        )
                        text_src_lang = gr.Dropdown(
                            label="Source language",
                            choices=ALL_LANG_NAMES,
                            value="English",
                        )
                        text_input = gr.Textbox(
                            label="Input Text",
                            placeholder="Type or paste text here…",
                            lines=5,
                        )
                        text_btn = gr.Button("🌐 Translate Text", variant="primary", size="lg")

                        gr.Markdown("---")
                        text_status  = gr.Markdown("*Enter text and click Translate*")
                        text_preview = gr.Audio(label="🔊 TTS Preview", type="filepath")
                        text_table   = gr.Dataframe(
                            headers=["Language", "NLLB Code", "Translation"],
                            label="📋 All Translations",
                            wrap=True,
                            interactive=False,
                        )

                        text_btn.click(
                            fn=handle_text_translate,
                            inputs=[
                                text_input, text_src_lang,
                                selected_langs, num_beams, enable_tts,
                            ],
                            outputs=[text_status, text_table, text_preview],
                            show_progress=True,
                        )

                    # ── Tab 4: Listen to Any Language ─────────────────────
                    with gr.Tab("🔈 Play Any Language"):
                        gr.Markdown(
                            "After translating, select a language and enter the translated text "
                            "to hear it spoken aloud."
                        )
                        tts_lang_pick = gr.Dropdown(
                            label="Select language",
                            choices=ALL_LANG_NAMES,
                            value="Hindi",
                        )
                        tts_text_in = gr.Textbox(
                            label="Text to speak",
                            placeholder="Paste translated text here…",
                            lines=4,
                        )
                        tts_btn    = gr.Button("🔊 Generate Audio", variant="primary")
                        tts_out    = gr.Audio(label="Audio Output", type="filepath")

                        tts_btn.click(
                            fn=handle_tts_single,
                            inputs=[tts_lang_pick, tts_text_in],
                            outputs=tts_out,
                        )

                    # ── Tab 5: Export / Download ──────────────────────────
                    with gr.Tab("⬇️ Export"):
                        gr.Markdown(
                            "Export the latest translation results as JSON. "
                            "Run a translation first, then click **Generate JSON**."
                        )
                        export_transcript_in = gr.Textbox(
                            label="Transcription (auto-filled after translation)",
                            lines=2,
                            interactive=True,
                        )
                        export_src_in = gr.Textbox(
                            label="Source info", lines=1, interactive=True
                        )
                        export_table_in = gr.Dataframe(
                            headers=["Language", "NLLB Code", "Translation"],
                            label="Translations to export (auto-filled)",
                            interactive=False,
                        )
                        export_btn = gr.Button("📄 Generate JSON", variant="secondary")
                        export_out = gr.Code(label="JSON Output", language="json", lines=20)

                        export_btn.click(
                            fn=handle_export_json,
                            inputs=[export_table_in, export_transcript_in, export_src_in],
                            outputs=export_out,
                        )

                        # Wire mic tab results to export tab automatically
                        mic_table.change(
                            fn=lambda t: t, inputs=mic_table, outputs=export_table_in
                        )
                        mic_transcript.change(
                            fn=lambda t: t, inputs=mic_transcript, outputs=export_transcript_in
                        )
                        mic_src_info.change(
                            fn=lambda t: t, inputs=mic_src_info, outputs=export_src_in
                        )

                    # ── Tab 6: Architecture Info ──────────────────────────
                    with gr.Tab("ℹ️ Architecture"):
                        gr.Markdown("""
## 🧠 Model Architecture

### 🟦 Audio-Spectrogram Transformer (Whisper)
```
Raw Audio (16 kHz)
    ↓
Log-Mel Spectrogram (80 bins, 25ms window, 10ms hop)
    ↓
2D Convolutional Stem → Patch Embeddings
    ↓
Transformer Encoder  (Multi-Head Attention + FFN) × N layers
    ↓
Auto-regressive Decoder (cross-attention over encoder output)
    ↓
Transcribed Text + Language Code
```

### 🟧 Conformer (Fallback ASR)
```
Audio Features
    ↓
[Feed-Forward → Multi-Head Self-Attention →
 Convolution Module (local) → Feed-Forward] × N
    ↓
CTC Decoder → Text
```
*Captures both local phoneme patterns (Conv) and global context (Attention)*

### 🟩 NLLB-200 Seq2Seq Encoder-Decoder
```
Source Text + src_lang token
    ↓
12-layer Bidirectional Transformer Encoder
    ↓
12-layer Decoder + Cross-Attention
(forced BOS = target_lang token)
    ↓
Beam Search (width=4) → Translated Text
```
*Supports 200 languages via FLORES+ codes. Distilled from 54B parameter model.*

### 🔈 gTTS Text-to-Speech
```
Translated Text
    ↓
Google Neural TTS API (40+ languages)
    ↓
MP3 Audio Stream
```

---

## 📊 Language Coverage
| | Count |
|---|---|
| 🇮🇳 Indian Languages | 22 |
| 🌍 International Languages | 27 |
| **Total** | **49** |
                        """)

        # ── Footer ────────────────────────────────────────────────────────────
        gr.HTML(f"""
        <div style="text-align:center; color:#6e7681; font-size:0.82rem;
                    padding:16px 0; border-top:1px solid #30363d; margin-top:24px;">
            🌐 Multilingual AI Translator &nbsp;·&nbsp;
            Whisper AST · NLLB-200 · gTTS &nbsp;·&nbsp;
            Running on <strong>{DEVICE.upper()}</strong> &nbsp;·&nbsp;
            49 Languages Supported
        </div>
        """)

    return demo


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Starting Multilingual AI Translator (Gradio)...")
    print(f"  Device : {DEVICE.upper()}")
    print(f"  Languages : {len(ALL_LANGUAGES)}")
    print(f"  ASR model : {ASR_MODEL}")
    print(f"  Translation model : {TRANSLATION_MODEL}")
    print()

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
    )
