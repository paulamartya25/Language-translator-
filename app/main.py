"""
🌐 Multilingual AI Translator — Streamlit Application
======================================================
Entry point: streamlit run app/main.py

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
from typing import Dict, List, Optional

import numpy as np
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Project imports ────────────────────────────────────────────────────────────
from config.languages import (
    INDIAN_LANGUAGES,
    INTERNATIONAL_LANGUAGES,
    ALL_LANGUAGES,
    get_nllb_code,
    whisper_to_nllb,
)
from config.settings import (
    APP_TITLE,
    APP_ICON,
    APP_DESCRIPTION,
    ASR_MODEL,
    TRANSLATION_MODEL,
    DEVICE,
    SAMPLE_RATE,
    COLUMNS_PER_ROW,
    DEFAULT_RECORD_SECONDS,
    MAX_AUDIO_RECORD_SECONDS,
    WHISPER_MODEL_OPTIONS,
    TTS_SLOW_MODE,
)
from utils.audio_utils import bytes_to_audio_array, audio_array_to_bytes, is_silent
from utils.text_utils import clean_text, detect_script

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multilingual AI Translator",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ───────────────────────────────────────────────────────────────────
_CSS_PATH = Path(__file__).parent / "ui" / "styles.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cached model loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_whisper(model_id: str, device: str):
    """Load and cache Whisper ASR model."""
    from models.asr.whisper_asr import WhisperASR
    return WhisperASR(model_id=model_id, device=device)


@st.cache_resource(show_spinner=False)
def _load_conformer(device: str):
    """Load and cache Conformer fallback ASR model."""
    from models.asr.conformer_asr import ConformerASR
    try:
        return ConformerASR(device=device)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_translator(model_id: str, device: str):
    """Load and cache NLLB-200 translation model."""
    from models.translation.nllb_translator import NLLBTranslator
    return NLLBTranslator(model_id=model_id, device=device)


@st.cache_resource(show_spinner=False)
def _load_tts():
    """Load and cache TTS engine."""
    from models.tts.gtts_engine import TTSEngine
    return TTSEngine(slow=TTS_SLOW_MODE)


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def _inject_html(html: str):
    st.markdown(html, unsafe_allow_html=True)


def _render_header():
    _inject_html("""
    <div class="translator-header">
        <h1>🌐 Multilingual AI Translator</h1>
        <p>Speech-to-Text · Neural Machine Translation · Text-to-Speech<br>
           Powered by Whisper ASR · NLLB-200 · gTTS</p>
    </div>
    """)


def _render_transcription_card(text: str, lang_name: str, lang_code: str):
    script = detect_script(text)
    _inject_html(f"""
    <div class="transcription-card">
        <div class="label">📝 Transcription</div>
        <div class="text">{text}</div>
        <div class="meta">
            🗣️ Detected language: <strong>{lang_name}</strong>
            &nbsp;|&nbsp; Code: <code>{lang_code}</code>
            &nbsp;|&nbsp; Script: {script}
        </div>
    </div>
    """)


def _render_lang_card(lang_name: str, info: dict, translated_text: str):
    flag = info.get("flag", "🌐")
    _inject_html(f"""
    <div class="lang-card">
        <div class="lang-header">
            <span class="lang-flag">{flag}</span>
            <span class="lang-name">{lang_name}</span>
        </div>
        <div class="lang-text">{translated_text or "<em style='color:#8892A4'>Translation unavailable</em>"}</div>
    </div>
    """)


def _record_audio(duration: float) -> Optional[np.ndarray]:
    """Record from microphone using sounddevice."""
    try:
        import sounddevice as sd
        placeholder = st.empty()
        for i in range(int(duration), 0, -1):
            placeholder.markdown(
                f'<div class="recording-indicator">🔴 Recording… {i}s remaining</div>',
                unsafe_allow_html=True,
            )
            time.sleep(1)
        placeholder.empty()

        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return audio.flatten()
    except Exception as exc:
        st.error(f"Microphone recording failed: {exc}")
        return None


def _translations_to_json(result: dict) -> str:
    """Serialize translation results to JSON (excluding audio bytes)."""
    export = {
        "transcription": result.get("transcription", ""),
        "source_language": result.get("source_language", ""),
        "translations": {
            lang: {"text": data["text"], "nllb_code": data.get("nllb_code", "")}
            for lang, data in result.get("translations", {}).items()
        },
    }
    return json.dumps(export, ensure_ascii=False, indent=2)


def _get_target_lang_list(selection: List[str]) -> List[str]:
    """Return the list of target language names from sidebar selection."""
    return [lang for lang in selection if lang in ALL_LANGUAGES]


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def _build_sidebar() -> dict:
    """Render sidebar controls and return settings dict."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.divider()

        # Whisper model selector
        st.markdown("### 🎤 ASR Model")
        whisper_label = st.selectbox(
            "Whisper size",
            options=list(WHISPER_MODEL_OPTIONS.keys()),
            index=1,
            help="Larger models are more accurate but slower to load.",
        )
        whisper_model_id = WHISPER_MODEL_OPTIONS[whisper_label]

        # Source language
        st.markdown("### 🗣️ Source Language")
        src_mode = st.radio(
            "Detection mode",
            ["Auto-detect", "Manual"],
            index=0,
        )
        manual_src_lang = None
        if src_mode == "Manual":
            manual_src_lang = st.selectbox(
                "Source language",
                options=list(ALL_LANGUAGES.keys()),
                index=0,
            )

        st.divider()

        # Target languages
        st.markdown("### 🎯 Target Languages")
        group = st.radio(
            "Language group",
            ["All Indian", "All International", "Custom Selection"],
            index=2,
        )

        if group == "All Indian":
            selected_langs = list(INDIAN_LANGUAGES.keys())
        elif group == "All International":
            selected_langs = list(INTERNATIONAL_LANGUAGES.keys())
        else:
            # Custom multiselect – default to common set
            default_custom = [
                "Hindi", "Bengali", "Tamil", "Telugu", "Malayalam",
                "Kannada", "Gujarati", "Marathi", "Punjabi",
                "English", "French", "German", "Spanish", "Japanese",
                "Chinese (Simplified)", "Arabic", "Russian", "Korean",
            ]
            selected_langs = st.multiselect(
                "Choose languages",
                options=list(ALL_LANGUAGES.keys()),
                default=[l for l in default_custom if l in ALL_LANGUAGES],
            )

        st.divider()

        # TTS toggle
        st.markdown("### 🔊 Text-to-Speech")
        enable_tts = st.checkbox("Generate audio for each translation", value=True)

        st.divider()

        # Translation quality
        st.markdown("### 🔬 Translation Quality")
        num_beams = st.slider(
            "Beam search width",
            min_value=1,
            max_value=8,
            value=4,
            help="Higher = better quality but slower.",
        )

        st.divider()

        # Device info
        st.markdown(f"**Device:** `{DEVICE.upper()}`")
        st.markdown(f"**Languages selected:** `{len(selected_langs)}`")

        return {
            "whisper_model_id": whisper_model_id,
            "src_mode": src_mode,
            "manual_src_lang": manual_src_lang,
            "selected_langs": selected_langs,
            "enable_tts": enable_tts,
            "num_beams": num_beams,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Model loading (with status messages)
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_models(whisper_model_id: str):
    """Load models into session state if not already loaded."""
    need_reload = (
        "whisper" not in st.session_state
        or st.session_state.get("loaded_whisper_id") != whisper_model_id
    )

    if need_reload:
        with st.spinner("⏳ Loading Whisper ASR model… (first run takes a while)"):
            try:
                st.session_state["whisper"] = _load_whisper(whisper_model_id, DEVICE)
                st.session_state["loaded_whisper_id"] = whisper_model_id
            except Exception as exc:
                st.error(f"Failed to load Whisper: {exc}")
                st.session_state["whisper"] = None

    if "translator" not in st.session_state:
        with st.spinner("⏳ Loading NLLB-200 translation model…"):
            try:
                st.session_state["translator"] = _load_translator(TRANSLATION_MODEL, DEVICE)
            except Exception as exc:
                st.error(f"Failed to load NLLB-200: {exc}")
                st.session_state["translator"] = None

    if "tts" not in st.session_state:
        st.session_state["tts"] = _load_tts()


# ══════════════════════════════════════════════════════════════════════════════
# Processing
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(
    audio_array: Optional[np.ndarray],
    text_input: Optional[str],
    settings: dict,
) -> Optional[dict]:
    """Run the full translator pipeline and return result dict."""
    from pipeline.translator_pipeline import TranslatorPipeline

    whisper = st.session_state.get("whisper")
    translator = st.session_state.get("translator")
    tts = st.session_state.get("tts")

    if translator is None:
        st.error("Translation model is not loaded. Please check your environment.")
        return None

    pipeline = TranslatorPipeline(
        asr_model=whisper,
        translator=translator,
        tts_engine=tts,
    )

    target_langs = settings["selected_langs"]
    num_beams = settings["num_beams"]
    enable_tts = settings["enable_tts"]

    progress_bar = st.progress(0)
    status_text = st.empty()

    def _progress(step, total, label):
        pct = int(step / max(total, 1) * 100)
        progress_bar.progress(pct)
        status_text.markdown(f"**{label}**")

    try:
        if audio_array is not None:
            src_override = None
            if settings["src_mode"] == "Manual" and settings["manual_src_lang"]:
                # Get whisper-compatible lang code from manual selection
                lang_info = ALL_LANGUAGES.get(settings["manual_src_lang"], {})
                src_override = lang_info.get("gtts_lang")  # approx. BCP-47

            result = pipeline.process_audio(
                audio_input=audio_array,
                target_languages=target_langs,
                sample_rate=SAMPLE_RATE,
                source_lang_override=src_override,
                synthesize_audio=enable_tts,
                num_beams=num_beams,
                progress_callback=_progress,
            )
        else:
            # Text input mode
            src_lang_nllb = "eng_Latn"
            if settings["manual_src_lang"]:
                src_lang_nllb = get_nllb_code(settings["manual_src_lang"]) or "eng_Latn"

            result = pipeline.process_text(
                text=text_input,
                source_lang_nllb=src_lang_nllb,
                target_languages=target_langs,
                synthesize_audio=enable_tts,
                num_beams=num_beams,
                progress_callback=_progress,
            )

    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        logger.exception("Pipeline failed")
        return None
    finally:
        progress_bar.empty()
        status_text.empty()

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Results display
# ══════════════════════════════════════════════════════════════════════════════

def _display_results(result: dict):
    """Render transcription + translation cards."""
    if not result:
        return

    if result.get("error"):
        st.error(f"❌ {result['error']}")
        return

    # ── Transcription ──────────────────────────────────────────────────────
    transcription = result.get("transcription", "")
    if transcription:
        _render_transcription_card(
            text=transcription,
            lang_name=result.get("source_language", "Unknown"),
            lang_code=result.get("source_lang_code", "?"),
        )

    translations = result.get("translations", {})
    if not translations:
        st.info("No translations were generated.")
        return

    # ── Download button ────────────────────────────────────────────────────
    json_str = _translations_to_json(result)
    st.download_button(
        label="⬇️ Download All Translations (JSON)",
        data=json_str,
        file_name="translations.json",
        mime="application/json",
    )

    st.divider()

    # ── Indian Languages section ───────────────────────────────────────────
    indian_results = {k: v for k, v in translations.items() if k in INDIAN_LANGUAGES}
    intl_results   = {k: v for k, v in translations.items() if k in INTERNATIONAL_LANGUAGES}

    def _render_group(group_dict: dict, section_label: str):
        if not group_dict:
            return
        _inject_html(f'<div class="section-label">{section_label}</div>')
        items = list(group_dict.items())
        for i in range(0, len(items), COLUMNS_PER_ROW):
            cols = st.columns(COLUMNS_PER_ROW)
            for j, (lang_name, data) in enumerate(items[i : i + COLUMNS_PER_ROW]):
                lang_info = ALL_LANGUAGES.get(lang_name, {})
                with cols[j]:
                    _render_lang_card(lang_name, lang_info, data.get("text", ""))
                    # Audio player
                    audio_bytes = data.get("audio")
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

    _render_group(indian_results,  "🇮🇳 Indian Languages")
    _render_group(intl_results,    "🌍 International Languages")


# ══════════════════════════════════════════════════════════════════════════════
# Main app
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    _render_header()

    # Sidebar settings
    settings = _build_sidebar()

    # Load models
    _ensure_models(settings["whisper_model_id"])

    # ── Input Tabs ─────────────────────────────────────────────────────────────
    tab_mic, tab_upload, tab_text, tab_viz, tab_history, tab_about = st.tabs([
        "🎤 Live Recording",
        "📁 Upload Audio",
        "✍️ Text Input",
        "📊 Visualizer",
        "📜 History",
        "ℹ️ About",
    ])

    audio_array: Optional[np.ndarray] = None
    text_input: Optional[str] = None
    trigger_pipeline = False

    # ── Tab 1: Microphone Recording ────────────────────────────────────────────
    with tab_mic:
        st.markdown("### 🎙️ Record from Microphone")
        st.info(
            "Click **Start Recording**, speak clearly, then click **Translate**. "
            "Make sure your microphone is enabled."
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            rec_duration = st.slider(
                "Recording duration (seconds)",
                min_value=3,
                max_value=MAX_AUDIO_RECORD_SECONDS,
                value=DEFAULT_RECORD_SECONDS,
                step=1,
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔴 Start Recording", use_container_width=True, type="primary"):
                with st.spinner("Getting ready to record…"):
                    time.sleep(0.5)
                recorded = _record_audio(float(rec_duration))
                if recorded is not None and not is_silent(recorded):
                    st.session_state["mic_audio"] = recorded
                    wav_bytes = audio_array_to_bytes(recorded, SAMPLE_RATE)
                    st.session_state["mic_wav"] = wav_bytes
                    st.success(f"✅ Recorded {rec_duration}s of audio!")
                else:
                    st.warning("⚠️ No audio detected. Check your microphone and try again.")

        # Playback recorded audio
        if "mic_wav" in st.session_state:
            st.markdown("**Preview recorded audio:**")
            st.audio(st.session_state["mic_wav"], format="audio/wav")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🌐 Translate Recorded Audio", use_container_width=True, type="primary"):
                    audio_array = st.session_state["mic_audio"]
                    trigger_pipeline = True
            with col_b:
                if st.button("🗑️ Clear Recording", use_container_width=True):
                    del st.session_state["mic_audio"]
                    del st.session_state["mic_wav"]
                    st.rerun()

        # Streamlit file-uploader alternative for mic (audio blob from browser)
        st.divider()
        st.markdown("#### 📎 Or upload a pre-recorded clip")
        mic_alt_file = st.file_uploader(
            "Upload WAV/MP3/M4A (alternative mic source)",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            key="mic_alt",
        )
        if mic_alt_file is not None:
            raw = mic_alt_file.read()
            arr, sr = bytes_to_audio_array(raw)
            # Resample to 16 kHz if needed
            if sr != SAMPLE_RATE:
                from utils.audio_utils import resample_audio
                arr = resample_audio(arr, sr, SAMPLE_RATE)
            if st.button("🌐 Translate Uploaded (mic tab)", key="mic_alt_btn", type="primary"):
                audio_array = arr
                trigger_pipeline = True

    # ── Tab 2: File Upload ─────────────────────────────────────────────────────
    with tab_upload:
        st.markdown("### 📂 Upload an Audio File")
        st.info("Supported formats: WAV, MP3, M4A, OGG, FLAC")

        uploaded_file = st.file_uploader(
            "Choose audio file",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            key="file_upload",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            st.audio(file_bytes, format=f"audio/{uploaded_file.name.split('.')[-1]}")

            arr, sr = bytes_to_audio_array(file_bytes)
            if sr != SAMPLE_RATE:
                from utils.audio_utils import resample_audio
                arr = resample_audio(arr, sr, SAMPLE_RATE)

            duration_s = len(arr) / SAMPLE_RATE
            st.caption(
                f"📊 File: **{uploaded_file.name}** | "
                f"Duration: **{duration_s:.1f}s** | "
                f"Sample rate: **{sr} Hz**"
            )

            if st.button("🌐 Translate Audio File", use_container_width=True, type="primary"):
                audio_array = arr
                trigger_pipeline = True

    # ── Tab 3: Text Input ──────────────────────────────────────────────────────
    with tab_text:
        st.markdown("### ✍️ Enter Text Directly")

        if settings["src_mode"] == "Auto-detect":
            st.info(
                "For text input, please select **Manual** source language in the sidebar "
                "so NLLB-200 knows which language to translate from."
            )

        user_text = st.text_area(
            "Input text",
            placeholder="Type or paste text here…",
            height=180,
            key="text_area",
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            word_count = len(user_text.split()) if user_text.strip() else 0
            st.caption(f"📝 {word_count} words · {len(user_text)} characters")
        with col2:
            if st.button("🌐 Translate Text", use_container_width=True, type="primary"):
                cleaned = clean_text(user_text)
                if not cleaned:
                    st.warning("Please enter some text first.")
                else:
                    text_input = cleaned
                    trigger_pipeline = True

    # ── Run Pipeline ───────────────────────────────────────────────────────────
    if trigger_pipeline:
        if not settings["selected_langs"]:
            st.warning("⚠️ Please select at least one target language in the sidebar.")
        else:
            st.divider()
            st.markdown("## 🔄 Processing…")

            with st.spinner("Running AI pipeline…"):
                result = _run_pipeline(audio_array, text_input, settings)

            if result:
                st.session_state["last_result"] = result
                # Save to history
                try:
                    from app.pages.history import save_to_history
                    save_to_history(result)
                except Exception:
                    pass

    # ── Display last results ───────────────────────────────────────────────────
    if "last_result" in st.session_state:
        st.divider()
        st.markdown("## 📊 Results")
        _display_results(st.session_state["last_result"])

        if st.button("🔄 Clear Results"):
            del st.session_state["last_result"]
            st.rerun()

    # ── Visualizer Tab ─────────────────────────────────────────────────────────
    with tab_viz:
        from app.pages.visualizer import render_visualizer_page
        viz_audio = st.session_state.get("mic_audio")
        viz_sr = SAMPLE_RATE
        render_visualizer_page(viz_audio, viz_sr)

    # ── History Tab ────────────────────────────────────────────────────────────
    with tab_history:
        from app.pages.history import render_history_page
        render_history_page()

    # ── About Tab ──────────────────────────────────────────────────────────────
    with tab_about:
        from app.pages.about import render_about_page
        render_about_page()

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#8892A4;font-size:0.82rem;'>"
        "🌐 Multilingual AI Translator &nbsp;·&nbsp; "
        "Whisper ASR · NLLB-200 · gTTS &nbsp;·&nbsp; "
        f"Running on <strong>{DEVICE.upper()}</strong>"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
