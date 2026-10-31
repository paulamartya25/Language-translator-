"""
ℹ️ About Page
=============
Architecture overview, model descriptions, and project information.
"""

import streamlit as st


def render_about_page():
    """Render the About / Architecture page."""

    st.markdown("## 🧠 Architecture & Models")

    # ── Architecture diagram ───────────────────────────────────────────────────
    st.markdown("""
```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTILINGUAL AI TRANSLATOR                       │
│                                                                     │
│  🎤 Audio Input                                                     │
│      ↓                                                              │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  AUDIO PREPROCESSING                                    │        │
│  │  • Resample to 16 kHz                                   │        │
│  │  • 80-channel Log-Mel Spectrogram                       │        │
│  │  • 30s window chunking (2s overlap)                     │        │
│  └─────────────────────────────────────────────────────────┘        │
│      ↓                                                              │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  AUDIO-SPECTROGRAM TRANSFORMER (AST)                    │        │
│  │  Model: openai/whisper-large-v3                         │        │
│  │  • 2D Conv stem → patch embeddings                      │        │
│  │  • 32-layer Transformer Encoder                         │        │
│  │  • Auto-regressive Decoder                              │        │
│  │  Output: Transcribed text + language code               │        │
│  └─────────────────────────────────────────────────────────┘        │
│      ↓ (fallback for noisy audio)                                   │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  CONFORMER ASR                                          │        │
│  │  Model: facebook/wav2vec2-conformer-rel-pos-large       │        │
│  │  • Conv Module (local patterns) +                       │        │
│  │    Multi-Head Self-Attention (global context)           │        │
│  │  • CTC decoder                                          │        │
│  └─────────────────────────────────────────────────────────┘        │
│      ↓                                                              │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  SEQ2SEQ ENCODER-DECODER (NLLB-200)                     │        │
│  │  Model: facebook/nllb-200-distilled-600M                │        │
│  │  • 12-layer Encoder (source language)                   │        │
│  │  • 12-layer Decoder + cross-attention                   │        │
│  │  • Forced BOS token → target language steering          │        │
│  │  • 200 languages via FLORES+ codes                      │        │
│  │  Output: Translated text (×49 languages)                │        │
│  └─────────────────────────────────────────────────────────┘        │
│      ↓                                                              │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  TEXT-TO-SPEECH                                         │        │
│  │  Engine: gTTS (Google Neural TTS)                       │        │
│  │  Output: MP3 audio bytes per language                   │        │
│  └─────────────────────────────────────────────────────────┘        │
│      ↓                                                              │
│  🖥️  Streamlit UI — Cards + Audio Players                           │
└─────────────────────────────────────────────────────────────────────┘
```
    """)

    st.divider()

    # ── Model cards ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🟦 Whisper (AST)")
        st.markdown("""
**OpenAI Whisper Large v3**

- **Architecture**: Audio-Spectrogram Transformer
- **Input**: 80-channel Log-Mel Spectrogram
- **Encoder**: 32 Transformer layers
- **Decoder**: Auto-regressive with multilingual vocabulary
- **Languages**: 99 spoken languages
- **Strengths**: Extremely robust to accents, noise, and multilingual speech

**How it works:**
1. Audio → STFT → 80-channel Mel filterbank
2. 2D Conv stem extracts spatial features from spectrogram
3. Transformer encoder attends across time
4. Decoder generates tokens with language prefix tokens
        """)

    with col2:
        st.markdown("### 🟧 Conformer")
        st.markdown("""
**Facebook wav2vec2-Conformer**

- **Architecture**: Conformer (CNN + Transformer hybrid)
- **Key Block**: Conv → MHSA → Conv → FFN sandwich
- **Purpose**: Noise-robust phoneme boundary detection

**Conformer Block:**
```
Input → LayerNorm → Feed-Forward (½)
      → Multi-Head Self-Attention
      → Convolution Module
        (Pointwise → GLU → Depthwise → BN → Swish → Pointwise)
      → Feed-Forward (½)
      → LayerNorm → Output
```
**Advantage over pure Transformer:**
Convolution captures local acoustic patterns (phonemes, coarticulation) while attention captures global context (sentence structure).
        """)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 🟩 NLLB-200 (Seq2Seq)")
        st.markdown("""
**Meta NLLB-200 Distilled 600M**

- **Architecture**: Seq2Seq Encoder-Decoder Transformer
- **Parameters**: 600M (distilled from 54B model)
- **Languages**: 200 via FLORES+ codes
- **Tokenizer**: SentencePiece multilingual

**Translation Process:**
1. Tokenize source text with src_lang set
2. Encoder: bidirectional self-attention over source tokens
3. Forced BOS token = target language code
4. Decoder: cross-attends encoder, generates target tokens
5. Beam search (width=4) for best translation
        """)

    with col4:
        st.markdown("### 🔈 gTTS (Text-to-Speech)")
        st.markdown("""
**Google Text-to-Speech**

- **Engine**: Google Neural TTS API
- **Languages**: 40+ languages
- **Output**: MP3 audio stream
- **Quality**: Neural voices for Indian & international languages

**Supported Indian Language TTS:**
- Hindi, Bengali, Tamil, Telugu, Malayalam
- Gujarati, Kannada, Marathi, Punjabi, Urdu
- Odia, Assamese, Nepali, Sindhi, Sanskrit

**Fallback for rare languages:**
Languages without direct gTTS support use the closest related language (e.g., Maithili → Hindi TTS)
        """)

    st.divider()

    # ── Language stats ────────────────────────────────────────────────────────
    st.markdown("### 🌏 Language Coverage")
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    try:
        from config.languages import INDIAN_LANGUAGES, INTERNATIONAL_LANGUAGES
        c1, c2, c3 = st.columns(3)
        c1.metric("Indian Languages", len(INDIAN_LANGUAGES))
        c2.metric("International Languages", len(INTERNATIONAL_LANGUAGES))
        c3.metric("Total Languages", len(INDIAN_LANGUAGES) + len(INTERNATIONAL_LANGUAGES))

        with st.expander("🇮🇳 All Indian Languages"):
            for lang, info in INDIAN_LANGUAGES.items():
                st.markdown(
                    f"- {info['flag']} **{lang}** — Script: {info.get('script','?')}, "
                    f"Region: {info.get('region','?')}, NLLB: `{info['nllb_code']}`"
                )

        with st.expander("🌍 All International Languages"):
            for lang, info in INTERNATIONAL_LANGUAGES.items():
                st.markdown(
                    f"- {info['flag']} **{lang}** — Script: {info.get('script','?')}, "
                    f"Region: {info.get('region','?')}, NLLB: `{info['nllb_code']}`"
                )
    except ImportError:
        pass

    st.divider()
    st.markdown("""
### 📦 Tech Stack
| Component | Library |
|---|---|
| UI Framework | Streamlit |
| ASR | 🤗 Transformers + Whisper |
| Conformer | 🤗 Transformers + wav2vec2 |
| Translation | 🤗 Transformers + NLLB-200 |
| TTS | gTTS |
| Audio I/O | sounddevice, librosa, soundfile, pydub |
| ML Backend | PyTorch |
| Tokenization | SentencePiece |
    """)
