# 🌐 Multilingual AI Language Translator

A production-quality, end-to-end multilingual translation system that converts
speech (or text) into 40+ Indian and international languages — with audio
playback for every translation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Streamlit Web UI (app/main.py)                 │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────────────────┐ │
│  │ 🎤 Live    │  │ 📁 Upload    │  │ ✍️ Text Input               │ │
│  │ Recording  │  │ Audio File   │  │ (skip ASR)                  │ │
│  └─────┬──────┘  └──────┬───────┘  └──────────────┬──────────────┘ │
└────────┼────────────────┼─────────────────────────┼───────────────-┘
         │                │                          │
         ▼                ▼                          │
┌─────────────────────────────────────────────────┐ │
│              ASR Pipeline                        │ │
│  ┌──────────────────────────────────────────┐   │ │
│  │  AudioProcessor                          │   │ │
│  │  · Resample → 16 kHz mono               │   │ │
│  │  · Chunk into 30-second windows          │   │ │
│  │  · Compute 80-ch Log-Mel Spectrogram     │   │ │
│  └──────────────────┬───────────────────────┘   │ │
│                     │                            │ │
│  ┌──────────────────▼───────────────────────┐   │ │
│  │  WhisperASR (Audio-Spectrogram Transform.)│   │ │
│  │  openai/whisper-base  (configurable)     │   │ │
│  │  · Encoder: Conv stem + Transformer      │   │ │
│  │  · Decoder: Cross-attention, auto-regr.  │   │ │
│  │  · Output: text + detected language      │   │ │
│  └──────────────────┬───────────────────────┘   │ │
│                     │  (if empty / noisy)        │ │
│  ┌──────────────────▼───────────────────────┐   │ │
│  │  ConformerASR (Fallback)                 │   │ │
│  │  facebook/wav2vec2-conformer-rel-pos-large│   │ │
│  │  · Conv encoder + Conformer blocks       │   │ │
│  │  · CTC decoding                          │   │ │
│  └──────────────────┬───────────────────────┘   │ │
└─────────────────────┼───────────────────────────┘ │
                      │  transcription + lang code   │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────┐
│              Translation Pipeline                  │
│  ┌──────────────────────────────────────────────┐  │
│  │  NLLBTranslator (Seq2Seq Encoder-Decoder)    │  │
│  │  facebook/nllb-200-distilled-600M            │  │
│  │  · Tokenize with FLORES+ lang codes          │  │
│  │  · Encode source → Decode with forced BOS    │  │
│  │  · Translate to N languages in sequence      │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────┬───────────────────────────┘
                         │  {lang_name: translated_text}
                         ▼
┌────────────────────────────────────────────────────┐
│              TTS Pipeline                          │
│  ┌──────────────────────────────────────────────┐  │
│  │  TTSEngine (gTTS)                            │  │
│  │  · Synthesize MP3 audio for each language    │  │
│  │  · Returns bytes → st.audio() playback       │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

---

## Supported Languages

### 🇮🇳 Indian Languages (22)

| Language   | NLLB Code    | Script      |
|------------|--------------|-------------|
| Hindi      | hin_Deva     | Devanagari  |
| Bengali    | ben_Beng     | Bengali     |
| Telugu     | tel_Telu     | Telugu      |
| Marathi    | mar_Deva     | Devanagari  |
| Tamil      | tam_Taml     | Tamil       |
| Urdu       | urd_Arab     | Nastaliq    |
| Gujarati   | guj_Gujr     | Gujarati    |
| Kannada    | kan_Knda     | Kannada     |
| Odia       | ory_Orya     | Odia        |
| Malayalam  | mal_Mlym     | Malayalam   |
| Punjabi    | pan_Guru     | Gurmukhi    |
| Assamese   | asm_Beng     | Bengali     |
| Maithili   | mai_Deva     | Devanagari  |
| Sanskrit   | san_Deva     | Devanagari  |
| Nepali     | npi_Deva     | Devanagari  |
| Konkani    | kok_Deva     | Devanagari  |
| Sindhi     | snd_Arab     | Arabic      |
| Kashmiri   | kas_Arab     | Nastaliq    |
| Manipuri   | mni_Beng     | Bengali     |
| Dogri      | dgo_Deva     | Devanagari  |
| Bodo       | brx_Deva     | Devanagari  |
| Santali    | sat_Olck     | Ol Chiki    |

### 🌍 International Languages (26+)

English, Japanese, Chinese (Simplified/Traditional), German, French, Dutch,
Spanish, Portuguese, Russian, Arabic, Korean, Italian, Turkish, Vietnamese,
Thai, Swedish, Polish, Indonesian, Greek, Hebrew, Swahili, Malay, Filipino,
Ukrainian, Czech, Romanian.

---

## Models

| Component     | Model                                    | Params | Task                         |
|---------------|------------------------------------------|--------|------------------------------|
| ASR (primary) | `openai/whisper-base`                    | 74M    | Speech → Text + lang detect  |
| ASR (fallback)| `facebook/wav2vec2-conformer-rel-pos-large` | 600M | Noisy speech → Text (CTC)  |
| Translation   | `facebook/nllb-200-distilled-600M`       | 600M   | Text → 200 language targets  |
| TTS           | `gTTS` (Google TTS API)                  | Cloud  | Text → MP3 audio             |

> **Tip:** Switch Whisper to `openai/whisper-large-v3` in the sidebar for
> best ASR quality (requires ~3 GB VRAM or patience on CPU).

---

## Installation

### Prerequisites

- Python 3.10+
- `ffmpeg` in PATH (required by pydub for MP3 decoding)
  - Windows: `winget install ffmpeg` or download from https://ffmpeg.org

### Steps

```bash
# 1. Clone / enter the project directory
cd LANGUAGE_TRANSLATER

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/main.py
```

The first run will download model weights from HuggingFace Hub
(~1-2 GB total; cached locally after first download).

---

## Usage

### 🎤 Live Recording
1. Open the **Live Recording** tab.
2. Set the recording duration (3–60 seconds).
3. Click **Start Recording** and speak clearly.
4. Click **Translate Recorded Audio**.

### 📁 Upload Audio
1. Open the **Upload Audio** tab.
2. Upload a WAV/MP3/M4A/OGG/FLAC file.
3. Click **Translate Audio File**.

### ✍️ Text Input
1. Open the **Text Input** tab.
2. Select source language in the sidebar (Manual mode).
3. Type or paste text.
4. Click **Translate Text**.

### Results
- Each translated language appears in a card with the national flag.
- Click the ▶ play button beneath each card to hear the TTS audio.
- Click **Download All Translations (JSON)** to export results.

---

## Hardware Requirements

| Mode       | RAM   | VRAM   | Speed (per language) |
|------------|-------|--------|----------------------|
| CPU only   | 8 GB  | —      | ~15–30 s             |
| GPU (CUDA) | 8 GB  | 4 GB+  | ~1–3 s               |

> On CPU, consider using `openai/whisper-tiny` or `whisper-base` for faster
> response and `facebook/nllb-200-distilled-600M` (already the smallest NLLB).

---

## Project Structure

```
LANGUAGE_TRANSLATER/
├── app/
│   ├── main.py                   ← Streamlit entry point
│   └── ui/
│       └── styles.css            ← Custom dark-theme CSS
├── config/
│   ├── languages.py              ← Language registry (NLLB codes, flags)
│   └── settings.py               ← Global model & app settings
├── models/
│   ├── asr/
│   │   ├── whisper_asr.py        ← Whisper ASR pipeline
│   │   └── conformer_asr.py      ← wav2vec2-Conformer fallback
│   ├── translation/
│   │   └── nllb_translator.py    ← NLLB-200 translation
│   └── tts/
│       └── gtts_engine.py        ← gTTS synthesis
├── pipeline/
│   ├── audio_processor.py        ← Recording, mel-spec, chunking
│   ├── language_detector.py      ← Language detection wrapper
│   └── translator_pipeline.py    ← End-to-end orchestration
├── utils/
│   ├── audio_utils.py            ← Audio format helpers
│   └── text_utils.py             ← Text cleaning, script detection
├── requirements.txt
└── README.md
```

---

## License

This project is for educational and research purposes.
Model weights are subject to their respective licenses:
- Whisper: MIT
- NLLB-200: CC-BY-NC 4.0
- gTTS: MIT
