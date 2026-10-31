# 🌐 Multilingual AI Language Translator

> **Speak → Transcribe → Translate → Hear** — in 49 languages, powered by 3 custom deep learning architectures built from scratch in PyTorch.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 🎙️ **Live Recording** | Record microphone audio → auto-transcribe → translate |
| 📝 **Text Input** | Type English text → translate + hear in any language |
| 🌍 **49 Languages** | 22 Indian + 27 International, all with voice output |
| 🏗️ **Custom Architectures** | 3 models built from scratch — no wrapper magic |
| 🎨 **Beautiful UI** | Dark ocean theme + floating multilingual letters |
| 🚀 **Kaggle Training** | 4 self-contained notebooks to train on free GPUs |

---

## 🏗️ Custom Model Architectures

### 1. 🎙️ MicroConformerASR (~25M parameters)
```
Waveform → Log-Mel Spectrogram (80 mels)
         → Conv Subsampling (4×)
         → 6× ConformerBlocks
              ├── FeedForward Module (Swish, half-step)
              ├── RelPos Multi-Head Self-Attention
              ├── Convolution Module (depthwise, GLU)
              └── FeedForward Module (Swish, half-step)
         → 6× Transformer Decoder Blocks (cross-attention)
         ├── CTC Head (30% weight)
         └── CE Head (70% weight, weight-tied LM head)
```
- **Training**: FLEURS (Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam) + CommonVoice English
- **Output**: `asr_model.pt`

### 2. 🌐 MultilingualSeq2Seq (~35M parameters)
```
Source Tokens → Shared Embedding (weight-tied)
              → 6× Encoder Blocks
                   ├── RoPE Multi-Head Self-Attention
                   └── GELU Feed-Forward Network
              → 6× Decoder Blocks
                   ├── RoPE Masked Self-Attention
                   ├── Cross-Attention (no RoPE on encoder K,V)
                   └── GELU Feed-Forward Network
              → Weight-Tied LM Head
              → Label Smoothing Loss (ε=0.1)
```
- **Training**: Samanantar (11 Indian pairs) + OPUS-100 (international)
- **Output**: `nmt_model.pt`

### 3. 🔊 FastSpeech2TTS (~20M parameters)
```
Phonemes → Embedding
         → 4× FFT Encoder Blocks (Conv1D FFN, kernel=9)
         → Variance Adaptor
              ├── Duration Predictor → Length Regulator
              ├── Pitch Predictor → Quantized Embedding
              └── Energy Predictor → Quantized Embedding
         → 4× FFT Decoder Blocks
         → Linear → Mel-Spectrogram
         → 5-layer PostNet (refinement)
         → Griffin-Lim Vocoder (training-free)
```
- **Training**: LJSpeech-1.1 + IndicTTS
- **Output**: `tts_model.pt`

---

## 🌍 Supported Languages

### 🇮🇳 Indian Languages (22)
| # | Language | Script | Flag |
|---|----------|--------|------|
| 1 | Hindi | Devanagari | 🇮🇳 |
| 2 | Bengali | Bengali | 🇧🇩 |
| 3 | Tamil | Tamil | 🇮🇳 |
| 4 | Telugu | Telugu | 🇮🇳 |
| 5 | Marathi | Devanagari | 🇮🇳 |
| 6 | Gujarati | Gujarati | 🇮🇳 |
| 7 | Kannada | Kannada | 🇮🇳 |
| 8 | Malayalam | Malayalam | 🇮🇳 |
| 9 | Punjabi | Gurmukhi | 🇮🇳 |
| 10 | Odia | Odia | 🇮🇳 |
| 11 | Assamese | Bengali | 🇮🇳 |
| 12 | Urdu | Arabic | 🇵🇰 |
| 13 | Sanskrit | Devanagari | 🇮🇳 |
| 14 | Maithili | Devanagari | 🇮🇳 |
| 15 | Sindhi | Arabic | 🇮🇳 |
| 16 | Nepali | Devanagari | 🇳🇵 |
| 17 | Konkani | Devanagari | 🇮🇳 |
| 18 | Manipuri | Meitei | 🇮🇳 |
| 19 | Bodo | Devanagari | 🇮🇳 |
| 20 | Dogri | Devanagari | 🇮🇳 |
| 21 | Kashmiri | Arabic | 🇮🇳 |
| 22 | Santali | Ol Chiki | 🇮🇳 |

### 🌏 International Languages (27)
Japanese 🇯🇵 · Chinese 🇨🇳 · German 🇩🇪 · French 🇫🇷 · Dutch 🇳🇱 · Spanish 🇪🇸 · Italian 🇮🇹 · Portuguese 🇵🇹 · Russian 🇷🇺 · Arabic 🇸🇦 · Korean 🇰🇷 · Turkish 🇹🇷 · Polish 🇵🇱 · Swedish 🇸🇪 · Norwegian 🇳🇴 · Danish 🇩🇰 · Finnish 🇫🇮 · Greek 🇬🇷 · Hebrew 🇮🇱 · Thai 🇹🇭 · Vietnamese 🇻🇳 · Indonesian 🇮🇩 · Malay 🇲🇾 · Filipino 🇵🇭 · Swahili 🇰🇪 · Ukrainian 🇺🇦 · Czech 🇨🇿 · Romanian 🇷🇴

---

## 🚀 Quick Start

### Option 1 — Windows (double-click)
```
run.bat
```

### Option 2 — Command Line
```bash
# Clone
git clone https://github.com/paulamartya25/Language-translator-.git
cd Language-translator-

# Install dependencies
pip install -r requirements.txt

# Launch
python app/gradio_app.py
```
Open **http://localhost:7860** in your browser.

---

## 🗺️ Project Structure

```
LANGUAGE_TRANSLATER/
│
├── app/
│   └── gradio_app.py           # 🎨 Main Gradio UI (dark ocean + floating letters)
│
├── models/
│   ├── architectures/
│   │   ├── micro_conformer_asr.py    # 🎙️ Custom ASR (~25M)
│   │   ├── multilingual_seq2seq.py   # 🌐 Custom NMT (~35M)
│   │   ├── fastspeech2_tts.py        # 🔊 Custom TTS (~20M)
│   │   └── demo_architectures.py     # Quick sanity-check demo
│   ├── asr/
│   │   ├── whisper_asr.py            # HuggingFace Whisper wrapper (fallback)
│   │   └── conformer_asr.py          # Wav2Vec2-Conformer wrapper (fallback)
│   ├── translation/
│   │   └── nllb_translator.py        # HuggingFace NLLB-200 wrapper (fallback)
│   └── tts/
│       └── gtts_engine.py            # gTTS wrapper (fallback)
│
├── inference/
│   └── full_inference.py        # End-to-end pipeline class
│
├── pipeline/
│   ├── audio_processor.py
│   ├── language_detector.py
│   └── translator_pipeline.py
│
├── training/
│   ├── finetune_whisper.py      # LoRA fine-tuning scripts
│   ├── finetune_nllb.py
│   └── TRAINING_GUIDE.md
│
├── kaggle/
│   ├── notebook_1_train_asr.ipynb   # Train MicroConformerASR on FLEURS
│   ├── notebook_2_train_nmt.ipynb   # Train Seq2Seq NMT on Samanantar/OPUS
│   ├── notebook_3_train_tts.ipynb   # Train FastSpeech2 on LJSpeech
│   └── notebook_4_live_app.ipynb    # Live Gradio app (share=True)
│
├── config/
│   ├── languages.py             # 49 language configs + FLORES+ codes
│   └── settings.py              # Model IDs, device, audio params
│
├── utils/
│   ├── audio_utils.py
│   └── text_utils.py
│
├── requirements.txt
├── run.bat                      # Windows launcher
├── run.sh                       # Linux/Mac launcher
└── README.md
```

---

## 🎓 Kaggle Training Pipeline

Train each model separately using free Kaggle GPU (T4 x2), save as Kaggle Datasets, then combine in Notebook 4:

```
📓 Notebook 1  →  asr_model.pt   (Kaggle Dataset: multilingual-asr-model)
📓 Notebook 2  →  nmt_model.pt   (Kaggle Dataset: multilingual-nmt-model)
📓 Notebook 3  →  tts_model.pt   (Kaggle Dataset: multilingual-tts-model)
                        ↓
📓 Notebook 4  →  Loads all 3 .pt files → Gradio app with share=True
                  → Public URL valid for 72 hours
```

### Steps
1. Open each notebook on [kaggle.com/code](https://kaggle.com/code) → New Notebook → Upload `.ipynb`
2. Enable GPU accelerator (T4 x2)
3. Run all cells
4. After training completes: **Output → Save as Dataset**
5. In Notebook 4: **Add Data** → Your Datasets → add all 3 model datasets
6. Run Notebook 4 → copy the `gradio.live` URL

---

## 🎨 UI Features

- **Dark ocean background** (`#060d1f`) with radial cyan glow
- **Floating multilingual letters** from 15+ scripts scrolling across the screen:
  - Devanagari (क ख ग) · Bengali (অ আ ই) · Tamil (அ ஆ இ) · Telugu (అ ఆ ఇ)
  - Gujarati (અ આ) · Kannada (ಅ ಆ) · Malayalam (അ ആ) · Gurmukhi (ਅ ਆ)
  - Hiragana (あいう) · Katakana (アイウ) · CJK (你好我) · Hangul (가나다)
  - Arabic (ا ب ت) · Greek (α β γ) · Cyrillic (А Б В) · Hebrew (א ב ג) · Thai (ก ข ค)
- **7 tabs**: Live Recording · Text Input · Multi-Language · All 49 · Indian · International · Architecture
- **Color-coded models**: Cyan=ASR · Violet=NMT · Amber=TTS

---

## 🔧 Dependencies

```
Python 3.10+
torch >= 2.2.0
torchaudio >= 2.2.0
transformers >= 4.40.0
huggingface-hub >= 1.16.0
gradio >= 6.0.0
gTTS >= 2.5.0
librosa >= 0.10.0
sentencepiece >= 0.2.0
```

---

## 📜 License

MIT License — free for personal and educational use.

---

*Built with ❤️ using PyTorch · Gradio · HuggingFace Transformers*
