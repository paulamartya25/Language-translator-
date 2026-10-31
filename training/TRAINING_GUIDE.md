# 🎓 Fine-tuning Guide — Multilingual AI Translator

This folder contains complete fine-tuning scripts for both AI models in the pipeline.

---

## Why Fine-tune?

| Model | Pre-trained on | Why Fine-tune? |
|---|---|---|
| **Whisper** | 680K hours of internet audio (multilingual) | Improve accuracy on your specific accent, dialect, domain (medical, legal, etc.) |
| **NLLB-200** | CCNet, CommonCrawl, Wikipedia (200 langs) | Improve translation quality for domain-specific text (e.g. legal docs, medical records) |

> Pre-trained models are already very good. Fine-tune only if you have:
> - Domain-specific vocabulary (medical, legal, technical)
> - Regional dialect audio not well represented in training data
> - Your own parallel text corpus

---

## 1. Fine-tune Whisper ASR

### Requirements
```bash
pip install peft evaluate jiwer datasets librosa soundfile
```

### Using Google FLEURS (Indian languages — recommended)
```bash
python training/finetune_whisper.py \
    --lang hi \
    --model openai/whisper-small \
    --dataset fleurs \
    --epochs 5 \
    --batch_size 8
```

### Using Mozilla Common Voice
```bash
python training/finetune_whisper.py \
    --lang ta \
    --model openai/whisper-base \
    --dataset common_voice \
    --epochs 3
```

### Using your own data
Prepare a CSV file:
```csv
audio_path,transcript,language
/data/audio/clip001.wav,नमस्ते कैसे हो आप,hi
/data/audio/clip002.wav,मुझे पानी चाहिए,hi
```

Then run:
```bash
python training/finetune_whisper.py \
    --lang hi \
    --dataset custom \
    --data_path data/my_hindi_data.csv \
    --epochs 10
```

### After fine-tuning
Update `config/settings.py`:
```python
ASR_MODEL = "checkpoints/whisper-finetuned/whisper-hi/final"
```

---

## 2. Fine-tune NLLB-200 Translation

### Requirements
```bash
pip install peft evaluate sacrebleu datasets
```

### Using Samanantar (49M Indian parallel sentences — best for Indian langs)
```bash
python training/finetune_nllb.py \
    --src_lang eng_Latn \
    --tgt_lang hin_Deva \
    --dataset samanantar \
    --epochs 3
```

### All Indian languages (batch script)
```bash
for lang in hin_Deva ben_Beng tam_Taml tel_Telu mar_Deva; do
    python training/finetune_nllb.py \
        --src_lang eng_Latn \
        --tgt_lang $lang \
        --dataset samanantar \
        --max_train_samples 100000
done
```

### Using your own parallel corpus
Prepare JSON:
```json
[
  {"source": "Hello, how are you?", "target": "नमस्ते, आप कैसे हैं?"},
  {"source": "Good morning", "target": "सुप्रभात"},
  ...
]
```

```bash
python training/finetune_nllb.py \
    --src_lang eng_Latn \
    --tgt_lang hin_Deva \
    --dataset custom \
    --data_path data/en_hi_pairs.json
```

### After fine-tuning
Update `config/settings.py`:
```python
TRANSLATION_MODEL = "checkpoints/nllb-finetuned/nllb-eng_Latn-hin_Deva/final"
```

---

## Memory Requirements

| Model | Full Fine-tune | LoRA Fine-tune |
|---|---|---|
| Whisper Tiny | 2 GB GPU | 500 MB GPU |
| Whisper Base | 3 GB GPU | 800 MB GPU |
| Whisper Small | 6 GB GPU | 1.5 GB GPU |
| Whisper Large-v3 | 24 GB GPU | 6 GB GPU |
| NLLB-200 600M | 8 GB GPU | 2 GB GPU |
| NLLB-200 1.3B | 16 GB GPU | 4 GB GPU |

> **LoRA** (Low-Rank Adaptation) is enabled by default. It adds small rank-r matrices
> to attention layers, reducing trainable parameters by ~97% while preserving quality.

---

## Monitor Training

```bash
tensorboard --logdir checkpoints/
```

Open http://localhost:6006 to see:
- Training/validation loss curves
- WER (Word Error Rate) for Whisper
- BLEU / chrF scores for NLLB

---

## Datasets

| Dataset | Size | Languages | Use for |
|---|---|---|---|
| [Google FLEURS](https://huggingface.co/datasets/google/fleurs) | 12h/lang | 102 langs | Whisper ASR |
| [Mozilla Common Voice](https://commonvoice.mozilla.org/) | Varies | 100+ langs | Whisper ASR |
| [AI4Bharat IndicVoices](https://huggingface.co/datasets/ai4bharat/indicvoices_r) | 7600h | 22 Indian | Whisper ASR |
| [Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) | 49M pairs | 11 Indian | NLLB Translation |
| [OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) | 1M/pair | 100 langs | NLLB Translation |
