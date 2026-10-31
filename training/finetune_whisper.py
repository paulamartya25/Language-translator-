"""
🎓 Fine-tuning Whisper for Indian Language ASR
================================================
Fine-tunes OpenAI Whisper on custom Indian-language speech data
using HuggingFace Trainer + LoRA PEFT for memory efficiency.

Architecture: Audio-Spectrogram Transformer (AST)
- Encoder: Log-Mel Spectrogram → 2D Conv → Transformer blocks
- Decoder: Cross-attends encoder, auto-regressively generates tokens
- Fine-tuning: We only train the decoder + last 2 encoder layers (LoRA)

Datasets supported:
  - Custom CSV/JSON  (audio_path, transcript, language)
  - HuggingFace: mozilla-foundation/common_voice_13_0
  - HuggingFace: google/fleurs (Indian languages)
  - HuggingFace: ai4bharat/indicvoices_r

Usage:
    python training/finetune_whisper.py --lang hi --epochs 5 --batch_size 8
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ── Config ─────────────────────────────────────────────────────────────────────

SUPPORTED_INDIAN_LANGS = {
    "hi": "Hindi",
    "bn": "Bengali",
    "te": "Telugu",
    "ta": "Tamil",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
    "ne": "Nepali",
    "sa": "Sanskrit",
    "sd": "Sindhi",
}

FLEURS_LANG_MAP = {
    "hi": "hi_in",
    "bn": "bn_in",
    "te": "te_in",
    "ta": "ta_in",
    "mr": "mr_in",
    "gu": "gu_in",
    "kn": "kn_in",
    "ml": "ml_in",
    "pa": "pa_in",
    "or": "or_in",
    "as": "as_in",
    "ur": "ur_pk",
    "ne": "ne_np",
}

# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WhisperFinetuneConfig:
    """All hyperparameters for Whisper fine-tuning."""
    base_model: str = "openai/whisper-small"   # tiny/base/small/medium/large-v3
    language: str = "hi"                        # BCP-47 code
    task: str = "transcribe"
    dataset_name: str = "google/fleurs"         # or "common_voice" or "custom"
    custom_data_path: Optional[str] = None      # CSV or JSON if dataset_name="custom"

    # Training
    output_dir: str = "checkpoints/whisper-finetuned"
    num_epochs: int = 5
    batch_size: int = 8
    gradient_accumulation: int = 2
    learning_rate: float = 1e-5
    warmup_steps: int = 200
    max_steps: int = 5000
    eval_steps: int = 500
    save_steps: int = 500
    fp16: bool = True                           # set False for CPU

    # LoRA (memory-efficient fine-tuning)
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Audio
    max_audio_length: float = 30.0             # seconds
    sample_rate: int = 16000

    # Misc
    push_to_hub: bool = False
    hub_model_id: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_fleurs(lang_code: str, split: str = "train"):
    """Load Google FLEURS dataset for an Indian language."""
    from datasets import load_dataset

    fleurs_lang = FLEURS_LANG_MAP.get(lang_code, f"{lang_code}_in")
    print(f"Loading FLEURS: {fleurs_lang} ({split} split)...")

    ds = load_dataset(
        "google/fleurs",
        fleurs_lang,
        split=split,
        trust_remote_code=True,
    )
    print(f"  Loaded {len(ds)} samples from FLEURS/{fleurs_lang}")
    return ds


def load_common_voice(lang_code: str, split: str = "train"):
    """Load Mozilla Common Voice for an Indian language."""
    from datasets import load_dataset

    print(f"Loading Common Voice 13: {lang_code} ({split} split)...")
    ds = load_dataset(
        "mozilla-foundation/common_voice_13_0",
        lang_code,
        split=split,
        trust_remote_code=True,
        use_auth_token=True,  # requires HF token for CV
    )
    print(f"  Loaded {len(ds)} samples from Common Voice/{lang_code}")
    return ds


def load_custom_dataset(data_path: str):
    """
    Load a custom dataset from CSV or JSON.

    Expected CSV format:
        audio_path,transcript,language
        /path/to/audio.wav,नमस्ते कैसे हो,hi

    Expected JSON format:
        [{"audio_path": "...", "transcript": "...", "language": "hi"}, ...]
    """
    from datasets import Dataset

    path = Path(data_path)
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    elif path.suffix == ".csv":
        import csv
        data = []
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                data.append(row)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Use .csv or .json")

    print(f"Loaded {len(data)} custom samples from {data_path}")
    return Dataset.from_list(data)


# ══════════════════════════════════════════════════════════════════════════════
# Data Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def build_preprocessor(processor, config: WhisperFinetuneConfig):
    """Build dataset-agnostic preprocessing function."""

    def preprocess_fleurs(batch):
        audio = batch["audio"]
        array = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

        # Resample if needed
        if sr != config.sample_rate:
            import librosa
            array = librosa.resample(array, orig_sr=sr, target_sr=config.sample_rate)

        # Compute log-Mel spectrogram (80 bins) — this is the AST input
        inputs = processor.feature_extractor(
            array,
            sampling_rate=config.sample_rate,
            return_tensors="pt",
        )
        batch["input_features"] = inputs.input_features[0]

        # Tokenize transcript
        with processor.as_target_processor():
            batch["labels"] = processor.tokenizer(
                batch["transcription"], return_tensors="pt"
            ).input_ids[0]

        return batch

    def preprocess_common_voice(batch):
        audio = batch["audio"]
        array = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

        if sr != config.sample_rate:
            import librosa
            array = librosa.resample(array, orig_sr=sr, target_sr=config.sample_rate)

        inputs = processor.feature_extractor(
            array,
            sampling_rate=config.sample_rate,
            return_tensors="pt",
        )
        batch["input_features"] = inputs.input_features[0]

        with processor.as_target_processor():
            batch["labels"] = processor.tokenizer(
                batch["sentence"], return_tensors="pt"
            ).input_ids[0]

        return batch

    def preprocess_custom(batch):
        import soundfile as sf

        array, sr = sf.read(batch["audio_path"], dtype="float32")
        if array.ndim > 1:
            array = array.mean(axis=1)
        if sr != config.sample_rate:
            import librosa
            array = librosa.resample(array, orig_sr=sr, target_sr=config.sample_rate)

        inputs = processor.feature_extractor(
            array,
            sampling_rate=config.sample_rate,
            return_tensors="pt",
        )
        batch["input_features"] = inputs.input_features[0]

        with processor.as_target_processor():
            batch["labels"] = processor.tokenizer(
                batch["transcript"], return_tensors="pt"
            ).input_ids[0]

        return batch

    return {
        "fleurs": preprocess_fleurs,
        "common_voice": preprocess_common_voice,
        "custom": preprocess_custom,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Data Collator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WhisperDataCollator:
    """
    Custom data collator for Whisper fine-tuning.

    Pads input_features (spectrograms) and labels (token ids) to equal lengths.
    Replaces padding tokens in labels with -100 (ignored by cross-entropy loss).
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id with -100 so loss ignores them
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove decoder_start_token if already prepended
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def build_compute_metrics(processor):
    """Build WER (Word Error Rate) metric computation."""
    import evaluate

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

        print(f"\nSample predictions:")
        for p, l in zip(pred_str[:3], label_str[:3]):
            print(f"  PRED : {p}")
            print(f"  TRUE : {l}")
            print()

        return {"wer": wer, "cer": cer}

    return compute_metrics


# ══════════════════════════════════════════════════════════════════════════════
# LoRA Setup (memory-efficient fine-tuning)
# ══════════════════════════════════════════════════════════════════════════════

def apply_lora(model, config: WhisperFinetuneConfig):
    """
    Apply LoRA (Low-Rank Adaptation) to Whisper's attention layers.

    Instead of updating all 244M parameters, LoRA adds small rank-r
    matrices to Q and V projections:
        W' = W + B·A   (where B: d×r, A: r×d, r << d)

    This reduces trainable params from ~244M → ~3-5M while preserving
    most of the pre-trained knowledge.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj"],   # Whisper attention Q and V
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print(
            f"LoRA applied: r={config.lora_r}, alpha={config.lora_alpha}, "
            f"target=q_proj,v_proj"
        )
        return model

    except ImportError:
        print("PEFT not installed. Installing...")
        os.system("pip install peft -q")
        return apply_lora(model, config)


# ══════════════════════════════════════════════════════════════════════════════
# Main Training Function
# ══════════════════════════════════════════════════════════════════════════════

def finetune_whisper(config: WhisperFinetuneConfig):
    """Run the full Whisper fine-tuning pipeline."""

    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        EarlyStoppingCallback,
    )

    print("=" * 60)
    print("  Whisper ASR Fine-tuning")
    print("=" * 60)
    print(f"  Base model : {config.base_model}")
    print(f"  Language   : {config.language} ({SUPPORTED_INDIAN_LANGS.get(config.language, '?')})")
    print(f"  Dataset    : {config.dataset_name}")
    print(f"  Device     : {config.device}")
    print(f"  LoRA       : {'Yes' if config.use_lora else 'No'}")
    print(f"  Epochs     : {config.num_epochs}")
    print()

    # ── Load processor ────────────────────────────────────────────────────────
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        config.base_model,
        language=config.language,
        task=config.task,
    )
    processor.tokenizer.set_prefix_tokens(
        language=config.language, task=config.task
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {config.base_model}...")
    torch_dtype = torch.float16 if config.fp16 and config.device == "cuda" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.base_model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.generation_config.language = config.language
    model.generation_config.task = config.task
    model.generation_config.forced_decoder_ids = None

    # Apply LoRA for memory-efficient fine-tuning
    if config.use_lora:
        model = apply_lora(model, config)

    model.to(config.device)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset: {config.dataset_name}...")

    if config.dataset_name == "fleurs":
        train_ds = load_fleurs(config.language, "train")
        eval_ds = load_fleurs(config.language, "validation")
        preprocess_key = "fleurs"
    elif config.dataset_name == "common_voice":
        train_ds = load_common_voice(config.language, "train")
        eval_ds = load_common_voice(config.language, "validation")
        preprocess_key = "common_voice"
    elif config.dataset_name == "custom":
        if not config.custom_data_path:
            raise ValueError("custom_data_path required when dataset_name='custom'")
        full_ds = load_custom_dataset(config.custom_data_path)
        split = full_ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]
        preprocess_key = "custom"
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    print("Preprocessing dataset (computing spectrograms)...")
    preprocessors = build_preprocessor(processor, config)
    fn = preprocessors[preprocess_key]

    train_ds = train_ds.map(fn, remove_columns=train_ds.column_names, num_proc=2)
    eval_ds = eval_ds.map(fn, remove_columns=eval_ds.column_names, num_proc=2)

    print(f"  Train samples: {len(train_ds)}")
    print(f"  Eval  samples: {len(eval_ds)}")

    # ── Data collator ─────────────────────────────────────────────────────────
    collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── Training arguments ────────────────────────────────────────────────────
    output_dir = Path(config.output_dir) / f"whisper-{config.language}"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        num_train_epochs=config.num_epochs,
        fp16=config.fp16 and config.device == "cuda",
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=25,
        report_to=["tensorboard"],
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id if config.push_to_hub else None,
        predict_with_generate=True,
        generation_max_length=225,
        dataloader_num_workers=2,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor.feature_extractor,
        data_collator=collator,
        compute_metrics=build_compute_metrics(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    print(f"  Output: {output_dir}")
    print(f"  Logs  : tensorboard --logdir {output_dir}/runs\n")

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\nSaving fine-tuned model...")
    model.save_pretrained(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))

    if config.use_lora:
        # Merge LoRA weights back into base model
        merged = model.merge_and_unload()
        merged.save_pretrained(str(output_dir / "final_merged"))
        print(f"LoRA-merged model saved to: {output_dir}/final_merged")

    print(f"\nFine-tuning complete! Model at: {output_dir}/final")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nRunning final evaluation...")
    results = trainer.evaluate()
    print(f"  WER : {results.get('eval_wer', 'N/A'):.2f}%")
    print(f"  CER : {results.get('eval_cer', 'N/A'):.2f}%")

    return str(output_dir / "final")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper for Indian language ASR"
    )
    parser.add_argument("--lang", default="hi", choices=list(SUPPORTED_INDIAN_LANGS.keys()),
                        help="Language code (e.g. hi, ta, bn)")
    parser.add_argument("--model", default="openai/whisper-small",
                        help="Base Whisper model ID")
    parser.add_argument("--dataset", default="fleurs",
                        choices=["fleurs", "common_voice", "custom"],
                        help="Dataset to use")
    parser.add_argument("--data_path", default=None,
                        help="Path to custom dataset CSV/JSON (if --dataset=custom)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--output_dir", default="checkpoints/whisper-finetuned")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = WhisperFinetuneConfig(
        base_model=args.model,
        language=args.lang,
        dataset_name=args.dataset,
        custom_data_path=args.data_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        use_lora=not args.no_lora,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    model_path = finetune_whisper(cfg)
    print(f"\nUse your fine-tuned model in settings.py:")
    print(f"  ASR_MODEL = '{model_path}'")
