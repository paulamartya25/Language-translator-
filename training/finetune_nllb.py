"""
🔁 Fine-tuning NLLB-200 Translation Model
==========================================
Fine-tunes Meta's NLLB-200 (Seq2Seq Encoder-Decoder) on custom
parallel translation corpora for Indian language pairs.

Architecture: Seq2Seq Encoder-Decoder Transformer
- Encoder: 12-layer bidirectional Transformer over source tokens
- Decoder: 12-layer cross-attentive Transformer generating target tokens
- Fine-tuning: LoRA on Q/V projections (saves ~90% GPU memory)

Datasets supported:
  - Custom CSV/JSON (source_text, target_text, source_lang, target_lang)
  - HuggingFace: ai4bharat/samanantar (largest Indian parallel corpus)
  - HuggingFace: Helsinki-NLP/opus-100

Usage:
    python training/finetune_nllb.py \\
        --src_lang eng_Latn \\
        --tgt_lang hin_Deva \\
        --dataset samanantar \\
        --epochs 3

    # With custom data:
    python training/finetune_nllb.py \\
        --src_lang eng_Latn \\
        --tgt_lang tam_Taml \\
        --dataset custom \\
        --data_path data/en_ta_pairs.json
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ── NLLB FLORES+ codes for Indian languages ────────────────────────────────────
NLLB_INDIAN_LANGS = {
    "hin_Deva": "Hindi",
    "ben_Beng": "Bengali",
    "tel_Telu": "Telugu",
    "tam_Taml": "Tamil",
    "mar_Deva": "Marathi",
    "urd_Arab": "Urdu",
    "guj_Gujr": "Gujarati",
    "kan_Knda": "Kannada",
    "ory_Orya": "Odia",
    "mal_Mlym": "Malayalam",
    "pan_Guru": "Punjabi",
    "asm_Beng": "Assamese",
    "npi_Deva": "Nepali",
    "san_Deva": "Sanskrit",
    "mai_Deva": "Maithili",
}

# Samanantar language code map
SAMANANTAR_LANG_MAP = {
    "hin_Deva": "hi",
    "ben_Beng": "bn",
    "tel_Telu": "te",
    "tam_Taml": "ta",
    "mar_Deva": "mr",
    "urd_Arab": "ur",
    "guj_Gujr": "gu",
    "kan_Knda": "kn",
    "ory_Orya": "or",
    "mal_Mlym": "ml",
    "pan_Guru": "pa",
    "asm_Beng": "as",
}


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NLLBFinetuneConfig:
    """Hyperparameters for NLLB-200 fine-tuning."""
    base_model: str = "facebook/nllb-200-distilled-600M"
    source_lang: str = "eng_Latn"
    target_lang: str = "hin_Deva"
    dataset_name: str = "samanantar"
    custom_data_path: Optional[str] = None

    # Training
    output_dir: str = "checkpoints/nllb-finetuned"
    num_epochs: int = 3
    batch_size: int = 16
    gradient_accumulation: int = 4
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.06
    max_steps: int = 10000
    eval_steps: int = 500
    save_steps: int = 500
    max_source_length: int = 256
    max_target_length: int = 256
    fp16: bool = True

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Misc
    push_to_hub: bool = False
    hub_model_id: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_train_samples: Optional[int] = None    # limit for quick testing
    max_eval_samples: int = 1000


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_samanantar(tgt_lang_nllb: str, split: str = "train"):
    """
    Load Samanantar — largest publicly available Indian parallel corpus.
    Contains ~49M sentence pairs across 11 Indian languages.
    """
    from datasets import load_dataset

    lang = SAMANANTAR_LANG_MAP.get(tgt_lang_nllb)
    if not lang:
        raise ValueError(f"Samanantar does not have data for {tgt_lang_nllb}")

    print(f"Loading Samanantar: English → {lang}...")
    ds = load_dataset("ai4bharat/samanantar", lang, split=split, trust_remote_code=True)
    print(f"  Loaded {len(ds)} parallel sentences")
    return ds


def load_opus100(src_lang_nllb: str, tgt_lang_nllb: str, split: str = "train"):
    """Load OPUS-100 multilingual parallel corpus."""
    from datasets import load_dataset

    # Map NLLB → ISO 639-1
    nllb_to_iso = {
        "eng_Latn": "en", "hin_Deva": "hi", "tam_Taml": "ta",
        "tel_Telu": "te", "ben_Beng": "bn", "mar_Deva": "mr",
        "fra_Latn": "fr", "deu_Latn": "de", "jpn_Jpan": "ja",
        "zho_Hans": "zh",
    }
    src = nllb_to_iso.get(src_lang_nllb, "en")
    tgt = nllb_to_iso.get(tgt_lang_nllb, "hi")
    lang_pair = f"{src}-{tgt}"

    print(f"Loading OPUS-100: {lang_pair}...")
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", lang_pair, split=split)
    except Exception:
        lang_pair = f"{tgt}-{src}"
        ds = load_dataset("Helsinki-NLP/opus-100", lang_pair, split=split)

    print(f"  Loaded {len(ds)} parallel sentences")
    return ds


def load_custom_parallel(data_path: str):
    """
    Load custom parallel corpus.

    JSON format:
      [{"source": "Hello world", "target": "नमस्ते दुनिया"}, ...]

    CSV format:
      source,target
      Hello world,नमस्ते दुनिया
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
                data.append({"source": row["source"], "target": row["target"]})
    else:
        raise ValueError(f"Unsupported: {path.suffix}")

    print(f"Loaded {len(data)} custom parallel pairs")
    return Dataset.from_list(data)


# ══════════════════════════════════════════════════════════════════════════════
# Data Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def build_preprocess_fn(tokenizer, config: NLLBFinetuneConfig):
    """Build tokenization function for seq2seq training."""

    def preprocess(batch):
        # Get source/target text based on dataset format
        if "src" in batch:
            # Samanantar format
            sources = batch["src"]
            targets = batch["tgt"]
        elif "translation" in batch:
            # OPUS format
            src_iso = config.source_lang.split("_")[0].lower()[:2]
            tgt_iso = config.target_lang.split("_")[0].lower()[:2]
            sources = [ex.get(src_iso, "") for ex in batch["translation"]]
            targets = [ex.get(tgt_iso, "") for ex in batch["translation"]]
        else:
            # Custom format
            sources = batch["source"]
            targets = batch["target"]

        # Tokenize source
        tokenizer.src_lang = config.source_lang
        model_inputs = tokenizer(
            sources,
            max_length=config.max_source_length,
            truncation=True,
            padding="max_length",
        )

        # Tokenize target
        tokenizer.src_lang = config.target_lang
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=config.max_target_length,
                truncation=True,
                padding="max_length",
            )

        # Replace pad token in labels with -100
        label_ids = labels["input_ids"]
        label_ids = [
            [(t if t != tokenizer.pad_token_id else -100) for t in lab]
            for lab in label_ids
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    return preprocess


# ══════════════════════════════════════════════════════════════════════════════
# LoRA for NLLB
# ══════════════════════════════════════════════════════════════════════════════

def apply_lora_nllb(model, config: NLLBFinetuneConfig):
    """
    Apply LoRA to NLLB-200's encoder and decoder attention layers.

    NLLB attention modules: q_proj, k_proj, v_proj, out_proj
    We target q_proj and v_proj for minimal trainable params.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        return model

    except ImportError:
        os.system("pip install peft -q")
        return apply_lora_nllb(model, config)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def build_metrics(tokenizer):
    """Compute BLEU and chrF scores."""
    import evaluate

    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = [[t if t != -100 else tokenizer.pad_token_id for t in lab] for lab in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]

        bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        chrf_score = chrf.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            "bleu": bleu_score["score"],
            "chrf": chrf_score["score"],
        }

    return compute_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main Fine-tuning Function
# ══════════════════════════════════════════════════════════════════════════════

def finetune_nllb(config: NLLBFinetuneConfig):
    """Run the full NLLB-200 fine-tuning pipeline."""

    from transformers import (
        AutoModelForSeq2SeqLM,
        NllbTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
    )

    print("=" * 60)
    print("  NLLB-200 Seq2Seq Fine-tuning")
    print("=" * 60)
    print(f"  Base model    : {config.base_model}")
    print(f"  Source lang   : {config.source_lang}")
    print(f"  Target lang   : {config.target_lang} ({NLLB_INDIAN_LANGS.get(config.target_lang, '?')})")
    print(f"  Dataset       : {config.dataset_name}")
    print(f"  Device        : {config.device}")
    print(f"  LoRA          : {'Yes' if config.use_lora else 'No'}")
    print()

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = NllbTokenizer.from_pretrained(
        config.base_model,
        src_lang=config.source_lang,
        tgt_lang=config.target_lang,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {config.base_model}...")
    torch_dtype = torch.float16 if config.fp16 and config.device == "cuda" else torch.float32

    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    if config.use_lora:
        model = apply_lora_nllb(model, config)

    model.to(config.device)

    # ── Load dataset ──────────────────────────────────────────────────────────
    if config.dataset_name == "samanantar":
        train_ds = load_samanantar(config.target_lang, "train")
        eval_ds = load_samanantar(config.target_lang, "validation")
    elif config.dataset_name == "opus100":
        train_ds = load_opus100(config.source_lang, config.target_lang, "train")
        eval_ds = load_opus100(config.source_lang, config.target_lang, "validation")
    elif config.dataset_name == "custom":
        full_ds = load_custom_parallel(config.custom_data_path)
        split = full_ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")

    # Optionally limit samples for quick testing
    if config.max_train_samples:
        train_ds = train_ds.select(range(min(config.max_train_samples, len(train_ds))))
    if config.max_eval_samples:
        eval_ds = eval_ds.select(range(min(config.max_eval_samples, len(eval_ds))))

    # ── Preprocess ────────────────────────────────────────────────────────────
    print("Tokenizing dataset...")
    preprocess_fn = build_preprocess_fn(tokenizer, config)

    train_ds = train_ds.map(
        preprocess_fn, batched=True, batch_size=256,
        remove_columns=train_ds.column_names, num_proc=2,
    )
    eval_ds = eval_ds.map(
        preprocess_fn, batched=True, batch_size=256,
        remove_columns=eval_ds.column_names, num_proc=2,
    )

    print(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── Collator ──────────────────────────────────────────────────────────────
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # ── Training args ─────────────────────────────────────────────────────────
    output_dir = Path(config.output_dir) / f"nllb-{config.source_lang}-{config.target_lang}"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        max_steps=config.max_steps,
        num_train_epochs=config.num_epochs,
        fp16=config.fp16 and config.device == "cuda",
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_steps=50,
        report_to=["tensorboard"],
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id if config.push_to_hub else None,
        dataloader_num_workers=2,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=build_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting training → {output_dir}")
    print(f"  Monitor: tensorboard --logdir {output_dir}/runs\n")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    if config.use_lora:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(output_dir / "final_merged"))
        print(f"Merged model: {output_dir}/final_merged")

    # ── Final eval ────────────────────────────────────────────────────────────
    results = trainer.evaluate()
    print(f"\nFinal BLEU  : {results.get('eval_bleu', 0):.2f}")
    print(f"Final chrF  : {results.get('eval_chrf', 0):.2f}")
    print(f"\nUse your model in settings.py:")
    print(f"  TRANSLATION_MODEL = '{output_dir}/final'")

    return str(output_dir / "final")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune NLLB-200 translation model")
    parser.add_argument("--src_lang", default="eng_Latn", help="Source NLLB FLORES+ code")
    parser.add_argument("--tgt_lang", default="hin_Deva",
                        choices=list(NLLB_INDIAN_LANGS.keys()),
                        help="Target NLLB FLORES+ code")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--dataset", default="samanantar",
                        choices=["samanantar", "opus100", "custom"])
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--output_dir", default="checkpoints/nllb-finetuned")
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = NLLBFinetuneConfig(
        base_model=args.model,
        source_lang=args.src_lang,
        target_lang=args.tgt_lang,
        dataset_name=args.dataset,
        custom_data_path=args.data_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        max_train_samples=args.max_train_samples,
        output_dir=args.output_dir,
        use_lora=not args.no_lora,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )
    finetune_nllb(cfg)
