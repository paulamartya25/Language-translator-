"""
Architecture Demo Script
========================
Runs forward passes on all 3 custom models and prints
tensor shapes + parameter counts for verification.

Usage:
    cd c:\\Users\\amart\\Desktop\\LANGUAGE_TRANSLATER
    python models/architectures/demo_architectures.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from models.architectures.micro_conformer_asr import MicroConformerASR, MicroConformerConfig, demo as demo1
from models.architectures.multilingual_seq2seq import MultilingualSeq2Seq, Seq2SeqConfig, demo as demo2
from models.architectures.fastspeech2_tts import FastSpeech2TTS, TTSConfig, demo as demo3


def summary_table():
    print("\n" + "=" * 60)
    print("  PARAMETER COUNT SUMMARY")
    print("=" * 60)

    # Small configs for quick CPU demo
    asr = MicroConformerASR(MicroConformerConfig(
        d_model=256, n_heads=4, n_enc_layers=6, n_dec_layers=6,
        d_ff=1024, vocab_size=5000, conv_kernel_size=31,
    ))
    nmt = MultilingualSeq2Seq(Seq2SeqConfig(
        vocab_size=32000, d_model=512, n_heads=8,
        n_enc_layers=6, n_dec_layers=6, d_ff=2048,
    ))
    tts = FastSpeech2TTS(TTSConfig(
        vocab_size=200, d_model=256, n_heads=4,
        n_fft_blocks=4, d_ff=1024,
    ))

    rows = [
        ("Model 1 — Micro-Conformer ASR", asr.count_parameters(), "~25M"),
        ("Model 2 — Multilingual Seq2Seq", nmt.count_parameters(), "~35M"),
        ("Model 3 — FastSpeech2 TTS",     tts.count_parameters(), "~20M"),
    ]

    total = sum(r[1] for r in rows)
    print(f"\n{'Model':<34} {'Actual':>12} {'Target':>8}")
    print("-" * 56)
    for name, count, target in rows:
        print(f"{name:<34} {count:>12,} {target:>8}")
    print("-" * 56)
    print(f"{'TOTAL':<34} {total:>12,}")
    print()


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  MULTILINGUAL AI TRANSLATOR — ARCHITECTURE DEMO")
    print("  3 Custom PyTorch Models Built From Scratch")
    print("█" * 60)

    demo1()   # Micro-Conformer ASR
    demo2()   # Multilingual Seq2Seq
    demo3()   # FastSpeech2 TTS
    summary_table()

    print("All 3 architectures verified successfully!")
