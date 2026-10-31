"""
Custom Architecture Implementations — Built from Scratch in PyTorch
====================================================================
Three models following the confirmed project plan:

  Model 1 — MicroConformerASR      (~25M params)  Speech-to-Text
  Model 2 — MultilingualSeq2Seq    (~35M params)  Neural Machine Translation
  Model 3 — FastSpeech2TTS         (~20M params)  Text-to-Speech
"""

from .micro_conformer_asr import MicroConformerASR, MicroConformerConfig
from .multilingual_seq2seq import MultilingualSeq2Seq, Seq2SeqConfig
from .fastspeech2_tts import FastSpeech2TTS, TTSConfig

__all__ = [
    "MicroConformerASR", "MicroConformerConfig",
    "MultilingualSeq2Seq", "Seq2SeqConfig",
    "FastSpeech2TTS", "TTSConfig",
]
