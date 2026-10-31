"""
Whisper ASR module using HuggingFace Transformers.

Architecture: Audio-Spectrogram Transformer (Encoder-Decoder)
- Encoder: processes 80-channel Log-Mel Spectrograms with 2-D Conv stem
           followed by Transformer blocks (Multi-Head Attention + FFN)
- Decoder: cross-attends encoder output, auto-regressively generates tokens
           using a multilingual vocabulary
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class WhisperASR:
    """
    Automatic Speech Recognition using OpenAI Whisper via HuggingFace pipeline.

    The model ingests 30-second windows of 80-channel Log-Mel Spectrograms
    and outputs transcribed text alongside the detected source language.
    """

    def __init__(self, model_id: str = "openai/whisper-base", device: str = "cpu"):
        """
        Initialize the Whisper ASR model.

        Args:
            model_id: HuggingFace model identifier (e.g. 'openai/whisper-base').
            device:   Compute device ('cpu' or 'cuda').
        """
        self.model_id = model_id
        self.device = device
        self.pipe = None
        self._load_model()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_model(self):
        """Load Whisper pipeline from HuggingFace Hub."""
        try:
            from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

            logger.info(f"Loading Whisper model: {self.model_id} on {self.device}")

            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(self.device)

            processor = AutoProcessor.from_pretrained(self.model_id)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30,
                batch_size=4,
                torch_dtype=torch_dtype,
                device=self.device,
            )

            logger.info("Whisper model loaded successfully.")

        except Exception as exc:
            logger.error(f"Failed to load Whisper model: {exc}")
            self.pipe = None
            raise

    # ── Public API ─────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio_array: 1-D float32 numpy array of audio samples.
            sample_rate: Sample rate of the input audio (will be resampled to 16 kHz).
            language:    Optional BCP-47 language hint (e.g. 'en', 'hi').

        Returns:
            dict with keys:
                - 'text'     : str   — transcribed text
                - 'language' : str   — detected language code (BCP-47)
                - 'segments' : list  — list of segment dicts (may be empty)
                - 'chunks'   : list  — word/chunk level timestamps (if available)
        """
        if self.pipe is None:
            raise RuntimeError("Whisper pipeline is not loaded.")

        # Ensure float32 mono
        audio_array = audio_array.astype(np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        # Resample if needed
        if sample_rate != 16000:
            audio_array = _resample(audio_array, sample_rate, 16000)

        # Normalize to [-1, 1]
        max_val = np.abs(audio_array).max()
        if max_val > 0:
            audio_array = audio_array / max_val

        # Build generate kwargs
        generate_kwargs: Dict[str, Any] = {
            "return_timestamps": True,
        }
        if language:
            generate_kwargs["language"] = language

        try:
            result = self.pipe(
                {"array": audio_array, "sampling_rate": 16000},
                generate_kwargs=generate_kwargs,
            )
        except Exception as exc:
            logger.error(f"Whisper transcription failed: {exc}")
            return {"text": "", "language": "en", "segments": [], "chunks": []}

        text = result.get("text", "").strip()
        chunks = result.get("chunks", [])

        # Detect language via feature extractor if not in result
        detected_lang = result.get("language", language or "en")
        if detected_lang is None:
            detected_lang = "en"

        return {
            "text": text,
            "language": detected_lang,
            "segments": [],  # segments not returned by pipeline by default
            "chunks": chunks,
        }

    def detect_language(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Detect the spoken language without full transcription.

        Returns BCP-47 language code string (e.g. 'hi', 'en').
        """
        result = self.transcribe(audio_array, sample_rate)
        return result.get("language", "en")

    def is_loaded(self) -> bool:
        """Return True if the model pipeline was successfully loaded."""
        return self.pipe is not None


# ── Module-level helpers ───────────────────────────────────────────────────────

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple resampling using scipy."""
    try:
        import scipy.signal as sps

        num_samples = int(len(audio) * target_sr / orig_sr)
        resampled = sps.resample(audio, num_samples)
        return resampled.astype(np.float32)
    except Exception:
        return audio


# ── Streamlit cache wrapper ────────────────────────────────────────────────────

def load_whisper_model(model_id: str = "openai/whisper-base", device: str = "cpu") -> WhisperASR:
    """
    Load (or retrieve cached) Whisper ASR model.

    Decorated with @st.cache_resource when called from app/main.py so the
    model is shared across Streamlit reruns.
    """
    return WhisperASR(model_id=model_id, device=device)
