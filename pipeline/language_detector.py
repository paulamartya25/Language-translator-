"""
Language detection module.

Uses Whisper's built-in language identification capability to detect
the spoken language from raw audio, then maps the result to NLLB FLORES+
codes for use in the translation pipeline.
"""

import logging
from typing import Optional

import numpy as np

from config.languages import whisper_to_nllb, get_language_name_from_whisper

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Detect the language of spoken audio using Whisper's language ID.

    Whisper uses the first 30 seconds of audio and its multilingual encoder
    to produce a probability distribution over ~100 languages. The argmax
    gives the detected language code.
    """

    def __init__(self, whisper_model=None):
        """
        Initialize the detector.

        Args:
            whisper_model: An already-loaded WhisperASR instance (optional).
                           If provided, detection reuses the loaded model.
        """
        self._asr = whisper_model

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect_language(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """
        Detect the spoken language and return a Whisper BCP-47 code.

        Args:
            audio_array: 1-D float32 numpy audio array.
            sample_rate: Sample rate of the audio.

        Returns:
            BCP-47 language code string (e.g. 'hi', 'en', 'ta').
        """
        if self._asr is not None and self._asr.is_loaded():
            result = self._asr.transcribe(audio_array, sample_rate)
            return result.get("language", "en")

        # Fallback: try direct HuggingFace pipeline
        return self._detect_via_pipeline(audio_array, sample_rate)

    def detect_language_nllb(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """
        Detect language and return NLLB FLORES+ code.

        Args:
            audio_array: 1-D float32 numpy audio array.
            sample_rate: Sample rate.

        Returns:
            NLLB FLORES+ code (e.g. 'hin_Deva', 'eng_Latn').
        """
        whisper_code = self.detect_language(audio_array, sample_rate)
        return whisper_to_nllb(whisper_code)

    def detect_language_name(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """
        Detect language and return a human-readable name.

        Returns:
            Language name string (e.g. 'Hindi', 'English').
        """
        whisper_code = self.detect_language(audio_array, sample_rate)
        return get_language_name_from_whisper(whisper_code)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _detect_via_pipeline(
        self, audio_array: np.ndarray, sample_rate: int
    ) -> str:
        """Standalone language detection without a pre-loaded model."""
        try:
            from transformers import pipeline

            pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                generate_kwargs={"task": "transcribe"},
            )
            result = pipe(
                {"array": audio_array.astype(np.float32), "sampling_rate": sample_rate}
            )
            return result.get("language", "en")
        except Exception as exc:
            logger.warning(f"Standalone language detection failed: {exc}")
            return "en"
