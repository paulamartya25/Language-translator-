"""
Conformer ASR module — fallback for noisy audio.

Architecture: wav2vec2-Conformer
- Convolutional feature encoder (extracts local features from raw waveform)
- Conformer blocks: alternating Multi-Head Self-Attention + Convolution modules
  (point-wise + depth-wise conv) to capture both global and local dependencies
- CTC head for sequence decoding

Used when Whisper's no-speech probability is high or transcription confidence
is low (noisy recording conditions).
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ConformerASR:
    """
    Fallback Automatic Speech Recognition using wav2vec2-Conformer.

    This model is particularly robust to noisy audio due to its combined
    local (convolution) and global (attention) feature extraction.
    """

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-conformer-rel-pos-large",
        device: str = "cpu",
    ):
        """
        Initialize the Conformer ASR model.

        Args:
            model_id: HuggingFace model identifier.
            device:   Compute device ('cpu' or 'cuda').
        """
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False
        self._load_model()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_model(self):
        """Load model and processor from HuggingFace Hub."""
        try:
            from transformers import (
                Wav2Vec2ConformerForCTC,
                AutoProcessor,
            )

            logger.info(f"Loading Conformer model: {self.model_id} on {self.device}")

            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = Wav2Vec2ConformerForCTC.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

            self._loaded = True
            logger.info("Conformer model loaded successfully.")

        except Exception as exc:
            logger.warning(
                f"Conformer model failed to load: {exc}. "
                "Fallback ASR will be unavailable."
            )
            self._loaded = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def transcribe(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio using the Conformer model via CTC decoding.

        Args:
            audio_array: 1-D float32 numpy array, 16 kHz.
            sample_rate: Source sample rate (will resample to 16 kHz if needed).

        Returns:
            Transcribed text string, or empty string on failure.
        """
        if not self._loaded:
            logger.warning("Conformer model not loaded; returning empty string.")
            return ""

        try:
            # Ensure float32 mono
            audio_array = audio_array.astype(np.float32)
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)

            # Resample if needed
            if sample_rate != 16000:
                audio_array = _resample(audio_array, sample_rate, 16000)

            # Normalize
            max_val = np.abs(audio_array).max()
            if max_val > 0:
                audio_array = audio_array / max_val

            inputs = self.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )

            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                logits = self.model(input_values).logits

            # Greedy CTC decoding
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            return transcription.strip()

        except Exception as exc:
            logger.error(f"Conformer transcription failed: {exc}")
            return ""

    def is_loaded(self) -> bool:
        """Return True if the model was successfully loaded."""
        return self._loaded


# ── Module-level helpers ───────────────────────────────────────────────────────

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple resampling via scipy."""
    try:
        import scipy.signal as sps

        num_samples = int(len(audio) * target_sr / orig_sr)
        return sps.resample(audio, num_samples).astype(np.float32)
    except Exception:
        return audio


# ── Streamlit cache wrapper ────────────────────────────────────────────────────

def load_conformer_model(
    model_id: str = "facebook/wav2vec2-conformer-rel-pos-large",
    device: str = "cpu",
) -> ConformerASR:
    """
    Load (or retrieve cached) Conformer ASR model.

    Decorated with @st.cache_resource when called from app/main.py.
    """
    return ConformerASR(model_id=model_id, device=device)
