"""
Audio utility helpers.
"""

import io
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def audio_array_to_bytes(array: np.ndarray, sr: int = 16000) -> bytes:
    """
    Convert a float32 numpy array to WAV bytes.

    Args:
        array: 1-D float32 audio array.
        sr:    Sample rate.

    Returns:
        WAV-encoded bytes.
    """
    try:
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, array.astype(np.float32), sr, format="WAV")
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        logger.error(f"audio_array_to_bytes failed: {exc}")
        return b""


def bytes_to_audio_array(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Convert audio bytes (WAV/MP3/OGG/etc.) to a float32 numpy array.

    Args:
        audio_bytes: Raw audio file bytes.

    Returns:
        Tuple (audio_array, sample_rate).
    """
    try:
        import soundfile as sf

        buf = io.BytesIO(audio_bytes)
        array, sr = sf.read(buf, dtype="float32", always_2d=False)
        if array.ndim > 1:
            array = array.mean(axis=1)
        return array, sr

    except Exception:
        try:
            from pydub import AudioSegment

            buf = io.BytesIO(audio_bytes)
            seg = AudioSegment.from_file(buf)
            seg = seg.set_channels(1).set_frame_rate(16000)
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            samples /= 32768.0
            return samples, 16000
        except Exception as exc2:
            logger.error(f"bytes_to_audio_array failed: {exc2}")
            return np.zeros(16000, dtype=np.float32), 16000


def resample_audio(
    array: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """
    Resample audio to a different sample rate.

    Args:
        array:     Input float32 audio array.
        orig_sr:   Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled float32 numpy array.
    """
    if orig_sr == target_sr:
        return array

    try:
        import librosa
        return librosa.resample(array, orig_sr=orig_sr, target_sr=target_sr)
    except Exception:
        try:
            import scipy.signal as sps
            num = int(len(array) * target_sr / orig_sr)
            return sps.resample(array, num).astype(np.float32)
        except Exception as exc:
            logger.error(f"resample_audio failed: {exc}")
            return array


def trim_silence(
    array: np.ndarray, sr: int = 16000, top_db: int = 20
) -> np.ndarray:
    """
    Trim leading and trailing silence from an audio array.

    Args:
        array:  Input float32 audio array.
        sr:     Sample rate (unused but kept for API consistency).
        top_db: Silence threshold in dB below reference.

    Returns:
        Trimmed audio array.
    """
    try:
        import librosa
        trimmed, _ = librosa.effects.trim(array, top_db=top_db)
        return trimmed
    except Exception as exc:
        logger.warning(f"trim_silence failed: {exc}")
        return array


def is_silent(
    array: np.ndarray, threshold: float = 0.01
) -> bool:
    """
    Heuristic check for whether audio is essentially silent.

    Args:
        array:     Float32 audio array.
        threshold: RMS energy threshold below which audio is considered silent.

    Returns:
        True if the audio RMS energy is below threshold.
    """
    if len(array) == 0:
        return True
    rms = float(np.sqrt(np.mean(array.astype(np.float32) ** 2)))
    return rms < threshold
