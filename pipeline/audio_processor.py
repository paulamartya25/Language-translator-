"""
Audio processing pipeline utilities.

Handles microphone recording, audio format conversion, mel spectrogram
computation, chunking, and normalization.
"""

import io
import logging
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default audio parameters
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1


class AudioProcessor:
    """
    Comprehensive audio processing class for the translation pipeline.

    Handles:
    - Microphone recording via sounddevice
    - Audio-to-mel-spectrogram conversion (librosa)
    - Long-audio chunking with overlap
    - Format conversions (bytes ↔ numpy array)
    - Normalization
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_audio(
        self, duration_seconds: float = 10.0, sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """
        Record audio from the default microphone.

        Args:
            duration_seconds: Recording duration in seconds.
            sample_rate:      Sample rate; defaults to self.sample_rate.

        Returns:
            1-D float32 numpy array of recorded audio samples.
        """
        sr = sample_rate or self.sample_rate
        try:
            import sounddevice as sd

            logger.info(f"Recording {duration_seconds}s of audio at {sr} Hz...")
            audio = sd.rec(
                int(duration_seconds * sr),
                samplerate=sr,
                channels=DEFAULT_CHANNELS,
                dtype="float32",
            )
            sd.wait()  # Block until done
            return audio.flatten()

        except Exception as exc:
            logger.error(f"Audio recording failed: {exc}")
            return np.zeros(int(duration_seconds * sr), dtype=np.float32)

    # ── Spectrogram ───────────────────────────────────────────────────────────

    def audio_to_mel_spectrogram(
        self,
        audio_array: np.ndarray,
        sr: Optional[int] = None,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
    ) -> np.ndarray:
        """
        Compute 80-channel Log-Mel Spectrogram.

        Whisper uses 80 mel bins, n_fft=400 (25 ms @ 16 kHz),
        hop_length=160 (10 ms @ 16 kHz).

        Args:
            audio_array: 1-D float32 audio array.
            sr:          Sample rate.
            n_mels:      Number of Mel filter banks.
            n_fft:       FFT window size.
            hop_length:  STFT hop length.

        Returns:
            2-D numpy array of shape (n_mels, time_frames) — log-mel spectrogram.
        """
        sr = sr or self.sample_rate
        try:
            import librosa

            audio_array = audio_array.astype(np.float32)
            mel = librosa.feature.melspectrogram(
                y=audio_array,
                sr=sr,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                fmin=0.0,
                fmax=8000.0,
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            return log_mel

        except Exception as exc:
            logger.error(f"Mel spectrogram computation failed: {exc}")
            return np.zeros((n_mels, 1), dtype=np.float32)

    # ── Chunking ──────────────────────────────────────────────────────────────

    def chunk_audio(
        self,
        audio_array: np.ndarray,
        chunk_size: int = 30,
        overlap: int = 2,
        sample_rate: Optional[int] = None,
    ):
        """
        Split long audio into overlapping chunks for Whisper's 30-second window.

        Args:
            audio_array: 1-D float32 audio array.
            chunk_size:  Chunk duration in seconds.
            overlap:     Overlap between consecutive chunks in seconds.
            sample_rate: Sample rate; defaults to self.sample_rate.

        Yields:
            Tuple (chunk_array, start_time_seconds)
        """
        sr = sample_rate or self.sample_rate
        chunk_samples = int(chunk_size * sr)
        overlap_samples = int(overlap * sr)
        step = chunk_samples - overlap_samples

        start = 0
        while start < len(audio_array):
            end = min(start + chunk_samples, len(audio_array))
            chunk = audio_array[start:end]
            yield chunk, start / sr
            if end >= len(audio_array):
                break
            start += step

    # ── Format conversion ─────────────────────────────────────────────────────

    def bytes_to_array(
        self, audio_bytes: bytes
    ) -> Tuple[np.ndarray, int]:
        """
        Convert audio bytes (WAV/MP3/etc.) to a numpy array.

        Args:
            audio_bytes: Raw audio bytes.

        Returns:
            Tuple (audio_array, sample_rate).
        """
        try:
            import soundfile as sf

            buf = io.BytesIO(audio_bytes)
            audio_array, sr = sf.read(buf, dtype="float32", always_2d=False)
            # Convert stereo to mono
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            return audio_array, sr

        except Exception:
            # Fallback: try pydub (handles MP3, M4A, etc.)
            try:
                from pydub import AudioSegment

                buf = io.BytesIO(audio_bytes)
                seg = AudioSegment.from_file(buf)
                seg = seg.set_channels(1).set_frame_rate(self.sample_rate)
                samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
                # Normalize int16 range
                samples /= 32768.0
                return samples, self.sample_rate

            except Exception as exc2:
                logger.error(f"bytes_to_array failed: {exc2}")
                return np.zeros(self.sample_rate, dtype=np.float32), self.sample_rate

    def array_to_bytes(
        self, audio_array: np.ndarray, sample_rate: Optional[int] = None
    ) -> bytes:
        """
        Convert a numpy audio array to WAV bytes.

        Args:
            audio_array: 1-D float32 audio array.
            sample_rate: Sample rate; defaults to self.sample_rate.

        Returns:
            WAV bytes.
        """
        sr = sample_rate or self.sample_rate
        try:
            import soundfile as sf

            buf = io.BytesIO()
            sf.write(buf, audio_array.astype(np.float32), sr, format="WAV")
            buf.seek(0)
            return buf.read()
        except Exception as exc:
            logger.error(f"array_to_bytes failed: {exc}")
            return b""

    # ── Normalization ─────────────────────────────────────────────────────────

    def normalize_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Normalize audio samples to the range [-1, 1].

        Args:
            audio_array: Input audio array.

        Returns:
            Normalized float32 array.
        """
        audio_array = audio_array.astype(np.float32)
        max_val = np.abs(audio_array).max()
        if max_val > 0:
            return audio_array / max_val
        return audio_array

    # ── Resampling ────────────────────────────────────────────────────────────

    def resample(
        self, audio_array: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to a different sample rate.

        Args:
            audio_array: Input audio array.
            orig_sr:     Original sample rate.
            target_sr:   Target sample rate.

        Returns:
            Resampled float32 array.
        """
        if orig_sr == target_sr:
            return audio_array

        try:
            import librosa
            return librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)
        except Exception:
            try:
                import scipy.signal as sps
                num = int(len(audio_array) * target_sr / orig_sr)
                return sps.resample(audio_array, num).astype(np.float32)
            except Exception as exc:
                logger.error(f"Resampling failed: {exc}")
                return audio_array

    # ── Silence trimming ──────────────────────────────────────────────────────

    def trim_silence(
        self,
        audio_array: np.ndarray,
        sr: Optional[int] = None,
        top_db: int = 20,
    ) -> np.ndarray:
        """
        Trim leading and trailing silence.

        Args:
            audio_array: Input audio array.
            sr:          Sample rate (unused, for API compatibility).
            top_db:      Silence threshold in dB below reference.

        Returns:
            Trimmed audio array.
        """
        try:
            import librosa
            trimmed, _ = librosa.effects.trim(audio_array, top_db=top_db)
            return trimmed
        except Exception as exc:
            logger.warning(f"Silence trimming failed: {exc}")
            return audio_array
