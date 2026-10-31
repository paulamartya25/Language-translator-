"""
Global configuration settings for the Multilingual AI Translator.
"""

import torch

# ── Model configuration ────────────────────────────────────────────────────────
# ASR: Whisper — switch to "openai/whisper-large-v3" for best quality
ASR_MODEL = "openai/whisper-base"

# Conformer fallback ASR model
CONFORMER_MODEL = "facebook/wav2vec2-conformer-rel-pos-large"

# Translation: NLLB-200 distilled 600M parameters
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"

# ── Audio settings ─────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000          # Target sample rate for ASR models (Hz)
CHUNK_DURATION = 30          # Max chunk duration in seconds (Whisper window)
OVERLAP_DURATION = 2         # Overlap between chunks in seconds
MEL_BINS = 80                # Number of Mel filter-bank bins
N_FFT = 400                  # FFT window size
HOP_LENGTH = 160             # STFT hop length

# ── App settings ───────────────────────────────────────────────────────────────
APP_TITLE = "🌐 Multilingual AI Translator"
APP_ICON = "🌐"
APP_DESCRIPTION = (
    "Translate speech or text into 40+ languages powered by "
    "Whisper ASR, NLLB-200, and gTTS."
)
MAX_TRANSLATION_BATCH = 10   # Languages per batch during parallel translation

# Available Whisper model sizes (for UI selector)
WHISPER_MODEL_OPTIONS = {
    "Tiny (fastest, ~39M params)": "openai/whisper-tiny",
    "Base (~74M params)": "openai/whisper-base",
    "Small (~244M params)": "openai/whisper-small",
    "Medium (~769M params)": "openai/whisper-medium",
    "Large v3 (best quality, ~1.5B params)": "openai/whisper-large-v3",
}

# ── TTS settings ───────────────────────────────────────────────────────────────
TTS_SLOW_MODE = False        # Slow down speech synthesis
TTS_TIMEOUT = 30             # gTTS request timeout in seconds

# ── Confidence thresholds ──────────────────────────────────────────────────────
# If Whisper no-speech probability > this, trigger Conformer fallback
WHISPER_NO_SPEECH_THRESHOLD = 0.6
WHISPER_LOGPROB_THRESHOLD = -1.0

# ── UI ─────────────────────────────────────────────────────────────────────────
COLUMNS_PER_ROW = 3          # Language cards per grid row
MAX_AUDIO_RECORD_SECONDS = 60  # Maximum microphone recording duration
DEFAULT_RECORD_SECONDS = 10    # Default recording duration
