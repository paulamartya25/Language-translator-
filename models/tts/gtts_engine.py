"""
Text-to-Speech engine using gTTS (Google Text-to-Speech).

Provides synthesis for all 40+ supported languages via the Google TTS API.
For Indian languages, gTTS uses Google's neural TTS voices which provide
acceptable quality for most Indic scripts.
"""

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Text-to-Speech synthesis using gTTS.

    Converts translated text strings into MP3 audio bytes that can be
    served directly through Streamlit's st.audio() widget.
    """

    def __init__(self, slow: bool = False, timeout: int = 30):
        """
        Initialize the TTS engine.

        Args:
            slow:    If True, synthesize speech at reduced speed.
            timeout: Network request timeout in seconds.
        """
        self.slow = slow
        self.timeout = timeout

    # ── Public API ─────────────────────────────────────────────────────────────

    def synthesize(self, text: str, lang_code: str) -> Optional[bytes]:
        """
        Synthesize speech for the given text.

        Args:
            text:      Text to convert to speech.
            lang_code: BCP-47 language code (e.g. 'hi', 'en', 'ja').

        Returns:
            MP3 audio as bytes, or None on failure.
        """
        if not text or not text.strip():
            return None

        try:
            from gtts import gTTS

            tts = gTTS(text=text.strip(), lang=lang_code, slow=self.slow)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer.read()

        except Exception as exc:
            logger.error(f"TTS synthesis failed for lang '{lang_code}': {exc}")
            return None

    def synthesize_to_file(
        self, text: str, lang_code: str, filepath: str
    ) -> bool:
        """
        Synthesize speech and save to a file.

        Args:
            text:      Text to convert to speech.
            lang_code: BCP-47 language code.
            filepath:  Destination file path (should end in .mp3).

        Returns:
            True on success, False on failure.
        """
        if not text or not text.strip():
            return False

        try:
            from gtts import gTTS

            tts = gTTS(text=text.strip(), lang=lang_code, slow=self.slow)
            tts.save(filepath)
            return True

        except Exception as exc:
            logger.error(f"TTS save failed for lang '{lang_code}': {exc}")
            return False

    def get_supported_languages(self) -> dict:
        """
        Return dict of languages supported by gTTS.

        Returns:
            dict mapping language name → BCP-47 code.
        """
        try:
            from gtts.lang import tts_langs
            return tts_langs()
        except Exception:
            return {}

    def is_language_supported(self, lang_code: str) -> bool:
        """Check if a language code is supported by gTTS."""
        supported = self.get_supported_languages()
        return lang_code in supported
