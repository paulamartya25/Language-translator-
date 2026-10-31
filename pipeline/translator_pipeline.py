"""
Main orchestration pipeline for the Multilingual AI Translator.

Ties together ASR (Whisper / Conformer), Translation (NLLB-200),
and TTS (gTTS) into a single, easy-to-call interface.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from config.languages import (
    ALL_LANGUAGES,
    whisper_to_nllb,
    get_language_name_from_whisper,
)
from config.settings import WHISPER_NO_SPEECH_THRESHOLD

logger = logging.getLogger(__name__)


class TranslatorPipeline:
    """
    End-to-end translation pipeline.

    Steps:
        1. ASR  — transcribe audio → text + detected source language
        2. NLLB — translate text to every selected target language
        3. TTS  — synthesise speech for each translation

    Falls back to Conformer ASR when Whisper confidence is low.
    """

    def __init__(self, asr_model, translator, tts_engine, conformer_model=None):
        """
        Initialise the pipeline.

        Args:
            asr_model:       WhisperASR instance.
            translator:      NLLBTranslator instance.
            tts_engine:      TTSEngine instance.
            conformer_model: ConformerASR instance (optional fallback).
        """
        self.asr = asr_model
        self.translator = translator
        self.tts = tts_engine
        self.conformer = conformer_model

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_audio(
        self,
        audio_input: np.ndarray,
        target_languages: List[str],
        sample_rate: int = 16000,
        source_lang_override: Optional[str] = None,
        synthesize_audio: bool = True,
        num_beams: int = 4,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: audio → transcription → translations → TTS.

        Args:
            audio_input:          1-D float32 audio numpy array.
            target_languages:     List of target language names (keys in ALL_LANGUAGES).
            sample_rate:          Sample rate of audio_input.
            source_lang_override: Optional Whisper BCP-47 code to skip auto-detection.
            synthesize_audio:     Whether to run TTS for each translation.
            num_beams:            Beam width for translation.
            progress_callback:    Optional callable(step: int, total: int, label: str).

        Returns:
            dict:
                - "transcription"  : str  — original transcribed text
                - "source_language": str  — detected language name
                - "source_lang_code": str — BCP-47 code
                - "nllb_source"    : str  — NLLB FLORES+ source code
                - "translations"   : dict {lang_name: {"text": str, "audio": bytes|None}}
                - "error"          : str|None
        """
        result: Dict[str, Any] = {
            "transcription": "",
            "source_language": "Unknown",
            "source_lang_code": "en",
            "nllb_source": "eng_Latn",
            "translations": {},
            "error": None,
        }

        total_steps = 2 + len(target_languages)  # ASR + N translations (TTS bundled)
        step = 0

        # ── Step 1: ASR ───────────────────────────────────────────────────────
        try:
            if progress_callback:
                progress_callback(step, total_steps, "🎤 Transcribing audio…")

            asr_result = self.asr.transcribe(
                audio_array=audio_input,
                sample_rate=sample_rate,
                language=source_lang_override,
            )

            transcription = asr_result.get("text", "").strip()
            detected_lang = asr_result.get("language", source_lang_override or "en")

            # Conformer fallback if transcription is empty / suspicious
            if (not transcription) and self.conformer and self.conformer.is_loaded():
                logger.info("Whisper returned empty result; trying Conformer fallback.")
                if progress_callback:
                    progress_callback(step, total_steps, "🔄 Conformer fallback ASR…")
                transcription = self.conformer.transcribe(audio_input, sample_rate)
                detected_lang = detected_lang or "en"

            result["transcription"] = transcription
            result["source_lang_code"] = detected_lang
            result["source_language"] = get_language_name_from_whisper(detected_lang)
            result["nllb_source"] = whisper_to_nllb(detected_lang)

            step += 1

        except Exception as exc:
            logger.error(f"ASR step failed: {exc}")
            result["error"] = f"ASR failed: {exc}"
            return result

        if not transcription:
            result["error"] = "No speech detected in the audio."
            return result

        # ── Step 2+: Translation + TTS per language ───────────────────────────
        translations: Dict[str, Dict[str, Any]] = {}

        for lang_name in target_languages:
            step += 1
            if lang_name not in ALL_LANGUAGES:
                logger.warning(f"Unknown language: {lang_name}; skipping.")
                continue

            lang_info = ALL_LANGUAGES[lang_name]
            nllb_code = lang_info["nllb_code"]
            gtts_code = lang_info["gtts_lang"]

            if progress_callback:
                flag = lang_info.get("flag", "")
                progress_callback(step, total_steps, f"🌐 Translating → {flag} {lang_name}…")

            # Translation
            translated_text = ""
            try:
                translated_text = self.translator.translate(
                    text=transcription,
                    source_lang_nllb=result["nllb_source"],
                    target_lang_nllb=nllb_code,
                    num_beams=num_beams,
                )
            except Exception as exc:
                logger.error(f"Translation to {lang_name} failed: {exc}")

            # TTS
            audio_bytes: Optional[bytes] = None
            if synthesize_audio and translated_text:
                try:
                    audio_bytes = self.tts.synthesize(translated_text, gtts_code)
                except Exception as exc:
                    logger.warning(f"TTS for {lang_name} failed: {exc}")

            translations[lang_name] = {
                "text": translated_text,
                "audio": audio_bytes,
                "nllb_code": nllb_code,
                "gtts_code": gtts_code,
                "flag": lang_info.get("flag", ""),
            }

        result["translations"] = translations

        if progress_callback:
            progress_callback(total_steps, total_steps, "✅ Done!")

        return result

    def process_text(
        self,
        text: str,
        source_lang_nllb: str,
        target_languages: List[str],
        synthesize_audio: bool = True,
        num_beams: int = 4,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Text-only pipeline (skip ASR): translate and synthesise TTS.

        Args:
            text:              Input text to translate.
            source_lang_nllb: NLLB FLORES+ code of the source language.
            target_languages: List of target language names.
            synthesize_audio: Whether to run TTS.
            num_beams:        Beam width for translation.
            progress_callback: Optional callable(step, total, label).

        Returns:
            Same structure as process_audio() but without ASR fields.
        """
        result: Dict[str, Any] = {
            "transcription": text,
            "source_language": "Manual Input",
            "source_lang_code": source_lang_nllb,
            "nllb_source": source_lang_nllb,
            "translations": {},
            "error": None,
        }

        total_steps = len(target_languages)
        step = 0

        translations: Dict[str, Dict[str, Any]] = {}

        for lang_name in target_languages:
            step += 1
            if lang_name not in ALL_LANGUAGES:
                continue

            lang_info = ALL_LANGUAGES[lang_name]
            nllb_code = lang_info["nllb_code"]
            gtts_code = lang_info["gtts_lang"]

            if progress_callback:
                flag = lang_info.get("flag", "")
                progress_callback(step, total_steps, f"🌐 Translating → {flag} {lang_name}…")

            translated_text = ""
            try:
                translated_text = self.translator.translate(
                    text=text,
                    source_lang_nllb=source_lang_nllb,
                    target_lang_nllb=nllb_code,
                    num_beams=num_beams,
                )
            except Exception as exc:
                logger.error(f"Translation to {lang_name} failed: {exc}")

            audio_bytes: Optional[bytes] = None
            if synthesize_audio and translated_text:
                try:
                    audio_bytes = self.tts.synthesize(translated_text, gtts_code)
                except Exception as exc:
                    logger.warning(f"TTS for {lang_name} failed: {exc}")

            translations[lang_name] = {
                "text": translated_text,
                "audio": audio_bytes,
                "nllb_code": nllb_code,
                "gtts_code": gtts_code,
                "flag": lang_info.get("flag", ""),
            }

        result["translations"] = translations
        return result
