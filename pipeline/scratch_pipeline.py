"""
End-to-end pipeline using models TRAINED FROM SCRATCH.

Instead of loading pre-trained HuggingFace models, this pipeline uses:
  - ASRTransformer  — custom Encoder-Decoder trained from scratch
  - NMTTransformer  — custom Seq2Seq trained from scratch
  - FastSpeech2     — custom Text-to-Mel trained from scratch
  - HiFiGANGenerator — custom vocoder trained from scratch

Load pre-trained checkpoints with .load_checkpoint() before using.
"""

import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


class ScratchTranslatorPipeline:
    """
    Full Speech-to-Speech Translation Pipeline using from-scratch trained models.

    Pipeline:
      Audio → LogMelFeatureExtractor → ASRTransformer → text
           → NMTTransformer (with lang tokens) → translated text
           → FastSpeech2 → mel spectrogram
           → HiFiGANGenerator → waveform bytes
    """

    def __init__(
        self,
        asr_checkpoint: Optional[str] = None,
        nmt_checkpoint: Optional[str] = None,
        tts_checkpoint: Optional[str] = None,
        hifigan_checkpoint: Optional[str] = None,
        # Model hyper-parameters (must match training config)
        asr_vocab_size: int = 50,
        nmt_vocab_size: int = 32000,
        tts_vocab_size: int = 200,
        d_model_asr: int = 256,
        d_model_nmt: int = 512,
        d_model_tts: int = 256,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        # ── Mel feature extractor ──────────────────────────────────────────────
        from modules.mel_features import LogMelFeatureExtractor
        self.mel_extractor = LogMelFeatureExtractor(sample_rate=16000).to(self.device)

        # ── ASR Transformer ────────────────────────────────────────────────────
        from models.asr.asr_transformer import ASRTransformer
        self.asr = ASRTransformer(
            vocab_size=asr_vocab_size,
            d_model=d_model_asr,
            num_enc_layers=6,
            num_dec_layers=6,
            num_heads=8,
            d_ff=d_model_asr * 4,
        ).to(self.device)
        if asr_checkpoint and os.path.exists(asr_checkpoint):
            self._load_checkpoint(self.asr, asr_checkpoint)
            logger.info(f"Loaded ASR checkpoint: {asr_checkpoint}")

        # ── ASR Tokenizer ──────────────────────────────────────────────────────
        from data.asr_dataset import BPETokenizer
        self.asr_tokenizer = BPETokenizer()

        # ── NMT Transformer ────────────────────────────────────────────────────
        from models.nmt.nmt_transformer import NMTTransformer
        self.nmt = NMTTransformer(
            vocab_size=nmt_vocab_size,
            d_model=d_model_nmt,
            num_enc_layers=6,
            num_dec_layers=6,
            num_heads=8,
            d_ff=d_model_nmt * 4,
        ).to(self.device)
        if nmt_checkpoint and os.path.exists(nmt_checkpoint):
            self._load_checkpoint(self.nmt, nmt_checkpoint)
            logger.info(f"Loaded NMT checkpoint: {nmt_checkpoint}")

        # ── NMT Tokenizer ──────────────────────────────────────────────────────
        from data.nmt_dataset import MultilingualTokenizer
        self.nmt_tokenizer = MultilingualTokenizer(vocab_size=nmt_vocab_size)

        # ── FastSpeech2 (Text-to-Mel) ──────────────────────────────────────────
        from models.tts.fastspeech2 import FastSpeech2
        self.tts = FastSpeech2(
            vocab_size=tts_vocab_size,
            d_model=d_model_tts,
            num_enc_layers=4,
            num_dec_layers=4,
        ).to(self.device)
        if tts_checkpoint and os.path.exists(tts_checkpoint):
            self._load_checkpoint(self.tts, tts_checkpoint)
            logger.info(f"Loaded TTS checkpoint: {tts_checkpoint}")

        # ── HiFi-GAN Vocoder ───────────────────────────────────────────────────
        from models.tts.hifigan import HiFiGANGenerator
        self.vocoder = HiFiGANGenerator(in_channels=80).to(self.device)
        if hifigan_checkpoint and os.path.exists(hifigan_checkpoint):
            self._load_checkpoint(self.vocoder, hifigan_checkpoint)
            logger.info(f"Loaded HiFi-GAN checkpoint: {hifigan_checkpoint}")

        logger.info("ScratchTranslatorPipeline initialized.")

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _load_checkpoint(model: torch.nn.Module, path: str):
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)

    def _audio_to_mel(self, audio: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """Convert numpy audio array to log-mel spectrogram tensor."""
        wav = torch.from_numpy(audio.astype(np.float32))
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav / (wav.abs().max() + 1e-8)
        mel = self.mel_extractor(wav.unsqueeze(0).to(self.device))
        return mel  # (1, 80, T)

    def _mel_to_wav_bytes(self, mel: torch.Tensor) -> bytes:
        """Convert mel spectrogram to WAV bytes via HiFi-GAN."""
        self.vocoder.eval()
        with torch.no_grad():
            wav = self.vocoder(mel)  # (1, 1, T_wav)
        wav = wav.squeeze().cpu().float()
        wav = torch.clamp(wav, -1.0, 1.0)
        # Encode as WAV
        buf = io.BytesIO()
        torchaudio.save(buf, wav.unsqueeze(0), sample_rate=16000, format="wav")
        buf.seek(0)
        return buf.read()

    # ── Public API ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        """ASR: audio → transcribed text."""
        self.asr.eval()
        mel = self._audio_to_mel(audio, sr)
        token_ids = self.asr.generate(
            mel,
            bos_id=self.asr_tokenizer.BOS,
            eos_id=self.asr_tokenizer.EOS,
            max_len=200,
        )
        return self.asr_tokenizer.decode(token_ids[0])

    @torch.no_grad()
    def translate(self, text: str, target_nllb: str) -> str:
        """NMT: text → translated text in target language."""
        self.nmt.eval()
        src_ids = (
            [self.nmt_tokenizer.BOS]
            + self.nmt_tokenizer.encode(text)[:250]
            + [self.nmt_tokenizer.EOS]
        )
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        lang_id = self.nmt_tokenizer.get_lang_token_id(target_nllb)
        token_ids = self.nmt.translate(src_tensor, lang_token_id=lang_id,
                                       eos_id=self.nmt_tokenizer.EOS)
        return self.nmt_tokenizer.decode(token_ids[0][1:])  # skip lang token

    @torch.no_grad()
    def synthesize(self, text: str) -> bytes:
        """TTS: text → WAV bytes via FastSpeech2 + HiFi-GAN."""
        self.tts.eval()
        # Character-level tokenize for TTS
        char_ids = [min(ord(c) % 190 + 5, 199) for c in text[:200]]
        tokens = torch.tensor(char_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        mel, _, _, _ = self.tts(tokens)  # (1, 80, T)
        return self._mel_to_wav_bytes(mel)

    @torch.no_grad()
    def process(
        self,
        audio: np.ndarray,
        target_languages: List[str],
        sr: int = 16000,
        synthesize_audio: bool = True,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: audio → transcription → translations → TTS.

        Args:
            audio:            1-D float32 numpy audio array
            target_languages: list of language names (keys in ALL_LANGUAGES)
            sr:               sample rate of audio
            synthesize_audio: whether to run TTS
            progress_callback: optional callable(step, total, label)

        Returns:
            dict with keys: transcription, translations (dict → text + audio bytes)
        """
        from config.languages import ALL_LANGUAGES

        total = 2 + len(target_languages)
        step = 0

        # Step 1: ASR
        if progress_callback:
            progress_callback(step, total, "🎤 Transcribing with ASR Transformer…")
        transcription = self.transcribe(audio, sr)
        step += 1

        translations: Dict[str, Dict] = {}
        for lang_name in target_languages:
            step += 1
            lang_info = ALL_LANGUAGES.get(lang_name, {})
            nllb_code = lang_info.get("nllb_code", "eng_Latn")

            if progress_callback:
                flag = lang_info.get("flag", "")
                progress_callback(step, total, f"🌐 Translating → {flag} {lang_name}…")

            # NMT Translation
            translated = ""
            try:
                translated = self.translate(transcription, nllb_code)
            except Exception as exc:
                logger.error(f"NMT translation to {lang_name} failed: {exc}")

            # TTS
            audio_bytes: Optional[bytes] = None
            if synthesize_audio and translated:
                try:
                    audio_bytes = self.synthesize(translated)
                except Exception as exc:
                    logger.warning(f"TTS for {lang_name} failed: {exc}")

            translations[lang_name] = {
                "text": translated,
                "audio": audio_bytes,
                "flag": lang_info.get("flag", "🌐"),
                "nllb_code": nllb_code,
            }

        if progress_callback:
            progress_callback(total, total, "✅ Done!")

        return {
            "transcription": transcription,
            "source_language": "Detected by ASR",
            "translations": translations,
        }
