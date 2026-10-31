"""
NLLB-200 Translation module.

Architecture: Seq2Seq Encoder-Decoder Transformer
- Encoder: multi-layer bidirectional Transformer with language-conditioned
           positional embeddings
- Decoder: cross-attends encoder output; uses forced BOS token for target
           language to steer generation
- Supports 200 languages via FLORES+ language codes

Model: facebook/nllb-200-distilled-600M (distilled from 54B parameter model)
"""

import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class NLLBTranslator:
    """
    Neural Machine Translation using Meta's NLLB-200 model.

    Translates text to any of 200 languages identified by FLORES+ codes.
    Supports single and batch (multi-target) translation.
    """

    def __init__(
        self,
        model_id: str = "facebook/nllb-200-distilled-600M",
        device: str = "cpu",
    ):
        """
        Initialize the NLLB-200 translation model.

        Args:
            model_id: HuggingFace model identifier.
            device:   Compute device ('cpu' or 'cuda').
        """
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._load_model()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_model(self):
        """Load model and tokenizer from HuggingFace Hub."""
        try:
            # Suppress version-check noise that raises ImportError on hub version mismatches
            import transformers
            from transformers import AutoModelForSeq2SeqLM

            # NllbTokenizer may not exist in older transformers — fall back to AutoTokenizer
            try:
                from transformers import NllbTokenizer
                self.tokenizer = NllbTokenizer.from_pretrained(self.model_id)
            except (ImportError, AttributeError):
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            logger.info(f"Loading NLLB-200 model: {self.model_id} on {self.device}")

            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                low_cpu_mem_usage=True,
            )
            if torch_dtype == torch.float16:
                self.model = self.model.half()
            self.model.to(self.device)
            self.model.eval()

            self._loaded = True
            logger.info("NLLB-200 model loaded successfully.")

        except Exception as exc:
            logger.error(f"Failed to load NLLB-200 model: {exc}")
            self._loaded = False
            raise

    def _get_target_token_id(self, target_lang_nllb: str) -> int:
        """Retrieve the forced BOS token ID for the target language."""
        try:
            # New transformers API
            if hasattr(self.tokenizer, "lang_code_to_id"):
                return self.tokenizer.lang_code_to_id[target_lang_nllb]
            # Fallback: convert token to ID directly
            tok_id = self.tokenizer.convert_tokens_to_ids(target_lang_nllb)
            if tok_id != self.tokenizer.unk_token_id:
                return tok_id
            raise KeyError(target_lang_nllb)
        except KeyError:
            logger.warning(
                f"Language code '{target_lang_nllb}' not found in tokenizer. "
                "Falling back to English."
            )
            fallback = getattr(self.tokenizer, "lang_code_to_id", {})
            return fallback.get("eng_Latn", self.tokenizer.convert_tokens_to_ids("eng_Latn") or 0)

    # ── Public API ─────────────────────────────────────────────────────────────

    def translate(
        self,
        text: str,
        source_lang_nllb: str = "eng_Latn",
        target_lang_nllb: str = "hin_Deva",
        *,
        tgt_lang: str | None = None,          # alias accepted by gradio_app.py
        src_lang: str | None = None,           # alias accepted by gradio_app.py
        max_length: int = 512,
        num_beams: int = 4,
    ) -> str:
        """
        Translate text from source to a single target language.

        Args:
            text:               Text to translate.
            source_lang_nllb:  FLORES+ source language code (e.g. 'eng_Latn').
            target_lang_nllb:  FLORES+ target language code (e.g. 'hin_Deva').
            max_length:         Maximum generated token length.
            num_beams:          Beam search width.

        Returns:
            Translated text string, or empty string on failure.
        """
        # Accept keyword aliases
        if tgt_lang is not None:
            target_lang_nllb = tgt_lang
        if src_lang is not None:
            source_lang_nllb = src_lang

        if not self._loaded:
            return ""
        if not text or not text.strip():
            return ""

        try:
            self.tokenizer.src_lang = source_lang_nllb

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            forced_bos_token_id = self._get_target_token_id(target_lang_nllb)

            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )

            translated = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            return translated[0].strip() if translated else ""

        except Exception as exc:
            logger.error(f"Translation failed ({source_lang_nllb}→{target_lang_nllb}): {exc}")
            return ""

    def translate_batch(
        self,
        text: str,
        source_lang_nllb: str,
        target_langs: List[Dict],
        max_length: int = 512,
        num_beams: int = 4,
    ) -> Dict[str, str]:
        """
        Translate text to multiple target languages.

        Args:
            text:              Text to translate.
            source_lang_nllb: FLORES+ source language code.
            target_langs:     List of dicts with keys 'name' and 'nllb_code'.
                              Example: [{"name": "Hindi", "nllb_code": "hin_Deva"}, ...]
            max_length:        Maximum generated token length.
            num_beams:         Beam search width.

        Returns:
            dict mapping language name → translated text.
        """
        results: Dict[str, str] = {}

        for lang_info in target_langs:
            lang_name = lang_info["name"]
            target_code = lang_info["nllb_code"]

            translated = self.translate(
                text=text,
                source_lang_nllb=source_lang_nllb,
                target_lang_nllb=target_code,
                max_length=max_length,
                num_beams=num_beams,
            )
            results[lang_name] = translated

        return results

    def is_loaded(self) -> bool:
        """Return True if the model was successfully loaded."""
        return self._loaded


# ── Streamlit cache wrapper ────────────────────────────────────────────────────

def load_translation_model(
    model_id: str = "facebook/nllb-200-distilled-600M",
    device: str = "cpu",
) -> NLLBTranslator:
    """
    Load (or retrieve cached) NLLB-200 translation model.

    Decorated with @st.cache_resource when called from app/main.py.
    """
    return NLLBTranslator(model_id=model_id, device=device)
