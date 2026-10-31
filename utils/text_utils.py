"""
Text utility helpers for pre/post-processing translation text.
"""

import re
import unicodedata
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Unicode script ranges for common scripts
_SCRIPT_RANGES = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali":    (0x0980, 0x09FF),
    "Telugu":     (0x0C00, 0x0C7F),
    "Tamil":      (0x0B80, 0x0BFF),
    "Gujarati":   (0x0A80, 0x0AFF),
    "Kannada":    (0x0C80, 0x0CFF),
    "Malayalam":  (0x0D00, 0x0D7F),
    "Gurmukhi":   (0x0A00, 0x0A7F),
    "Odia":       (0x0B00, 0x0B7F),
    "Arabic":     (0x0600, 0x06FF),
    "Hebrew":     (0x0590, 0x05FF),
    "CJK":        (0x4E00, 0x9FFF),
    "Hangul":     (0xAC00, 0xD7AF),
    "Thai":       (0x0E00, 0x0E7F),
    "Greek":      (0x0370, 0x03FF),
    "Cyrillic":   (0x0400, 0x04FF),
}


def clean_text(text: str) -> str:
    """
    Clean and normalise text for translation input.

    - Strips leading/trailing whitespace
    - Collapses multiple whitespace characters into a single space
    - Removes non-printable control characters (except newlines)
    - Normalises unicode to NFC form

    Args:
        text: Raw input string.

    Returns:
        Cleaned string.
    """
    if not text:
        return ""

    # Unicode normalisation
    text = unicodedata.normalize("NFC", text)

    # Remove non-printable control chars (keep newlines/tabs)
    text = re.sub(r"[^\S\n\t]+", " ", text)       # collapse whitespace
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # strip control

    # Strip outer whitespace
    text = text.strip()

    return text


def detect_script(text: str) -> str:
    """
    Detect the dominant writing script in text.

    Iterates over characters and tallies which Unicode script range they
    fall into. Returns the name of the most common script found.

    Args:
        text: Input string.

    Returns:
        Script name (e.g. 'Devanagari', 'Latin', 'Arabic', 'CJK') or
        'Latin' as the default fallback.
    """
    if not text:
        return "Latin"

    counts: dict = {}
    for ch in text:
        cp = ord(ch)
        for script_name, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[script_name] = counts.get(script_name, 0) + 1
                break

    if not counts:
        return "Latin"

    return max(counts, key=counts.get)


def truncate_text(
    text: str,
    max_length: int = 512,
    suffix: str = "…",
    word_boundary: bool = True,
) -> str:
    """
    Truncate text to a maximum character length, respecting word boundaries.

    Args:
        text:           Input string.
        max_length:     Maximum allowed character length.
        suffix:         String appended when truncation occurs.
        word_boundary:  If True, truncate at the last complete word.

    Returns:
        Possibly truncated string.
    """
    if not text or len(text) <= max_length:
        return text

    truncated = text[: max_length - len(suffix)]

    if word_boundary:
        # Find the last whitespace within the truncated slice
        last_space = truncated.rfind(" ")
        if last_space > max_length // 2:  # only snap if it's reasonably close
            truncated = truncated[:last_space]

    return truncated.rstrip() + suffix


def split_into_sentences(text: str, max_chars: int = 400) -> list:
    """
    Split text into sentences for chunked translation.

    Splits on '.', '!', '?' followed by whitespace or end-of-string.
    Merges short sentences to avoid tiny chunks.

    Args:
        text:      Input string.
        max_chars: Target maximum characters per chunk.

    Returns:
        List of sentence strings.
    """
    if not text:
        return []

    # Split on sentence-ending punctuation
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""

    for sentence in raw:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def count_words(text: str) -> int:
    """Count approximate word count in a string."""
    if not text:
        return 0
    return len(text.split())


def is_rtl_script(script: str) -> bool:
    """Return True if the script is right-to-left."""
    return script in {"Arabic", "Hebrew", "Nastaliq"}
