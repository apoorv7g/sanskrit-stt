"""Devanagari text clean-up for model output and reference transcripts."""

from __future__ import annotations

import re
import unicodedata

# Vedic svara/tone marks: Devanagari stress signs, Vedic Extensions, Devanagari Extended.
_VEDIC_ACCENTS = re.compile("[॑-॔᳐-᳿꣠-꣱]")
# Dandas, Devanagari/ASCII digits and common punctuation (used for scoring only).
_PUNCT = re.compile(r"[।॥०-९0-9.,;:!?\"'()\[\]{}\-–—|]")
_SPACES = re.compile(r"\s+")


def normalize(text: str, *, keep_accents: bool = False) -> str:
    """NFC-normalise, optionally strip Vedic accent marks, and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    if not keep_accents:
        text = _VEDIC_ACCENTS.sub("", text)
    return _SPACES.sub(" ", text).strip()


def normalize_for_scoring(text: str) -> str:
    """Normalisation applied to both hypothesis and reference before computing WER/CER."""
    return normalize(_PUNCT.sub(" ", text))
