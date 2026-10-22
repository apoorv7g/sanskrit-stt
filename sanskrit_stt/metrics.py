"""Word and character error rates (no external dependencies)."""

from __future__ import annotations

from collections.abc import Sequence

from sanskrit_stt.text import normalize_for_scoring


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize_for_scoring(reference).split()
    hyp = normalize_for_scoring(hypothesis).split()
    return edit_distance(ref, hyp) / max(len(ref), 1)


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate ignoring spaces.

    Sanskrit word boundaries are ambiguous under sandhi (e.g. "चेत्कर्मणस्ते" vs "चेत् कर्मणः ते"),
    so CER without spaces is the more meaningful number.
    """
    ref = normalize_for_scoring(reference).replace(" ", "")
    hyp = normalize_for_scoring(hypothesis).replace(" ", "")
    return edit_distance(ref, hyp) / max(len(ref), 1)
