import csv

import numpy as np
import pytest

from sanskrit_stt.audio import load_audio
from sanskrit_stt.config import SAMPLE_RATE
from sanskrit_stt.metrics import cer

pytestmark = pytest.mark.model


def _references(samples):
    return list(csv.reader((samples / "references.tsv").open(encoding="utf-8"), delimiter="\t"))


def test_samples_transcribe_close_to_reference(transcriber, samples):
    for name, reference in _references(samples):
        result = transcriber.transcribe(samples / name)
        assert result.text, name
        # All samples are under max_segment_s, so they go through the model in one pass.
        assert len(result.segments) == 1
        # Baseline is 14-25% CER per sample; guard against regressions, not perfection.
        assert cer(reference, result.text) < 0.32, (name, result.text)


def test_output_is_devanagari_without_accents(transcriber, samples):
    text = transcriber.transcribe(samples / "gita_3_1.wav").text
    assert all(ch == " " or "ऀ" <= ch <= "ॿ" for ch in text), text
    assert "॑" not in text and "॒" not in text


def test_long_audio_is_split_with_vad(transcriber, samples):
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    parts = [load_audio(samples / name) for name, _ in _references(samples)]
    long_audio = np.concatenate([p for part in parts for p in (part, silence)])
    assert long_audio.size / SAMPLE_RATE > transcriber.settings.max_segment_s

    result = transcriber.transcribe(long_audio)
    assert len(result.segments) >= 3
    assert all(s.end - s.start <= transcriber.settings.max_segment_s + 1 for s in result.segments)
    assert [s.start for s in result.segments] == sorted(s.start for s in result.segments)
    full_reference = " ".join(ref for _, ref in _references(samples))
    assert cer(full_reference, result.text) < 0.3, result.text
