import numpy as np
import pytest

from sanskrit_stt.audio import AudioError, duration_s, load_audio


def test_load_wav_path(samples):
    wav = load_audio(samples / "gita_3_1.wav")
    assert wav.dtype == np.float32
    assert 19 < duration_s(wav) < 21
    assert np.abs(wav).max() <= 1.0


def test_load_bytes_matches_path(samples):
    path = samples / "gita_3_4.wav"
    assert np.array_equal(load_audio(path.read_bytes()), load_audio(path))


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_audio("does-not-exist.wav")


def test_garbage_bytes():
    with pytest.raises(AudioError):
        load_audio(b"this is not audio")
