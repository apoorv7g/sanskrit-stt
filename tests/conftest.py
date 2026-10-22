from pathlib import Path

import pytest

from sanskrit_stt import SanskritTranscriber

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture(scope="session")
def samples() -> Path:
    return SAMPLES


@pytest.fixture(scope="session")
def transcriber() -> SanskritTranscriber:
    stt = SanskritTranscriber()
    stt.warmup()
    return stt
