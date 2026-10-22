import pytest
from fastapi.testclient import TestClient

from sanskrit_stt.server import create_app

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def client(transcriber):
    with TestClient(create_app(transcriber, warmup=False)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["quantization"] == "fp32"


def test_transcribe_upload(client, samples):
    with (samples / "gita_3_4.wav").open("rb") as f:
        resp = client.post("/transcribe", files={"file": ("gita_3_4.wav", f, "audio/wav")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "gita_3_4.wav"
    assert "समधि" in body["text"]
    assert body["segments"][0]["start"] == 0.0


def test_rejects_non_audio(client):
    resp = client.post("/transcribe", files={"file": ("x.wav", b"not audio", "audio/wav")})
    assert resp.status_code == 422


def test_rejects_empty(client):
    resp = client.post("/transcribe", files={"file": ("x.wav", b"", "audio/wav")})
    assert resp.status_code == 400
