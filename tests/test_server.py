import pytest
from fastapi.testclient import TestClient

from sanskrit_stt.server import create_app

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def client(transcriber):
    with TestClient(create_app(transcriber, warmup=False)) as c:
        yield c


def test_serves_web_app(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Tiro+Devanagari+Sanskrit" in page.text
    for asset in ("app.js", "styles.css", "capture-worklet.js"):
        assert client.get(f"/{asset}").status_code == 200, asset


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_transcribe_upload(client, samples):
    with (samples / "gita_3_4.wav").open("rb") as f:
        resp = client.post("/api/transcribe", files={"file": ("gita_3_4.wav", f, "audio/wav")})
    assert resp.status_code == 200
    body = resp.json()
    assert "समधि" in body["text"]
    assert body["segments"][0]["start"] == 0.0


def test_rejects_non_audio(client):
    resp = client.post("/api/transcribe", files={"file": ("x.wav", b"not audio", "audio/wav")})
    assert resp.status_code == 422


def test_rejects_empty(client):
    resp = client.post("/api/transcribe", files={"file": ("x.wav", b"", "audio/wav")})
    assert resp.status_code == 400
