"""Web app: serves the recorder UI and the transcription API from one process."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from sanskrit_stt import __version__
from sanskrit_stt.audio import AudioError
from sanskrit_stt.transcriber import SanskritTranscriber

WEB_DIR = Path(__file__).parent / "web"
# The UI sends <=25 s WAV chunks (~0.8 MB); this leaves room for uploaded recordings via the API.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def create_app(transcriber: SanskritTranscriber | None = None, *, warmup: bool = True) -> FastAPI:
    stt = transcriber or SanskritTranscriber()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if warmup:
            stt.warmup()
        yield

    app = FastAPI(title="Sanskrit STT", version=__version__, lifespan=lifespan)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": stt.settings.model}

    # Sync handler: FastAPI runs it in a worker thread, keeping the event loop free during inference.
    @app.post("/api/transcribe")
    def transcribe(file: UploadFile = File(..., description="Any audio format ffmpeg can decode")) -> dict[str, object]:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            raise HTTPException(400, "empty upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        try:
            return stt.transcribe(data).to_dict()
        except AudioError as exc:
            raise HTTPException(422, str(exc)) from exc

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
