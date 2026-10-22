"""HTTP API for the (future) UI: POST audio, get Devanagari text back."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from sanskrit_stt import __version__
from sanskrit_stt.audio import AudioError
from sanskrit_stt.config import Settings
from sanskrit_stt.transcriber import SanskritTranscriber

MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def create_app(transcriber: SanskritTranscriber | None = None, *, warmup: bool = True) -> FastAPI:
    stt = transcriber or SanskritTranscriber(Settings())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if warmup:
            stt.warmup()
        yield

    app = FastAPI(title="Sanskrit STT", version=__version__, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "model": stt.settings.model,
            "quantization": stt.settings.quantization or "fp32",
        }

    # Sync handler: FastAPI runs it in a worker thread, keeping the event loop free during inference.
    @app.post("/transcribe")
    def transcribe(
        file: UploadFile = File(..., description="Any audio format ffmpeg can decode"),
        keep_accents: bool = Query(False, description="Keep Vedic svara marks in the output"),
    ) -> dict[str, object]:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            raise HTTPException(400, "empty upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        try:
            result = stt.transcribe(data, keep_accents=keep_accents)
        except AudioError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"filename": file.filename, **result.to_dict()}

    return app
