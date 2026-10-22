"""Core transcription pipeline: decode -> (VAD split if long) -> IndicConformer CTC -> clean text."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx_asr

from sanskrit_stt.audio import duration_s, load_audio
from sanskrit_stt.config import SAMPLE_RATE, Settings
from sanskrit_stt.text import normalize


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    segments: list[Segment] = field(default_factory=list)
    duration_s: float = 0.0
    processing_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SanskritTranscriber:
    """Loads the model lazily once; safe to share across threads."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._model: Any = None
        self._vad_model: Any = None
        self._lock = threading.Lock()

    def load(self) -> SanskritTranscriber:
        with self._lock:
            if self._model is None:
                model = self.settings.model
                path = model if Path(model).is_dir() else None
                self._model = onnx_asr.load_model(
                    "nemo-conformer-ctc" if path else model,
                    path,
                    quantization=self.settings.quantization,
                )
                self._vad_model = self._model.with_vad(
                    onnx_asr.load_vad("silero"),
                    threshold=self.settings.vad_threshold,
                    max_speech_duration_s=self.settings.max_segment_s,
                    min_silence_duration_ms=self.settings.min_silence_ms,
                    speech_pad_ms=self.settings.speech_pad_ms,
                )
        return self

    def warmup(self) -> None:
        """Run one dummy pass; the first ONNX Runtime call is several times slower than later ones."""
        self.load()
        self._model.recognize(np.zeros(SAMPLE_RATE, dtype=np.float32), sample_rate=SAMPLE_RATE)

    def transcribe(
        self,
        source: str | Path | bytes | npt.NDArray[np.float32],
        *,
        keep_accents: bool | None = None,
    ) -> Transcript:
        """Transcribe a file path, raw file bytes, or a 16 kHz mono float32 array."""
        self.load()
        keep = self.settings.keep_accents if keep_accents is None else keep_accents
        waveform = source if isinstance(source, np.ndarray) else load_audio(source)
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)
        total = duration_s(waveform)

        started = time.perf_counter()
        if total <= self.settings.max_segment_s:
            text = normalize(self._model.recognize(waveform, sample_rate=SAMPLE_RATE), keep_accents=keep)
            segments = [Segment(0.0, round(total, 2), text)] if text else []
        else:
            segments = [
                Segment(round(s.start, 2), round(s.end, 2), normalize(s.text, keep_accents=keep))
                for s in self._vad_model.recognize(waveform, sample_rate=SAMPLE_RATE)
            ]
            segments = [s for s in segments if s.text]

        return Transcript(
            text=" ".join(s.text for s in segments),
            segments=segments,
            duration_s=round(total, 2),
            processing_s=round(time.perf_counter() - started, 3),
        )
