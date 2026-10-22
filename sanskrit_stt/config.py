"""Runtime settings, overridable through SANSKRIT_STT_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

SAMPLE_RATE = 16_000

# ONNX export of ai4bharat/indicconformer_stt_sa_hybrid_ctc_rnnt_large (CTC head, MIT).
DEFAULT_MODEL = "OpenVoiceOS/ai4bharat-indicconformer-sa-onnx"


def _env(name: str, default: str | None) -> str | None:
    value = os.environ.get(f"SANSKRIT_STT_{name}")
    return value if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))  # type: ignore[arg-type]


@dataclass
class Settings:
    # Hugging Face repo id, or a local directory containing model.onnx + vocab.txt + config.json.
    model: str = field(default_factory=lambda: _env("MODEL", DEFAULT_MODEL))  # type: ignore[arg-type]
    # None = fp32. "int8" is smaller but measured ~4x slower and less accurate on x86 laptops.
    quantization: str | None = field(default_factory=lambda: _env("QUANTIZATION", None))
    # Clips longer than this are split with Silero VAD; the model was trained on 5-25 s clips.
    # On the samples, 25 s gave lower CER than 20 s (20.1% vs 23.5%): fewer mid-verse cuts.
    max_segment_s: float = field(default_factory=lambda: _env_float("MAX_SEGMENT_S", 25.0))
    # VAD pieces separated by less than this are merged (up to max_segment_s). Short values cut
    # verses into 3-6 s fragments and raised long-audio CER from ~21% to ~30%.
    min_silence_ms: float = field(default_factory=lambda: _env_float("MIN_SILENCE_MS", 1000.0))
    vad_threshold: float = field(default_factory=lambda: _env_float("VAD_THRESHOLD", 0.5))
    speech_pad_ms: float = field(default_factory=lambda: _env_float("SPEECH_PAD_MS", 200.0))
    # Strip Vedic svara marks (udatta/anudatta etc.) that the model sometimes emits.
    keep_accents: bool = field(default_factory=lambda: _env("KEEP_ACCENTS", "0") == "1")
