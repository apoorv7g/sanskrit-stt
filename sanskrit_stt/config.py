"""Runtime settings, overridable through SANSKRIT_STT_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

SAMPLE_RATE = 16_000

# ONNX export of ai4bharat/indicconformer_stt_sa_hybrid_ctc_rnnt_large (CTC head, MIT).
# fp32 is used deliberately: int8 measured ~4x slower and less accurate on x86 laptops.
DEFAULT_MODEL = "OpenVoiceOS/ai4bharat-indicconformer-sa-onnx"


@dataclass
class Settings:
    # Hugging Face repo id, or a local directory containing model.onnx + vocab.txt + config.json.
    model: str = field(default_factory=lambda: os.environ.get("SANSKRIT_STT_MODEL", DEFAULT_MODEL))
    # Clips longer than this are split with Silero VAD; the model was trained on 5-25 s clips.
    # On the samples, 25 s gave lower CER than 20 s (20.1% vs 23.5%): fewer mid-verse cuts.
    max_segment_s: float = field(default_factory=lambda: float(os.environ.get("SANSKRIT_STT_MAX_SEGMENT_S", 25)))
    # VAD pieces separated by less than this are merged (up to max_segment_s). Short values cut
    # verses into 3-6 s fragments and raised long-audio CER from ~21% to ~30%.
    min_silence_ms: float = field(default_factory=lambda: float(os.environ.get("SANSKRIT_STT_MIN_SILENCE_MS", 1000)))
