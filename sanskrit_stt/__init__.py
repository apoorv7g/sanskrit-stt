"""Sanskrit speech-to-text built on AI4Bharat IndicConformer (ONNX)."""

from sanskrit_stt.config import Settings
from sanskrit_stt.transcriber import Segment, SanskritTranscriber, Transcript

__all__ = ["SanskritTranscriber", "Segment", "Settings", "Transcript"]
__version__ = "0.1.0"
