"""Decode any audio/video input to 16 kHz mono float32 using ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sanskrit_stt.config import SAMPLE_RATE


class AudioError(RuntimeError):
    """Raised when audio cannot be decoded."""


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise AudioError("ffmpeg not found on PATH; install it from https://ffmpeg.org/download.html")
    return path


def load_audio(source: str | Path | bytes) -> npt.NDArray[np.float32]:
    """Return a 16 kHz mono float32 waveform in [-1, 1] from a file path or raw file bytes."""
    if isinstance(source, (str, Path)):
        if not Path(source).is_file():
            raise FileNotFoundError(source)
        input_arg, stdin = str(source), None
    else:
        input_arg, stdin = "pipe:0", source

    cmd = [
        ffmpeg_path(), "-nostdin", "-v", "error",
        "-i", input_arg,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "pipe:1",
    ]
    # -nostdin conflicts with reading from pipe:0, so drop it for byte input.
    if stdin is not None:
        cmd.remove("-nostdin")

    proc = subprocess.run(cmd, input=stdin, capture_output=True, check=False)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg failed to decode audio: {proc.stderr.decode(errors='replace').strip()}")

    waveform = np.frombuffer(proc.stdout, dtype=np.float32)
    if waveform.size == 0:
        raise AudioError("audio contains no samples")
    return waveform.copy()


def duration_s(waveform: npt.NDArray[np.float32]) -> float:
    return waveform.size / SAMPLE_RATE
