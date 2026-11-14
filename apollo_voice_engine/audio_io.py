"""Audio decoding/encoding helpers shared by the demo servers."""

from __future__ import annotations

import base64
import io
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


class AudioDecodeError(RuntimeError):
    """Raised when uploaded audio cannot be decoded."""


def convert_to_wav(
    src_path: str | os.PathLike[str],
    dst_path: str | os.PathLike[str],
    sample_rate: int = 16000,
    timeout: float = 30.0,
) -> None:
    """Convert any ffmpeg-readable file to mono 16-bit PCM WAV at ``sample_rate``."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src_path),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(dst_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise AudioDecodeError("ffmpeg is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError("ffmpeg timed out while decoding audio") from exc
    if result.returncode != 0:
        raise AudioDecodeError(f"ffmpeg could not decode audio: {result.stderr.strip()[:200]}")


def read_wav(path: str | os.PathLike[str]) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV file as float32 in ``[-1, 1]`` (mono-mixed)."""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise AudioDecodeError("Only 16-bit PCM WAV is supported")
        channels = wf.getnchannels()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def decode_audio_bytes(data: bytes, sample_rate: int) -> np.ndarray:
    """Decode arbitrary encoded audio (webm, ogg, mp3, wav, ...) to float32 mono."""
    with tempfile.TemporaryDirectory(prefix="apollo_audio_") as tmp:
        src = Path(tmp) / "input"
        dst = Path(tmp) / "output.wav"
        src.write_bytes(data)
        convert_to_wav(src, dst, sample_rate=sample_rate)
        samples, _ = read_wav(dst)
    return samples


def float_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float waveform (any shape that squeezes to 1-D) as 16-bit WAV bytes."""
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buffer.getvalue()


def float_to_wav_base64(audio: np.ndarray, sample_rate: int) -> str:
    return base64.b64encode(float_to_wav_bytes(audio, sample_rate)).decode()
