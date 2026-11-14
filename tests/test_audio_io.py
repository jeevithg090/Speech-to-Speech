"""Tests for audio encode/decode helpers."""

from __future__ import annotations

import base64
import shutil
import wave

import numpy as np
import pytest

from apollo_voice_engine.audio_io import (
    AudioDecodeError,
    decode_audio_bytes,
    float_to_wav_base64,
    float_to_wav_bytes,
    read_wav,
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def tone(seconds=0.25, rate=16000):
    t = np.arange(int(seconds * rate)) / rate
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_wav_roundtrip(tmp_path):
    audio = tone()
    path = tmp_path / "x.wav"
    path.write_bytes(float_to_wav_bytes(audio, 16000))
    back, rate = read_wav(path)
    assert rate == 16000
    np.testing.assert_allclose(back, audio, atol=1e-4)


def test_wav_bytes_clip_and_flatten():
    data = float_to_wav_bytes(np.array([[2.0, -2.0]]), 8000)
    with wave.open(__import__("io").BytesIO(data)) as wf:
        assert wf.getnchannels() == 1
        frames = np.frombuffer(wf.readframes(2), dtype=np.int16)
    assert frames.tolist() == [32767, -32767]


def test_base64_wrapper():
    assert base64.b64decode(float_to_wav_base64(tone(), 16000))[:4] == b"RIFF"


def test_read_wav_stereo_is_mixed_down(tmp_path):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(np.array([1000, 3000, 1000, 3000], dtype=np.int16).tobytes())
    samples, _ = read_wav(path)
    np.testing.assert_allclose(samples, [2000 / 32768] * 2)


@needs_ffmpeg
def test_decode_audio_bytes_resamples():
    decoded = decode_audio_bytes(float_to_wav_bytes(tone(0.5, 16000), 16000), sample_rate=24000)
    assert abs(decoded.size - 12000) <= 32
    assert decoded.dtype == np.float32


@needs_ffmpeg
def test_decode_garbage_raises():
    with pytest.raises(AudioDecodeError):
        decode_audio_bytes(b"definitely not audio", sample_rate=16000)
