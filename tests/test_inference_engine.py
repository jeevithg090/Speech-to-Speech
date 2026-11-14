"""Tests for VAD and the streaming inference engine."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from apollo_voice_engine.inference import (
    InferenceConfig,
    InferenceEngine,
    StreamingSession,
    VADConfig,
    VoiceActivityDetector,
)


def pcm(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()


def speech_like(seconds: float, rate: int) -> np.ndarray:
    """Harmonic-rich, amplitude-modulated signal that WebRTC VAD classifies as speech."""
    t = np.arange(int(seconds * rate)) / rate
    voiced = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 20))
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    return (0.3 * voiced * envelope).astype(np.float32)


class FakeVad:
    """Deterministic stand-in: a frame is speech if its mean |amplitude| is large."""

    def is_speech(self, frame: bytes, rate: int) -> bool:
        return np.abs(np.frombuffer(frame, dtype=np.int16)).mean() > 1000


class TestVoiceActivityDetector:
    @pytest.mark.parametrize("cfg", [{"frame_duration_ms": 25}, {"aggressiveness": 4}])
    def test_rejects_invalid_config(self, cfg):
        with pytest.raises(ValueError):
            VoiceActivityDetector(VADConfig(**cfg))

    def test_resample(self):
        x = np.arange(240, dtype=np.int16)
        assert VoiceActivityDetector._resample(x, 24000, 16000).size == 160

    @pytest.mark.parametrize("rate", [16000, 24000])
    def test_silence_is_not_speech(self, rate):
        vad = VoiceActivityDetector(VADConfig(sample_rate=rate))
        assert vad.vad is not None, "webrtcvad should be installed"
        assert not vad.is_speech(pcm(np.zeros(rate // 2)))

    @pytest.mark.parametrize("rate", [16000, 24000])
    def test_speech_like_audio_detected_at_any_rate(self, rate):
        vad = VoiceActivityDetector(VADConfig(sample_rate=rate, aggressiveness=0))
        assert vad.is_speech(pcm(speech_like(0.5, rate)))

    def test_buffer_shorter_than_a_frame(self):
        vad = VoiceActivityDetector(VADConfig(sample_rate=16000))
        assert not vad.is_speech(pcm(np.zeros(10)))

    def test_without_webrtcvad_everything_is_speech(self):
        vad = VoiceActivityDetector(VADConfig())
        vad.vad = None
        assert vad.is_speech(b"\x00\x00" * 100)

    def test_end_of_speech_fires_once_per_utterance(self):
        vad = VoiceActivityDetector(VADConfig(sample_rate=16000, padding_duration_ms=90))
        vad.vad = FakeVad()
        loud, quiet = pcm(np.full(480, 0.5)), pcm(np.zeros(480))
        stream = [quiet] * 5 + [loud] * 3 + [quiet] * 6 + [loud] + [quiet] * 3
        calls: list[int] = []
        frames = list(vad.detect_end_of_speech(iter(stream), lambda: calls.append(1)))
        assert frames == stream  # passthrough
        assert len(calls) == 2  # leading silence does not trigger


class StubSnac:
    def __init__(self):
        self.decoded = None

    def encode(self, audio):
        return torch.tensor([[50000, 50001, 50002, 50003, 50004, 50005, 50006]])

    def decode(self, tokens):
        self.decoded = tokens
        return torch.zeros(4096)


class StubLLM:
    """Echoes an audio reply after the prompt, like generate() with input_ids."""

    def __init__(self, reply_ids):
        self.reply_ids = reply_ids
        self.prompt = None

    def build_audio_input_ids(self, tokens, text_prompt=None):
        self.prompt = torch.tensor([[1, 2, 3, 4]])
        return self.prompt

    def generate_response(self, input_ids=None, **kwargs):
        return torch.cat([input_ids, self.reply_ids], dim=1)

    def extract_audio_tokens(self, new_ids):
        assert new_ids.shape[1] == self.reply_ids.shape[1], "prompt must be stripped"
        return new_ids + 100


class TestInferenceEngine:
    def make_engine(self, reply=None):
        llm = StubLLM(torch.arange(7, 14).unsqueeze(0) if reply is None else reply)
        snac = StubSnac()
        return InferenceEngine(llm, snac, InferenceConfig(max_new_tokens=8)), llm, snac

    def test_generate_response_metrics(self):
        engine, _, snac = self.make_engine()
        audio, metrics = engine.generate_response(torch.zeros(24000))
        assert audio.shape == (4096,)
        assert snac.decoded.tolist() == [list(range(107, 114))]
        assert metrics["input_tokens"] == 4
        assert metrics["output_tokens"] == 7
        assert metrics["e2e_latency_ms"] >= metrics["ttft_ms"] >= 0
        assert engine.get_metrics()["total_requests"] == 1

    def test_less_than_one_frame_is_treated_as_no_audio(self):
        engine, _, snac = self.make_engine(reply=torch.tensor([[1, 2, 3]]))
        audio, _ = engine.generate_response(torch.zeros(24000))
        assert audio is None
        assert snac.decoded is None

    def test_no_audio_tokens(self):
        engine, llm, _ = self.make_engine()
        llm.extract_audio_tokens = lambda ids: None
        audio, metrics = engine.generate_response(torch.zeros(24000))
        assert audio is None
        assert metrics["decode_time_ms"] == 0.0

    def test_stream_response_chunks(self):
        engine, _, _ = self.make_engine()
        chunks = list(engine.stream_response(torch.zeros(100), chunk_size=1000))
        assert [len(c) for c in chunks] == [1000] * 4 + [96]

    def test_reset_metrics(self):
        engine, _, _ = self.make_engine()
        engine.generate_response(torch.zeros(10))
        engine.reset_metrics()
        assert engine.get_metrics()["total_requests"] == 0

    def test_collect_audio_stops_at_silence(self):
        engine, _, _ = self.make_engine()
        engine.vad.vad = FakeVad()
        engine.vad.config = VADConfig(sample_rate=16000)
        for chunk in [np.full(480, 0.5), np.full(480, 0.5), np.zeros(480), np.full(480, 0.5)]:
            engine.process_audio_chunk(chunk.astype(np.float32))
        audio = engine._collect_audio(timeout=1.0)
        assert audio.shape == (1440,)
        assert engine.audio_buffer.qsize() == 1

    def test_collect_audio_empty(self):
        engine, _, _ = self.make_engine()
        assert engine._collect_audio(timeout=0.2) is None

    def test_end_to_end_with_real_components(self, snac, audio_llm):
        engine = InferenceEngine(audio_llm, snac, InferenceConfig(max_new_tokens=6))
        audio, metrics = engine.generate_response(torch.randn(4096) * 0.1, text_context="hello")
        assert metrics["input_tokens"] == 1 + 1 + 14 + 1  # prompt + <as> + 2 frames + <ae>
        assert 0 < metrics["output_tokens"] <= 6
        assert audio is None or audio.dim() == 1


class TestStreamingSession:
    def test_turn_delivers_response_chunks(self):
        engine, _, _ = TestInferenceEngine().make_engine()
        session = StreamingSession(engine, "s1")
        received = []
        session.on_audio_chunk(np.zeros(100))  # ignored before start
        session.start(received.append)
        session.on_audio_chunk(np.zeros(100, dtype=np.float32))
        session.on_speech_end()
        assert session.turn_count == 1
        assert sum(len(c) for c in received) == 4096
        session.on_speech_end()  # no buffered audio -> no new turn
        assert session.turn_count == 1
        session.stop()
        assert not session.is_active
