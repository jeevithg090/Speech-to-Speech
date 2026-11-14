"""Tests for the unified SNAC + AudioLLM demo engine (session anchoring, fallback, API)."""

from __future__ import annotations

import asyncio
import base64
import shutil
import sys
import types

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from apollo_voice_engine.audio_io import float_to_wav_bytes
from apollo_voice_engine.models.speaker_encoder import SpeakerEncoder
from scripts import unified_demo
from scripts.unified_demo import UnifiedConfig, UnifiedVoiceEngine


def voice(seed: int, seconds: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * 24000)) / 24000
    return (0.3 * np.sin(2 * np.pi * rng.uniform(100, 300) * t)).astype(np.float32)


@pytest.fixture
def no_edge_tts(monkeypatch):
    """Make the Edge-TTS fallback fail fast (no network in tests)."""
    monkeypatch.setitem(sys.modules, "edge_tts", None)


@pytest.fixture
def engine(snac, audio_llm) -> UnifiedVoiceEngine:
    eng = UnifiedVoiceEngine(UnifiedConfig(device="cpu", max_response_tokens=4))
    eng.snac, eng.audio_llm = snac, audio_llm
    eng.speaker_encoder = SpeakerEncoder(device="cpu")
    eng.models_loaded = True
    return eng


def test_prepare_audio_bounds():
    eng = UnifiedVoiceEngine(UnifiedConfig(device="cpu", max_audio_duration=1.0))
    assert eng.prepare_audio(np.zeros(10)).shape == (1, 1, 4800)
    assert eng.prepare_audio(np.zeros(48000)).shape == (1, 1, 24000)
    assert eng.prepare_audio(torch.zeros(1, 10000)).shape == (1, 1, 10000)


def test_speaker_encoder_is_deterministic_and_distinct():
    enc = SpeakerEncoder(device="cpu")
    a1, a2, b = enc(voice(1)), enc(voice(1)), enc(voice(2))
    assert torch.equal(a1, a2)
    assert not torch.allclose(a1, b)
    assert torch.linalg.norm(a1).item() == pytest.approx(1.0)


@pytest.mark.parametrize("language", ["hi", "ta", "te", "kn", "en"])
def test_session_anchor_persists_until_reset(engine, language):
    """Ported from the former anchoring simulation script."""
    seen = []
    original = engine.audio_llm.prepare_audio_input

    def spy(tokens, text_prompt=None, speaker_embedding=None):
        seen.append(speaker_embedding)
        return original(tokens, text_prompt=text_prompt, speaker_embedding=speaker_embedding)

    engine.audio_llm.prepare_audio_input = spy
    override = torch.full((1, 14), 50000)
    run = asyncio.run

    run(
        engine.process_speech(
            voice(100), language, reset_anchor=True, cached_response_tokens=override
        )
    )
    anchor1 = engine.anchor_embedding
    run(
        engine.process_speech(
            voice(200), language, reset_anchor=False, cached_response_tokens=override
        )
    )
    assert torch.equal(engine.anchor_embedding, anchor1)  # different speaker, anchor locked
    run(
        engine.process_speech(
            voice(200), language, reset_anchor=True, cached_response_tokens=override
        )
    )
    assert not torch.equal(engine.anchor_embedding, anchor1)
    assert seen == []  # override bypasses the LLM entirely


def test_speaker_embedding_reaches_llm(engine, no_edge_tts):
    seen = []
    original = engine.audio_llm.prepare_audio_input

    def spy(tokens, text_prompt=None, speaker_embedding=None):
        seen.append(speaker_embedding)
        return original(tokens, text_prompt=text_prompt, speaker_embedding=speaker_embedding)

    engine.audio_llm.prepare_audio_input = spy
    asyncio.run(engine.process_speech(voice(1), "hi", reset_anchor=True))
    asyncio.run(engine.process_speech(voice(2), "hi"))
    assert len(seen) == 2
    assert torch.equal(seen[0], seen[1])


def test_process_speech_with_cached_tokens_returns_wav(engine):
    tokens = engine.snac.encode(torch.randn(4096) * 0.1)
    result = asyncio.run(engine.process_speech(voice(1), "hi", cached_response_tokens=tokens))
    assert result["success"] and not result["fallback_mode"]
    assert result["audio_format"] == "wav"
    assert base64.b64decode(result["audio"])[:4] == b"RIFF"
    assert set(result["metrics"]) == {
        "snac_encode_ms",
        "llm_generate_ms",
        "snac_decode_ms",
        "total_ms",
    }
    assert engine.metrics.counter("total_requests") == 1


def test_untrained_model_output_falls_back(engine, no_edge_tts):
    result = asyncio.run(engine.process_speech(voice(1), "kn"))
    assert result["fallback_mode"] is True
    assert result["success"] is False  # fallback TTS unavailable offline
    assert result["error"]


def test_models_not_loaded_uses_fallback(no_edge_tts):
    eng = UnifiedVoiceEngine(UnifiedConfig(device="cpu"))
    result = asyncio.run(eng.process_speech(voice(1), "zz"))
    assert result["fallback_mode"]
    assert "models not loaded" in result["transcription"]
    assert result["response_text"].startswith("Hello")


def test_fallback_with_tts(monkeypatch):
    class FakeCommunicate:
        def __init__(self, text, voice):
            self.voice = voice

        async def stream(self):
            yield {"type": "audio", "data": b"mp3"}
            yield {"type": "WordBoundary"}

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=FakeCommunicate))
    eng = UnifiedVoiceEngine(UnifiedConfig(device="cpu"))
    result = asyncio.run(eng.process_speech(voice(1), "ta"))
    assert result["success"]
    assert base64.b64decode(result["audio"]) == b"mp3"
    assert result["audio_format"] == "mp3"


class TestUnifiedApi:
    @pytest.fixture
    def client(self, engine, monkeypatch, no_edge_tts):
        monkeypatch.setattr(UnifiedVoiceEngine, "load_models", lambda self: None)
        with TestClient(unified_demo.app) as client:
            monkeypatch.setattr(unified_demo, "engine", engine)
            yield client

    def test_health_and_architecture(self, client):
        assert client.get("/api/health").json()["models_loaded"] is True
        arch = client.get("/api/architecture").json()
        assert arch["token_vocabulary"]["snac_tokens_per_frame"] == 7
        assert (
            arch["token_vocabulary"]["audio_token_offset"]
            == unified_demo.engine.audio_llm.audio_offset
        )

    def test_empty_upload_rejected(self, client):
        assert client.post("/api/speech-to-speech", files={"audio": ("a", b"")}).status_code == 400

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_speech_to_speech(self, client):
        wav = float_to_wav_bytes(voice(3), 24000)
        r = client.post(
            "/api/speech-to-speech",
            files={"audio": ("a.wav", wav)},
            data={"language": "hi", "reset_anchor": "true"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["anchor_status"] == "New Identity"
        assert len(body["anchor_hash"]) == 4
