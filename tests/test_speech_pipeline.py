"""Tests for the cascaded pipeline that do not need model downloads."""

from __future__ import annotations

import numpy as np
import pytest

from apollo_voice_engine.speech_pipeline import PipelineConfig, SpeechPipeline, resample


def test_resample_lengths():
    assert resample(np.zeros(24000, dtype=np.float32), 24000, 16000).size == 16000
    x = np.ones(10, dtype=np.float32)
    assert resample(x, 16000, 16000) is x


def test_emergency_short_circuits_without_llm():
    pipeline = SpeechPipeline(PipelineConfig(device="cpu"))
    text, latency, emergency = pipeline.generate_response("I need an ambulance", "ta")
    assert emergency
    assert text == SpeechPipeline.TRANSFER_MESSAGES["ta"]
    assert latency >= 0


def test_non_emergency_requires_llm():
    with pytest.raises(RuntimeError, match="LLM not loaded"):
        SpeechPipeline(PipelineConfig(device="cpu")).generate_response("pharmacy hours")


def test_transcribe_requires_whisper():
    with pytest.raises(RuntimeError, match="Whisper"):
        SpeechPipeline(PipelineConfig(device="cpu")).transcribe(np.zeros(16000))


def test_unknown_tts_language():
    with pytest.raises(ValueError, match="No TTS model"):
        SpeechPipeline(PipelineConfig(device="cpu"))._load_tts("fr")


def test_tts_models_cover_all_languages():
    assert set(PipelineConfig().tts_models) == set(SpeechPipeline.LANGUAGE_CODES)
