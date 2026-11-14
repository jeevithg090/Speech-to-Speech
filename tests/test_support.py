"""Tests for settings, metrics and config helpers."""

from __future__ import annotations

import threading

import pytest

from apollo_voice_engine import load_config
from apollo_voice_engine.metrics import RunningMetrics
from apollo_voice_engine.settings import load_settings


def test_settings_defaults(monkeypatch):
    for name in ["APOLLO_HOST", "APOLLO_PORT", "APOLLO_CORS_ORIGINS", "APOLLO_MAX_UPLOAD_MB"]:
        monkeypatch.delenv(name, raising=False)
    s = load_settings(default_port=6969)
    assert (s.host, s.port, s.cors_origins) == ("127.0.0.1", 6969, ["*"])
    assert s.max_upload_bytes == 10 * 1024 * 1024


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("APOLLO_HOST", "0.0.0.0")
    monkeypatch.setenv("APOLLO_PORT", "9000")
    monkeypatch.setenv("APOLLO_CORS_ORIGINS", "https://a.example, https://b.example,")
    monkeypatch.setenv("APOLLO_MAX_UPLOAD_MB", "2")
    s = load_settings()
    assert s.host == "0.0.0.0"
    assert s.port == 9000
    assert s.cors_origins == ["https://a.example", "https://b.example"]
    assert s.max_upload_bytes == 2 * 1024 * 1024


def test_settings_invalid_int(monkeypatch):
    monkeypatch.setenv("APOLLO_PORT", "eighty")
    with pytest.raises(ValueError, match="APOLLO_PORT"):
        load_settings()


def test_running_metrics():
    m = RunningMetrics()
    for v in (10.0, 20.0, 30.0):
        m.observe("latency_ms", v)
    m.increment("requests")
    m.increment("requests", 2)
    assert m.average("latency_ms") == pytest.approx(20.0)
    assert m.average("missing") is None
    snap = m.snapshot()
    assert snap["requests"] == 3
    assert snap["avg_latency_ms"] == 20.0
    assert "started_at" in snap


def test_running_metrics_thread_safe():
    m = RunningMetrics()
    threads = [
        threading.Thread(target=lambda: [m.increment("n") for _ in range(1000)]) for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.counter("n") == 8000


def test_reference_config_loads():
    config = load_config()
    assert config["audio"]["sample_rate"] == 24000
    assert {lang["code"] for lang in config["languages"]} == {"hi", "ta", "te", "kn"}


def test_load_random_sample_resolves_paths():
    import random

    from scripts.unified_cli import load_random_sample

    sample = load_random_sample("indic_voices_dataset", language="ta", rng=random.Random(0))
    assert sample["language"] == "ta"
    assert sample["audio_path"].startswith("indic_voices_dataset/")
    assert load_random_sample("indic_voices_dataset", language="xx") is None
    assert load_random_sample("does-not-exist") is None
