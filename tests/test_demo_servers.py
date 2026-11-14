"""HTTP-level tests for the demo servers (models and network services stubbed)."""

from __future__ import annotations

import base64
import shutil
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from apollo_voice_engine.audio_io import float_to_wav_bytes
from scripts import api_demo, speech_demo

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class TestApiDemo:
    @pytest.fixture
    def client(self, monkeypatch):
        from apollo_voice_engine.metrics import RunningMetrics

        monkeypatch.setattr(api_demo, "metrics", RunningMetrics())
        return TestClient(api_demo.app)

    def test_ui_served_with_metadata(self, client):
        html = client.get("/").text
        assert '<meta name="description"' in html
        assert 'aria-live="polite"' in html

    def test_languages(self, client):
        codes = [lang["code"] for lang in client.get("/api/languages").json()["languages"]]
        assert codes == ["hi", "ta", "te", "kn", "en"]

    def test_chat_and_metrics(self, client):
        r = client.post("/api/chat", json={"text": "Where is the pharmacy?", "language": "en"})
        assert r.status_code == 200
        body = r.json()
        assert body["action"] == "RESPOND"
        assert "pharmacy" in body["response"].lower()
        assert body["latency_ms"] >= 0

        r = client.post("/api/chat", json={"text": "मुझे छाती में दर्द हो रहा है", "language": "hi"})
        assert r.json()["action"] == "TRANSFER_TO_HUMAN"

        m = client.get("/api/metrics").json()
        assert m["total_requests"] == 2
        assert m["emergency_transfers"] == 1

    def test_unsupported_language_falls_back_to_english(self, client):
        r = client.post("/api/chat", json={"text": "hello", "language": "xx"})
        assert r.json()["language"] == "en"

    @pytest.mark.parametrize("payload", [{"text": ""}, {"text": "x" * 5000}, {}])
    def test_chat_validation(self, client, payload):
        assert client.post("/api/chat", json=payload).status_code == 422

    def test_classify(self, client):
        body = client.post("/api/classify", json={"text": "call an ambulance"}).json()
        assert body["level"] == "emergency"
        assert body["should_transfer"] is True


class TestSpeechDemo:
    @pytest.fixture
    def client(self, monkeypatch):
        async def fake_tts(text, lang):
            return b"ID3fake-mp3"

        monkeypatch.setattr(speech_demo, "synthesize_speech", fake_tts)
        fake_model = types.SimpleNamespace(
            transcribe=lambda path, **kw: {"text": " I have chest pain "}
        )
        monkeypatch.setattr(speech_demo, "get_whisper_model", lambda: fake_model)
        return TestClient(speech_demo.app)

    def test_ui_served(self, client):
        html = client.get("/").text
        assert "textContent = text" in html  # chat bubbles never render HTML

    def test_text_chat_returns_audio(self, client):
        body = client.post(
            "/api/chat", json={"text": "book an appointment", "language": "ta"}
        ).json()
        assert body["action"] == "RESPOND"
        assert base64.b64decode(body["audio"]) == b"ID3fake-mp3"
        assert body["tts_error"] is None

    def test_tts_failure_degrades_to_text(self, client, monkeypatch):
        async def broken_tts(text, lang):
            raise ConnectionError("offline")

        monkeypatch.setattr(speech_demo, "synthesize_speech", broken_tts)
        body = client.post("/api/chat", json={"text": "hello", "language": "en"}).json()
        assert body["audio"] == ""
        assert body["tts_error"]

    def test_rejects_tiny_upload(self, client):
        r = client.post("/api/speech-to-speech", files={"audio": ("a.webm", b"123")})
        assert r.status_code == 400

    def test_rejects_oversized_upload(self, client, monkeypatch):
        monkeypatch.setattr(
            speech_demo, "settings", speech_demo.settings.__class__(max_upload_bytes=2000)
        )
        r = client.post("/api/speech-to-speech", files={"audio": ("a.webm", b"0" * 3000)})
        assert r.status_code == 413

    @needs_ffmpeg
    def test_undecodable_upload(self, client):
        r = client.post("/api/speech-to-speech", files={"audio": ("a.webm", b"0" * 3000)})
        assert r.status_code == 422

    @needs_ffmpeg
    def test_speech_to_speech_flow(self, client):
        wav = float_to_wav_bytes(np.sin(np.linspace(0, 2000, 16000)) * 0.3, 16000)
        r = client.post(
            "/api/speech-to-speech",
            files={"audio": ("a.wav", wav, "audio/wav")},
            data={"language": "en"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["transcription"] == "I have chest pain"
        assert body["action"] == "TRANSFER_TO_HUMAN"
        assert body["stt_ms"] >= 0


class TestSharedWebUi:
    """Assets, favicon and the 404 page installed by scripts.web_ui."""

    @pytest.fixture
    def client(self):
        return TestClient(api_demo.app)

    def test_design_system_assets_served(self, client):
        css = client.get("/assets/apollo.css")
        assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
        assert "prefers-color-scheme: dark" in css.text
        assert client.get("/assets/apollo-ui.js").status_code == 200
        assert client.get("/favicon.ico").headers["content-type"] == "image/svg+xml"

    def test_pages_link_shared_assets_and_metadata(self, client):
        html = client.get("/").text
        for snippet in ('href="assets/apollo.css"', 'rel="icon"', 'property="og:title"', "<title>"):
            assert snippet in html

    def test_unknown_page_returns_html_404_for_browsers(self, client):
        r = client.get("/no-such-page", headers={"Accept": "text/html"})
        assert r.status_code == 404
        assert "Page not found" in r.text

    def test_unknown_api_route_stays_json(self, client):
        r = client.get("/api/nope", headers={"Accept": "text/html"})
        assert r.status_code == 404 and r.json() == {"detail": "Not Found"}
        assert client.get("/no-such-page").json() == {"detail": "Not Found"}

    def test_asset_traversal_blocked(self, client):
        assert client.get("/assets/../../pyproject.toml").status_code == 404
