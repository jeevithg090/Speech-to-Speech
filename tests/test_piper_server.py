"""Tests for the standard-library Piper TTS server (Piper itself is stubbed)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer

import pytest

from scripts import piper_tts_server as piper


class FakeVoice:
    def synthesize_wav(self, text, wav_file, syn_config=None):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 160)


@pytest.fixture
def base_url(monkeypatch):
    monkeypatch.setattr(
        piper,
        "VOICES",
        {
            "en_US-lessac-medium": {
                "voice": FakeVoice(),
                "config": piper.VOICE_CONFIG["en_US-lessac-medium"],
            }
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), piper.PiperHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def request(url, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.headers.get("Content-Type"), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers.get("Content-Type"), err.read()


def test_health_and_voices(base_url):
    status, _, body = request(f"{base_url}/health")
    assert status == 200 and json.loads(body)["voices_loaded"] == 1
    voices = json.loads(request(f"{base_url}/voices")[2])["voices"]
    assert voices[0]["id"] == "en_US-lessac-medium"


def test_demo_page(base_url):
    status, ctype, body = request(f"{base_url}/")
    assert status == 200 and ctype.startswith("text/html") and b"Piper" in body


@pytest.mark.parametrize("path", ["/README.md", "/.env.example", "/../pyproject.toml", "/scripts/"])
def test_does_not_serve_arbitrary_files(base_url, path):
    assert request(f"{base_url}{path}")[0] == 404


def test_synthesize_wav(base_url, tmp_path):
    status, ctype, body = request(f"{base_url}/synthesize", {"text": "hello", "length_scale": 1.2})
    assert status == 200 and ctype == "audio/wav"
    path = tmp_path / "out.wav"
    path.write_bytes(body)
    with wave.open(str(path)) as wf:
        assert wf.getnframes() == 160


def test_synthesize_base64(base_url):
    status, _, body = request(f"{base_url}/synthesize", {"text": "hello", "format": "base64"})
    assert status == 200 and json.loads(body)["format"] == "wav"


@pytest.mark.parametrize(
    "payload,code",
    [
        ({"text": ""}, 400),
        ({"text": "hi", "voice": "missing"}, 404),
        ({"text": "hi", "length_scale": "fast"}, 400),
        (["not", "an", "object"], 400),
    ],
)
def test_synthesize_errors(base_url, payload, code):
    assert request(f"{base_url}/synthesize", payload)[0] == code


def test_unknown_post_route(base_url):
    assert request(f"{base_url}/nope", {"text": "x"})[0] == 404
