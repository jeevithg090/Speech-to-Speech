"""
Piper TTS demo server.

A small standard-library HTTP server exposing local Piper neural TTS voices
plus a browser demo page (``demo/piper_demo.html``).

Usage:
    pip install -r requirements-piper.txt
    python3 -m piper.download_voices en_US-lessac-medium
    python scripts/piper_tts_server.py [--open]      # http://127.0.0.1:8085
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apollo_voice_engine.settings import load_settings  # noqa: E402

try:
    from piper import PiperVoice

    PIPER_AVAILABLE = True
except ImportError:
    PiperVoice = None
    PIPER_AVAILABLE = False

DEMO_PAGE = PROJECT_ROOT / "demo" / "piper_demo.html"
VOICE_DIR = Path(os.getenv("PIPER_VOICE_DIR", Path.home() / ".local" / "share" / "piper_voices"))
MAX_BODY_BYTES = 64 * 1024

VOICE_CONFIG: dict[str, dict[str, str]] = {
    "en_US-lessac-medium": {
        "name": "English (US) - Lessac",
        "language": "en",
        "description": "Clear American English voice",
    },
    "hi_IN-rohan-medium": {
        "name": "Hindi - Rohan",
        "language": "hi",
        "description": "Hindi male voice",
    },
}

VOICES: dict[str, dict[str, Any]] = {}


def find_voice_file(voice_name: str) -> Path | None:
    """Locate the ONNX model file for ``voice_name`` in the known voice directories."""
    candidates = [
        PROJECT_ROOT / f"{voice_name}.onnx",
        VOICE_DIR / f"{voice_name}.onnx",
        Path("/usr/share/piper-voices") / f"{voice_name}.onnx",
    ]
    return next((p for p in candidates if p.exists()), None)


def load_voices() -> None:
    """Load every configured voice that is present on disk."""
    if not PIPER_AVAILABLE:
        print("piper-tts not installed. Run: pip install -r requirements-piper.txt")
        return
    for voice_name, config in VOICE_CONFIG.items():
        voice_path = find_voice_file(voice_name)
        if voice_path is None:
            print(f"! Voice not found: {voice_name}")
            continue
        try:
            VOICES[voice_name] = {
                "voice": PiperVoice.load(str(voice_path)),
                "config": config,
                "path": str(voice_path),
            }
            print(f"Loaded voice: {voice_name}")
        except Exception as exc:  # corrupt/incompatible model files
            print(f"Failed to load {voice_name}: {exc}")


def synthesize(voice: Any, text: str, wav_file: wave.Wave_write, length_scale: float) -> None:
    """Synthesize ``text`` into ``wav_file``; ``length_scale`` > 1 is slower speech."""
    try:
        from piper import SynthesisConfig
    except ImportError:  # older piper-tts without SynthesisConfig
        voice.synthesize_wav(text, wav_file)
        return
    voice.synthesize_wav(text, wav_file, syn_config=SynthesisConfig(length_scale=length_scale))


class PiperHandler(BaseHTTPRequestHandler):
    """HTTP handler for the Piper TTS API. Only the routes below are served."""

    cors_origin = "*"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", self.cors_origin)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path in ("/", "/index.html"):
            if not DEMO_PAGE.exists():
                self._send_json(404, {"error": "Demo page not found"})
                return
            body = DEMO_PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/voices":
            voices = [{"id": name, **data["config"]} for name, data in VOICES.items()]
            self._send_json(200, {"voices": voices})
        elif self.path == "/health":
            self._send_json(
                200,
                {"status": "ok", "piper_available": PIPER_AVAILABLE, "voices_loaded": len(VOICES)},
            )
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.path != "/synthesize":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(400 if length <= 0 else 413, {"error": "Invalid request body size"})
            return

        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "Invalid JSON"})
            return
        if not isinstance(data, dict):
            self._send_json(400, {"error": "Expected a JSON object"})
            return

        text = str(data.get("text", "")).strip()
        voice_id = data.get("voice", "en_US-lessac-medium")
        if not text:
            self._send_json(400, {"error": "No text provided"})
            return
        if voice_id not in VOICES:
            self._send_json(404, {"error": f"Voice '{voice_id}' not available"})
            return

        try:
            length_scale = min(max(float(data.get("length_scale", 1.0)), 0.25), 4.0)
        except (TypeError, ValueError):
            self._send_json(400, {"error": "length_scale must be a number"})
            return

        try:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                synthesize(VOICES[voice_id]["voice"], text, wav_file, length_scale)
            wav_data = wav_buffer.getvalue()
        except Exception as exc:
            self.log_error("Synthesis error: %s", exc)
            self._send_json(500, {"error": "Synthesis failed"})
            return

        if data.get("format") == "base64":
            self._send_json(
                200,
                {
                    "audio": base64.b64encode(wav_data).decode(),
                    "format": "wav",
                    "text": text,
                    "voice": voice_id,
                },
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_data)))
        self.send_header("Access-Control-Allow-Origin", self.cors_origin)
        self.send_header("Content-Disposition", "attachment; filename=speech.wav")
        self.end_headers()
        self.wfile.write(wav_data)

    def do_OPTIONS(self) -> None:  # noqa: N802 - http.server API
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self.cors_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Piper TTS demo server")
    parser.add_argument("--open", action="store_true", help="Open the demo page in a browser")
    args = parser.parse_args()

    settings = load_settings(default_port=8085)
    # Access-Control-Allow-Origin takes a single value; use it when exactly one is configured.
    origins = settings.cors_origins
    PiperHandler.cors_origin = origins[0] if len(origins) == 1 else "*"

    load_voices()
    if not VOICES:
        print("No voices loaded. Download some with:")
        print("   python3 -m piper.download_voices en_US-lessac-medium")
        print("   python3 -m piper.download_voices hi_IN-rohan-medium")

    server = ThreadingHTTPServer((settings.host, settings.port), PiperHandler)
    url = f"http://{settings.host}:{settings.port}"
    print(f"Piper TTS server running at {url} (Ctrl+C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
