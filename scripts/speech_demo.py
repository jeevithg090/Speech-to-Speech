"""
Apollo Voice Engine - cascaded speech-to-speech demo (FastAPI).

Pipeline:
- STT: OpenAI Whisper (``openai-whisper`` package, "small" model by default)
- Dialogue: safety classifier + rule-based multilingual responses
- TTS: Microsoft Edge-TTS (network service)

Requires the ``speech`` extra and ``ffmpeg`` on PATH::

    pip install -e ".[speech]"
    python scripts/speech_demo.py     # http://127.0.0.1:8000
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from apollo_voice_engine.audio_io import AudioDecodeError, convert_to_wav  # noqa: E402
from apollo_voice_engine.dialogue import (  # noqa: E402
    LANGUAGES,
    DialoguePolicy,
    normalize_language,
)
from apollo_voice_engine.metrics import RunningMetrics  # noqa: E402
from apollo_voice_engine.settings import load_settings  # noqa: E402
from scripts.web_ui import install_web_ui  # noqa: E402

logger = logging.getLogger("speech_demo")

DEMO_DIR = PROJECT_ROOT / "demo"
WHISPER_MODEL_NAME = os.getenv("APOLLO_WHISPER_MODEL", "small")
MIN_AUDIO_BYTES = 1000

settings = load_settings(default_port=8000)

app = FastAPI(
    title="Apollo Omni-Indic Voice Engine",
    description="Speech-to-speech demo with Whisper + Edge-TTS",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
install_web_ui(app)

policy = DialoguePolicy()
metrics = RunningMetrics()

# Edge-TTS voices (female) for each supported language
VOICE_MAP = {
    "hi": "hi-IN-SwaraNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "kn": "kn-IN-SapnaNeural",
    "en": "en-IN-NeerjaNeural",
}

_whisper_model: Any = None
_whisper_lock = threading.Lock()


def get_whisper_model() -> Any:
    """Lazily load the Whisper model (thread-safe)."""
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            import whisper

            logger.info("Loading Whisper model '%s'...", WHISPER_MODEL_NAME)
            _whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    return _whisper_model


async def synthesize_speech(text: str, lang: str) -> bytes:
    """Synthesize ``text`` to MP3 bytes with Edge-TTS."""
    import edge_tts

    communicate = edge_tts.Communicate(text, VOICE_MAP.get(lang, VOICE_MAP["en"]))
    buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.write(chunk["data"])
    return buffer.getvalue()


def get_html_ui() -> str:
    """Return the demo web UI HTML."""
    return (DEMO_DIR / "speech_demo.html").read_text(encoding="utf-8")


async def _respond_with_audio(text: str, language: str) -> dict[str, Any]:
    llm_start = time.perf_counter()
    turn = policy.respond(text, language)
    llm_ms = (time.perf_counter() - llm_start) * 1000

    metrics.increment("total_requests")
    if turn.is_transfer:
        metrics.increment("emergency_transfers")

    tts_start = time.perf_counter()
    try:
        audio_b64 = base64.b64encode(await synthesize_speech(turn.response, turn.language)).decode()
        tts_error = None
    except Exception as exc:  # network service; degrade to text-only
        logger.warning("TTS failed: %s", exc)
        audio_b64, tts_error = "", "Text-to-speech is unavailable right now."
    tts_ms = (time.perf_counter() - tts_start) * 1000

    return {
        "response": turn.response,
        "audio": audio_b64,
        "tts_error": tts_error,
        "action": turn.action.value,
        "language": turn.language,
        "safety_level": turn.safety_level,
        "llm_ms": llm_ms,
        "tts_ms": tts_ms,
    }


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return get_html_ui()


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "stt": f"OpenAI Whisper ({WHISPER_MODEL_NAME})",
        "stt_loaded": _whisper_model is not None,
        "tts": "Microsoft Edge-TTS",
        "languages": list(VOICE_MAP),
    }


@app.get("/api/languages")
async def languages() -> dict:
    return {"languages": [{**lang, "voice": VOICE_MAP[lang["code"]]} for lang in LANGUAGES]}


@app.get("/api/metrics")
async def get_metrics() -> dict:
    data = metrics.snapshot()
    data.setdefault("total_requests", 0)
    data.setdefault("emergency_transfers", 0)
    return data


class TextChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=settings.max_text_chars)
    language: str = "en"


@app.post("/api/chat")
async def chat_text(request: TextChatRequest) -> dict:
    """Text in -> text + synthesized audio out."""
    start = time.perf_counter()
    result = await _respond_with_audio(request.text, request.language)
    total_ms = (time.perf_counter() - start) * 1000
    metrics.observe("tts_ms", result["tts_ms"])
    metrics.observe("total_ms", total_ms)
    return {"transcription": request.text, **result, "latency_ms": total_ms}


@app.post("/api/speech-to-speech")
async def speech_to_speech(audio: UploadFile = File(...), language: str = Form("en")) -> dict:
    """Recorded audio in -> transcription, response text and audio out."""
    start = time.perf_counter()
    language = normalize_language(language)

    audio_bytes = await audio.read(settings.max_upload_bytes + 1)
    if len(audio_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Audio upload is too large")
    if len(audio_bytes) < MIN_AUDIO_BYTES:
        raise HTTPException(status_code=400, detail="Recording was too short. Please speak longer.")

    with tempfile.TemporaryDirectory(prefix="apollo_") as tmp:
        src = os.path.join(tmp, "input")
        wav = os.path.join(tmp, "input.wav")
        Path(src).write_bytes(audio_bytes)
        try:
            convert_to_wav(src, wav)
        except AudioDecodeError as exc:
            logger.warning("Audio conversion failed: %s", exc)
            raise HTTPException(
                status_code=422, detail="Could not decode audio (is ffmpeg installed?)"
            ) from exc

        stt_start = time.perf_counter()
        try:
            model = get_whisper_model()
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail='Speech recognition is not installed. Run: pip install -e ".[speech]"',
            ) from exc
        except Exception as exc:  # download / load failure
            logger.exception("Whisper model failed to load")
            raise HTTPException(
                status_code=503,
                detail=f"The Whisper '{WHISPER_MODEL_NAME}' model could not be loaded. "
                "Check the server log and your network connection.",
            ) from exc
        result = model.transcribe(wav, language=language, task="transcribe", fp16=False)
        stt_ms = (time.perf_counter() - stt_start) * 1000

    transcription = result["text"].strip()
    if not transcription:
        raise HTTPException(status_code=422, detail="No speech detected in the recording")

    reply = await _respond_with_audio(transcription, language)
    total_ms = (time.perf_counter() - start) * 1000
    for name, value in (
        ("stt_ms", stt_ms),
        ("llm_ms", reply["llm_ms"]),
        ("tts_ms", reply["tts_ms"]),
        ("total_ms", total_ms),
    ):
        metrics.observe(name, value)

    return {"transcription": transcription, **reply, "stt_ms": stt_ms, "total_ms": total_ms}


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(f"Apollo Voice Engine speech demo on http://{settings.host}:{settings.port}")
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
