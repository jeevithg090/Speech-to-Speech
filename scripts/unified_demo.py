"""
Apollo Voice Engine - unified speech-to-speech demo (FastAPI).

Architecture::

    Audio In -> SNAC encode -> AudioLLM (extended Sarvam-1) -> SNAC decode -> Audio Out

Models load in a background thread on startup; until they are ready (or if
loading fails), requests are answered by an Edge-TTS fallback so the UI stays
usable. Note that an *untrained* AudioLLM produces arbitrary audio tokens:
meaningful speech output requires a checkpoint trained with
``scripts/train_audio_llm.py``.

Usage:
    pip install -e ".[speech]"
    python scripts/unified_demo.py     # http://127.0.0.1:6969
"""

from __future__ import annotations

import base64
import io
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402

from apollo_voice_engine.audio_io import (  # noqa: E402
    AudioDecodeError,
    decode_audio_bytes,
    float_to_wav_base64,
)
from apollo_voice_engine.dialogue import GREETINGS, normalize_language  # noqa: E402
from apollo_voice_engine.metrics import RunningMetrics  # noqa: E402
from apollo_voice_engine.models.audio_llm import AudioLLM  # noqa: E402
from apollo_voice_engine.models.snac_wrapper import SNACWrapper  # noqa: E402
from apollo_voice_engine.models.speaker_encoder import SpeakerEncoder  # noqa: E402
from apollo_voice_engine.safety.classifier import SafetyClassifier  # noqa: E402
from apollo_voice_engine.settings import load_settings  # noqa: E402

logger = logging.getLogger("unified_demo")

DEMO_DIR = PROJECT_ROOT / "demo"
settings = load_settings(default_port=6969)

SYSTEM_PROMPTS = {
    "hi": "आप अपोलो अस्पताल के एक सहायक हैं। रोगी की मदद करें।",
    "ta": "நீங்கள் அப்பல்லோ மருத்துவமனையின் உதவியாளர். நோயாளிக்கு உதவுங்கள்.",
    "te": "మీరు అపోలో ఆసుపత్రి సహాయకుడు. రోగికి సహాయం చేయండి.",
    "kn": "ನೀವು ಅಪೊಲೊ ಆಸ್ಪತ್ರೆಯ ಸಹಾಯಕ. ರೋಗಿಗೆ ಸಹಾಯ ಮಾಡಿ.",
    "en": "You are an Apollo Hospital assistant. Help the patient with their query.",
}

FALLBACK_VOICES = {
    "hi": "hi-IN-SwaraNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "kn": "kn-IN-SapnaNeural",
    "en": "en-IN-NeerjaNeural",
}


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class UnifiedConfig:
    """Configuration for the unified speech-to-speech engine."""

    sample_rate: int = 24000  # SNAC 24 kHz
    device: str = field(default_factory=_default_device)
    snac_model: str = "hubertsiuzdak/snac_24khz"
    llm_model: str = "sarvamai/sarvam-1"
    max_audio_duration: float = 10.0  # seconds of input audio kept
    min_audio_samples: int = 4800  # 0.2 s; shorter inputs are zero-padded
    max_response_tokens: int = 500
    temperature: float = 0.7


class UnifiedVoiceEngine:
    """
    Unified speech-to-speech engine: SNAC + AudioLLM + speaker conditioning.

    The first turn of a session computes a speaker embedding ("session
    anchor") that conditions every later turn until ``reset_anchor`` is set.
    """

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.device = config.device

        self.snac: Any = None
        self.audio_llm: Any = None
        self.speaker_encoder: Any = None
        self.safety_classifier = SafetyClassifier()

        self.anchor_embedding: torch.Tensor | None = None
        self.metrics = RunningMetrics()

        self.models_loaded = False
        self.load_error: str | None = None

    def load_models(self) -> None:
        """Load SNAC, AudioLLM and the speaker encoder. Errors switch to fallback mode."""
        try:
            logger.info("[1/3] Loading SNAC audio codec...")
            snac = SNACWrapper(
                model_name=self.config.snac_model,
                sample_rate=self.config.sample_rate,
                device=self.device,
            )
            snac.load_model()

            logger.info("[2/3] Loading AudioLLM (extended Sarvam-1)...")
            audio_llm = AudioLLM(
                model_name=self.config.llm_model,
                device=self.device,
                load_in_8bit=(self.device == "cuda"),
            )
            audio_llm.load_model()

            logger.info("[3/3] Initializing speaker encoder...")
            self.speaker_encoder = SpeakerEncoder(device=self.device)
            self.snac, self.audio_llm = snac, audio_llm
            self.models_loaded = True
            logger.info("All models loaded; engine ready")
        except Exception as exc:
            self.load_error = str(exc)
            logger.exception("Error loading models; running in fallback mode")

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    def prepare_audio(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert to a ``(1, 1, T)`` float tensor, truncated and padded to sane bounds."""
        tensor = torch.as_tensor(audio, dtype=torch.float32).reshape(-1)
        max_samples = int(self.config.max_audio_duration * self.config.sample_rate)
        tensor = tensor[:max_samples]
        if tensor.numel() < self.config.min_audio_samples:
            tensor = torch.nn.functional.pad(
                tensor, (0, self.config.min_audio_samples - tensor.numel())
            )
        return tensor.reshape(1, 1, -1)

    def encode_audio(self, audio: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, float]:
        """Encode audio to flat SNAC tokens. Returns ``(tokens, encode_ms)``."""
        start = time.perf_counter()
        tokens = self.snac.encode(self.prepare_audio(audio))
        return tokens, (time.perf_counter() - start) * 1000

    def decode_audio(self, tokens: torch.Tensor) -> tuple[np.ndarray, float]:
        """Decode SNAC tokens to a 1-D waveform. Returns ``(audio, decode_ms)``."""
        start = time.perf_counter()
        audio = self.snac.decode(tokens).detach().float().cpu().numpy().reshape(-1)
        return audio, (time.perf_counter() - start) * 1000

    def generate_response(
        self,
        input_tokens: torch.Tensor,
        language: str = "hi",
        speaker_embedding: torch.Tensor | None = None,
        override_output_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, float]:
        """
        Run the AudioLLM on the encoded input.

        Args:
            override_output_tokens: Pre-computed SNAC tokens to return instead
                of running generation (for demos and tests).

        Returns:
            ``(response_snac_tokens_or_None, generate_ms)``
        """
        start = time.perf_counter()
        if override_output_tokens is not None:
            return override_output_tokens, (time.perf_counter() - start) * 1000

        inputs_embeds, attention_mask = self.audio_llm.prepare_audio_input(
            input_tokens,
            text_prompt=SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["en"]),
            speaker_embedding=speaker_embedding,
        )
        output_ids = self.audio_llm.generate_response(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_response_tokens,
            temperature=self.config.temperature,
            do_sample=True,
        )
        response_tokens = self.audio_llm.extract_audio_tokens(output_ids)
        return response_tokens, (time.perf_counter() - start) * 1000

    async def process_speech(
        self,
        audio: np.ndarray,
        language: str = "hi",
        reset_anchor: bool = False,
        cached_response_tokens: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Full pipeline. Falls back to Edge-TTS if models are missing or a stage fails."""
        language = normalize_language(language)
        if not self.models_loaded:
            return await self._fallback_process(language, reason="models not loaded")

        total_start = time.perf_counter()
        stage_ms: dict[str, float] = {}
        try:
            input_tokens, stage_ms["snac_encode_ms"] = self.encode_audio(audio)

            if reset_anchor or self.anchor_embedding is None:
                self.anchor_embedding = self.speaker_encoder(audio)
                logger.info("New session anchor established")

            response_tokens, stage_ms["llm_generate_ms"] = self.generate_response(
                input_tokens,
                language=language,
                speaker_embedding=self.anchor_embedding,
                override_output_tokens=cached_response_tokens,
            )
            if response_tokens is None or response_tokens.shape[-1] < SNACWrapper.TOKENS_PER_FRAME:
                return await self._fallback_process(language, reason="no audio tokens generated")

            response_audio, stage_ms["snac_decode_ms"] = self.decode_audio(response_tokens)
        except Exception as exc:
            logger.exception("Unified pipeline failed")
            return await self._fallback_process(language, reason=f"pipeline error: {exc}")

        stage_ms["total_ms"] = (time.perf_counter() - total_start) * 1000
        self.metrics.increment("total_requests")
        for name, value in stage_ms.items():
            self.metrics.observe(name, value)

        return {
            "success": True,
            "audio": float_to_wav_base64(response_audio, self.config.sample_rate),
            "audio_format": "wav",
            # The unified model has no text head output wired up yet.
            "transcription": "[Audio response generated]",
            "response_text": "",
            "action": "RESPOND",
            "safety_level": "safe",
            "fallback_mode": False,
            "input_tokens": int(input_tokens.shape[-1]),
            "metrics": stage_ms,
        }

    async def _fallback_process(self, language: str, reason: str) -> dict[str, Any]:
        """Answer with a greeting synthesized by Edge-TTS (network service)."""
        start = time.perf_counter()
        response_text = GREETINGS.get(language, GREETINGS["en"])
        audio_b64 = ""
        error = None
        try:
            import edge_tts

            communicate = edge_tts.Communicate(
                response_text, FALLBACK_VOICES.get(language, FALLBACK_VOICES["en"])
            )
            buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buffer.write(chunk["data"])
            audio_b64 = base64.b64encode(buffer.getvalue()).decode()
        except Exception as exc:
            logger.warning("Fallback TTS unavailable: %s", exc)
            error = "Fallback text-to-speech is unavailable."

        self.metrics.increment("total_requests")
        self.metrics.increment("fallback_responses")
        return {
            "success": bool(audio_b64),
            "audio": audio_b64,
            "audio_format": "mp3",
            "transcription": f"[Fallback mode: {reason}]",
            "response_text": response_text,
            "action": "RESPOND",
            "safety_level": "safe",
            "fallback_mode": True,
            "error": error,
            "metrics": {"total_ms": (time.perf_counter() - start) * 1000},
        }


# ============================================================================
# FastAPI application
# ============================================================================

engine: UnifiedVoiceEngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    engine = UnifiedVoiceEngine(UnifiedConfig())
    threading.Thread(target=engine.load_models, daemon=True).start()
    yield


app = FastAPI(
    title="Apollo Voice Engine - Unified Architecture",
    description="Speech-to-speech using SNAC + AudioLLM",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return get_unified_demo_html()


@app.get("/api/health")
async def health() -> dict:
    loaded = bool(engine and engine.models_loaded)
    error = engine.load_error if engine else None
    return {
        "status": "healthy" if loaded else ("degraded" if error else "loading"),
        "models_loaded": loaded,
        "error": error,
        "architecture": "SNAC -> AudioLLM -> SNAC (unified)",
    }


@app.get("/api/metrics")
async def get_metrics() -> dict:
    return engine.metrics.snapshot() if engine else {}


@app.get("/api/architecture")
async def architecture() -> dict:
    llm = engine.audio_llm if engine and engine.models_loaded else None
    return {
        "name": "Apollo Omni-Indic Voice Engine",
        "architecture_type": "Unified Transformer",
        "components": {
            "audio_encoder": "SNAC 24kHz (multi-scale neural audio codec)",
            "language_model": "Extended Sarvam-1 2B (with audio token vocabulary)",
            "audio_decoder": "SNAC 24kHz",
            "speaker_conditioning": "Projected speaker embedding prefix (session anchor)",
        },
        "data_flow": "Audio -> SNAC encode -> LLM generate -> SNAC decode -> Audio",
        "token_vocabulary": {
            "audio_tokens": AudioLLM.AUDIO_VOCAB_SIZE,
            "snac_tokens_per_frame": SNACWrapper.TOKENS_PER_FRAME,
            "audio_token_offset": llm.audio_offset if llm else None,
        },
        "supported_languages": ["Hindi", "Tamil", "Telugu", "Kannada", "English"],
        "latency_target_ms": 500,
    }


@app.post("/api/speech-to-speech")
async def speech_to_speech(
    audio: UploadFile = File(...),
    language: str = Form("en"),
    reset_anchor: bool = Form(False),
) -> dict:
    """Audio flows through SNAC encode -> AudioLLM -> SNAC decode."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    audio_bytes = await audio.read(settings.max_upload_bytes + 1)
    if len(audio_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Audio upload is too large")
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    try:
        samples = decode_audio_bytes(audio_bytes, sample_rate=engine.config.sample_rate)
    except AudioDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = await engine.process_speech(samples, language, reset_anchor=reset_anchor)
    if engine.anchor_embedding is not None:
        digest = abs(int(torch.sum(engine.anchor_embedding).item() * 1000)) % 10000
        result["anchor_hash"] = str(digest).zfill(4)
        result["anchor_status"] = "New Identity" if reset_anchor else "Locked (Consistent)"
    return result


def get_unified_demo_html() -> str:
    """Return the demo web UI HTML."""
    return (DEMO_DIR / "unified_demo.html").read_text(encoding="utf-8")


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(f"Apollo Voice Engine unified demo on http://{settings.host}:{settings.port}")
    print("Models load in the background; check /api/health for status.")
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
