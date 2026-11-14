"""
Apollo Voice Engine - text chat demo server (FastAPI).

Serves a browser UI plus a small JSON API that routes text queries through
the safety classifier and the rule-based dialogue policy. No ML models are
loaded, so it starts instantly on any machine.

Usage:
    python scripts/api_demo.py            # http://127.0.0.1:8000
    APOLLO_PORT=9000 python scripts/api_demo.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from apollo_voice_engine.dialogue import LANGUAGES, DialoguePolicy  # noqa: E402
from apollo_voice_engine.metrics import RunningMetrics  # noqa: E402
from apollo_voice_engine.safety.classifier import SafetyClassifier  # noqa: E402
from apollo_voice_engine.settings import load_settings  # noqa: E402
from scripts.web_ui import install_web_ui  # noqa: E402

DEMO_DIR = PROJECT_ROOT / "demo"
settings = load_settings(default_port=8000)

app = FastAPI(
    title="Apollo Omni-Indic Voice Engine",
    description="Text chat demo: safety classification + multilingual canned responses",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
install_web_ui(app)

classifier = SafetyClassifier()
policy = DialoguePolicy(classifier)
metrics = RunningMetrics()


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=settings.max_text_chars)
    language: str = "hi"


class ChatResponse(BaseModel):
    response: str
    action: str  # RESPOND or TRANSFER_TO_HUMAN
    language: str
    safety_level: str
    latency_ms: float


class ClassifyRequest(BaseModel):
    text: str = Field(..., max_length=settings.max_text_chars)


class ClassifyResponse(BaseModel):
    level: str
    keywords: list[str]
    should_transfer: bool
    message: str


def get_html_ui() -> str:
    """Return the demo web UI HTML."""
    return (DEMO_DIR / "chat_demo.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return get_html_ui()


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "mode": "rule-based (no model loaded)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/languages")
async def languages() -> dict:
    return {"languages": LANGUAGES}


@app.get("/api/metrics")
async def get_metrics() -> dict:
    """Measured counters and average server-side latency for this process."""
    data = metrics.snapshot()
    data.setdefault("total_requests", 0)
    data.setdefault("emergency_transfers", 0)
    data["languages_supported"] = len(LANGUAGES)
    return data


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    result = classifier.classify(request.text)
    return ClassifyResponse(
        level=result.level.value,
        keywords=result.triggered_keywords,
        should_transfer=result.should_transfer,
        message=result.message,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    turn = policy.respond(request.text, request.language)

    metrics.increment("total_requests")
    if turn.is_transfer:
        metrics.increment("emergency_transfers")
    latency_ms = (time.perf_counter() - start) * 1000
    metrics.observe("latency_ms", latency_ms)

    return ChatResponse(
        response=turn.response,
        action=turn.action.value,
        language=turn.language,
        safety_level=turn.safety_level,
        latency_ms=round(latency_ms, 3),
    )


def main() -> None:
    import uvicorn

    print(f"Apollo Voice Engine chat demo on http://{settings.host}:{settings.port}")
    print(f"API docs: http://{settings.host}:{settings.port}/docs")
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
