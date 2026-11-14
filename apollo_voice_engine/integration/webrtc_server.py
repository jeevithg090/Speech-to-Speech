"""
Apollo Voice Engine - WebRTC Server for Kiosk Deployment

Real-time voice streaming interface for hospital kiosks.
"""

import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from apollo_voice_engine.settings import load_settings

logger = logging.getLogger(__name__)


@dataclass
class WebRTCConfig:
    """WebRTC server configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    stun_servers: list[str] = field(default_factory=lambda: ["stun:stun.l.google.com:19302"])
    max_connections: int = 25


class SessionLimitError(RuntimeError):
    """Raised when the server is already serving ``max_connections`` sessions."""


class KioskSession:
    """Represents a single kiosk voice session."""

    def __init__(self, session_id: str, language: str = "hi"):
        self.session_id = session_id
        self.language = language
        self.is_active = True
        self.audio_buffer = []
        self.transcript_history = []

    def add_audio_chunk(self, chunk: bytes):
        """Add incoming audio chunk to buffer."""
        self.audio_buffer.append(chunk)

    def get_audio_buffer(self) -> bytes:
        """Get and clear the audio buffer."""
        data = b"".join(self.audio_buffer)
        self.audio_buffer.clear()
        return data

    def add_to_history(self, role: str, text: str):
        """Add exchange to conversation history."""
        self.transcript_history.append({"role": role, "text": text})

    def close(self):
        """Close the session."""
        self.is_active = False


class WebRTCServer:
    """
    WebRTC server for Apollo Voice kiosks.

    Handles real-time audio streaming between kiosk tablets
    and the voice inference engine.
    """

    def __init__(
        self,
        config: WebRTCConfig,
        on_audio_received: Callable[[KioskSession, bytes], Awaitable[None]] | None = None,
        on_session_start: Callable[[KioskSession], Awaitable[None]] | None = None,
        on_session_end: Callable[[KioskSession], Awaitable[None]] | None = None,
    ):
        self.config = config
        self.sessions: dict[str, KioskSession] = {}
        self.on_audio_received = on_audio_received
        self.on_session_start = on_session_start
        self.on_session_end = on_session_end

    async def handle_offer(self, offer_sdp: str, session_id: str | None = None) -> str:
        """
        Handle WebRTC offer from client.

        Args:
            offer_sdp: SDP offer from client
            session_id: Unique session identifier (generated when empty)

        Returns:
            JSON-encoded answer for the client

        Raises:
            SessionLimitError: If ``max_connections`` sessions are already active.
        """
        session_id = session_id or uuid.uuid4().hex
        if session_id not in self.sessions and (
            self.get_active_sessions() >= self.config.max_connections
        ):
            raise SessionLimitError(f"Maximum of {self.config.max_connections} sessions reached")

        session = KioskSession(session_id)
        self.sessions[session_id] = session

        if self.on_session_start:
            await self.on_session_start(session)

        logger.info(f"New kiosk session: {session_id}")

        # In production, this would create actual WebRTC peer connection
        # and return proper SDP answer
        answer_sdp = self._create_mock_answer(offer_sdp)

        return answer_sdp

    def _create_mock_answer(self, offer: str) -> str:
        """Create mock SDP answer (placeholder for actual WebRTC)."""
        return json.dumps(
            {
                "type": "answer",
                "sdp": "mock_answer_sdp",
                "ice_servers": self.config.stun_servers,
            }
        )

    async def handle_audio_stream(self, session_id: str, audio_chunk: bytes):
        """
        Process incoming audio chunk from kiosk.

        Args:
            session_id: Session identifier
            audio_chunk: Raw audio bytes (24kHz, 16-bit PCM)
        """
        if session_id not in self.sessions:
            logger.warning(f"Unknown session: {session_id}")
            return

        session = self.sessions[session_id]
        session.add_audio_chunk(audio_chunk)

        if self.on_audio_received:
            await self.on_audio_received(session, audio_chunk)

    async def send_audio_response(self, session_id: str, audio_data: bytes):
        """
        Send audio response back to kiosk.

        Args:
            session_id: Session identifier
            audio_data: Response audio bytes
        """
        if session_id not in self.sessions:
            return

        # In production, this would stream through WebRTC data channel
        logger.info(f"Sending {len(audio_data)} bytes to session {session_id}")

    async def transfer_to_human(self, session_id: str, reason: str) -> dict[str, Any] | None:
        """
        Transfer session to human operator via SIP.

        Args:
            session_id: Session identifier
            reason: Reason for transfer (e.g., "emergency detected")
        """
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        logger.warning(f"HUMAN TRANSFER: {session_id} - {reason}")

        # Log the transfer event
        transfer_event = {
            "event": "human_transfer",
            "session_id": session_id,
            "reason": reason,
            "conversation_history": session.transcript_history,
        }

        # In production: Initiate SIP call to operator
        # sip_client.dial(OPERATOR_NUMBER, transfer_event)

        return transfer_event

    async def close_session(self, session_id: str):
        """Close a kiosk session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.close()

            if self.on_session_end:
                await self.on_session_end(session)

            del self.sessions[session_id]
            logger.info(f"Session closed: {session_id}")

    def get_active_sessions(self) -> int:
        """Get count of active sessions."""
        return len([s for s in self.sessions.values() if s.is_active])


def create_fastapi_app(server: WebRTCServer | None = None):
    """Create a FastAPI app exposing the kiosk signalling and audio endpoints."""
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        logger.error("FastAPI not installed. Run: pip install fastapi uvicorn")
        return None

    app = FastAPI(title="Apollo Voice Kiosk API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=load_settings().cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    server = server or WebRTCServer(WebRTCConfig())
    app.state.kiosk_server = server

    @app.get("/health")
    async def health():
        return {"status": "healthy", "active_sessions": server.get_active_sessions()}

    @app.post("/offer")
    async def create_offer(offer: dict):
        session_id = offer.get("session_id") or uuid.uuid4().hex
        try:
            answer = await server.handle_offer(offer.get("sdp", ""), session_id)
        except SessionLimitError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"session_id": session_id, "answer": answer}

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        if session_id not in server.sessions:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_bytes()
                await server.handle_audio_stream(session_id, data)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected: %s", session_id)
        except Exception:
            logger.exception("WebSocket error in session %s", session_id)
        finally:
            await server.close_session(session_id)

    return app
