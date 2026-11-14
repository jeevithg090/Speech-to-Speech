"""Tests for the kiosk session server and its FastAPI app."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from apollo_voice_engine.integration import (
    KioskSession,
    SessionLimitError,
    WebRTCConfig,
    WebRTCServer,
    create_fastapi_app,
)


def run(coro):
    return asyncio.run(coro)


def test_kiosk_session_buffer_and_history():
    session = KioskSession("abc")
    session.add_audio_chunk(b"ab")
    session.add_audio_chunk(b"cd")
    assert session.get_audio_buffer() == b"abcd"
    assert session.get_audio_buffer() == b""
    session.add_to_history("user", "hello")
    assert session.transcript_history == [{"role": "user", "text": "hello"}]
    session.close()
    assert not session.is_active


def test_config_defaults_are_not_shared():
    a, b = WebRTCConfig(), WebRTCConfig()
    a.stun_servers.append("stun:example.org")
    assert b.stun_servers == ["stun:stun.l.google.com:19302"]


def test_offer_creates_session_and_callbacks():
    events = []

    async def on_start(session):
        events.append(("start", session.session_id))

    async def on_audio(session, chunk):
        events.append(("audio", len(chunk)))

    async def on_end(session):
        events.append(("end", session.session_id))

    server = WebRTCServer(
        WebRTCConfig(), on_audio_received=on_audio, on_session_start=on_start, on_session_end=on_end
    )
    answer = json.loads(run(server.handle_offer("sdp", "s1")))
    assert answer["type"] == "answer"
    run(server.handle_audio_stream("s1", b"1234"))
    run(server.handle_audio_stream("missing", b"1234"))  # ignored
    assert server.get_active_sessions() == 1
    run(server.close_session("s1"))
    assert server.get_active_sessions() == 0
    assert events == [("start", "s1"), ("audio", 4), ("end", "s1")]


def test_offer_generates_session_id():
    server = WebRTCServer(WebRTCConfig())
    run(server.handle_offer("sdp"))
    assert len(server.sessions) == 1
    assert len(next(iter(server.sessions))) == 32


def test_session_limit_enforced():
    server = WebRTCServer(WebRTCConfig(max_connections=1))
    run(server.handle_offer("sdp", "a"))
    with pytest.raises(SessionLimitError):
        run(server.handle_offer("sdp", "b"))
    run(server.handle_offer("sdp", "a"))  # re-offer for an existing session is allowed


def test_transfer_to_human_includes_history():
    server = WebRTCServer(WebRTCConfig())
    run(server.handle_offer("sdp", "s"))
    server.sessions["s"].add_to_history("user", "chest pain")
    event = run(server.transfer_to_human("s", "emergency detected"))
    assert event["reason"] == "emergency detected"
    assert event["conversation_history"][0]["text"] == "chest pain"
    assert run(server.transfer_to_human("unknown", "x")) is None


class TestFastAPIApp:
    def test_health_offer_and_websocket(self):
        server = WebRTCServer(WebRTCConfig())
        client = TestClient(create_fastapi_app(server))
        assert client.get("/health").json() == {"status": "healthy", "active_sessions": 0}

        session_id = client.post("/offer", json={"session_id": "k1", "sdp": "x"}).json()[
            "session_id"
        ]
        assert session_id == "k1"
        received = []

        async def on_audio(session, chunk):
            received.append(chunk)

        server.on_audio_received = on_audio
        with client.websocket_connect("/ws/k1") as ws:
            ws.send_bytes(b"\x00\x01")
        assert received == [b"\x00\x01"]
        assert "k1" not in server.sessions  # closed on disconnect

    def test_websocket_rejects_unknown_session(self):
        client = TestClient(create_fastapi_app(WebRTCServer(WebRTCConfig())))
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/nope") as ws:
            ws.receive_bytes()

    def test_offer_returns_503_when_full(self):
        client = TestClient(create_fastapi_app(WebRTCServer(WebRTCConfig(max_connections=1))))
        assert client.post("/offer", json={}).status_code == 200
        assert client.post("/offer", json={}).status_code == 503
