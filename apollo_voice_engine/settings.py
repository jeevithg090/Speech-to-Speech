"""
Runtime settings read from environment variables.

All demo servers share these so they can be configured the same way
(see ``.env.example``). A ``.env`` file in the working directory is loaded
if ``python-dotenv`` is installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # optional convenience
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional dependency
    pass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class ServerSettings:
    """Network settings for the demo HTTP servers."""

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    max_upload_bytes: int = 10 * 1024 * 1024
    max_text_chars: int = 2000


def load_settings(default_port: int = 8000) -> ServerSettings:
    """Build :class:`ServerSettings` from ``APOLLO_*`` environment variables."""
    return ServerSettings(
        host=os.getenv("APOLLO_HOST", "127.0.0.1"),
        port=_env_int("APOLLO_PORT", default_port),
        cors_origins=_env_list("APOLLO_CORS_ORIGINS", "*") or ["*"],
        max_upload_bytes=_env_int("APOLLO_MAX_UPLOAD_MB", 10) * 1024 * 1024,
        max_text_chars=_env_int("APOLLO_MAX_TEXT_CHARS", 2000),
    )
