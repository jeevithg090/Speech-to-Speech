"""
Shared web-UI plumbing for the FastAPI demo servers.

* ``/assets/*`` serves the design system (CSS, JS, favicon) from ``demo/assets``.
* ``/favicon.ico`` serves the SVG favicon for browsers that ask for it.
* Unknown routes return the friendly ``demo/404.html`` page to browsers and the
  usual JSON ``{"detail": "Not Found"}`` to API clients.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "demo"
ASSETS_DIR = DEMO_DIR / "assets"
NOT_FOUND_PAGE = DEMO_DIR / "404.html"


def wants_html(request: Request) -> bool:
    """True for browser page loads (not API calls, not asset fetches)."""
    if request.url.path.startswith(("/api/", "/assets/")):
        return False
    return "text/html" in request.headers.get("accept", "")


def install_web_ui(app: FastAPI) -> None:
    """Mount shared assets, a favicon and an HTML 404 page on ``app``."""
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(ASSETS_DIR / "favicon.svg", media_type="image/svg+xml")

    @app.exception_handler(StarletteHTTPException)
    async def not_found_page(request: Request, exc: StarletteHTTPException) -> Response:
        if exc.status_code == 404 and wants_html(request) and NOT_FOUND_PAGE.exists():
            return HTMLResponse(NOT_FOUND_PAGE.read_text(encoding="utf-8"), status_code=404)
        return await http_exception_handler(request, exc)
