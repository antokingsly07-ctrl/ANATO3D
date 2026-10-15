"""FastAPI application factory.

The app exposes only the JSON API under ``/api``.  If a production frontend
build exists (``frontend/dist``) it is mounted at ``/`` so a single process can
serve both.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import api_router
from .api.deps import job_manager
from .config import get_settings
from .core.errors import AnatoError
from .core.logging_ import configure_logging, get_logger
from .core.storage import store

logger = get_logger("main")

_REAPER_STOP: threading.Event | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    configure_logging("DEBUG" if settings.debug else "INFO")
    logger.info(
        "%s %s starting (data dir: %s)",
        settings.app_name,
        settings.app_version,
        settings.work_root,
    )
    _start_reaper(settings.reaper_interval_seconds)
    try:
        yield
    finally:
        if _REAPER_STOP is not None:
            _REAPER_STOP.set()
        logger.info("Shutting down")


def _start_reaper(interval_seconds: int) -> None:
    global _REAPER_STOP
    if _REAPER_STOP is not None:
        return
    stop = threading.Event()
    _REAPER_STOP = stop

    def loop() -> None:
        while not stop.wait(max(30, interval_seconds)):
            try:
                removed = store.reap()
                pruned = job_manager.prune()
                if removed or pruned:
                    logger.info("Reaper: removed %d workspace(s), pruned %d job(s)", removed, pruned)
            except Exception:  # noqa: BLE001 - the reaper must never crash the app
                logger.exception("Background reaper iteration failed")

    thread = threading.Thread(target=loop, name="anato3d-reaper", daemon=True)
    thread.start()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging("DEBUG" if settings.debug else "INFO")

    app = FastAPI(
        title=f"{settings.app_name} API",
        version=settings.app_version,
        description=(
            "Patient-specific 3D anatomical reconstruction from DICOM imaging data. "
            "Research prototype -- NOT a certified medical device."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Slice-Index", "X-Slice-Count", "Content-Disposition"],
    )

    _install_exception_handlers(app)

    app.include_router(api_router)

    _mount_frontend(app)
    return app


def _install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AnatoError)
    async def handle_anato_error(request: Request, exc: AnatoError) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:  # noqa: ARG001
        problems = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            problems.append(f"{location or 'payload'}: {error.get('msg', 'invalid value')}")
        return JSONResponse(
            status_code=422,
            content={
                "code": "invalid_request",
                "message": "The request payload was rejected by schema validation.",
                "detail": "; ".join(problems)[:1000] or None,
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
        # Full details go to the server log only -- never to the client.
        logger.exception("Unhandled error while serving %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": (
                    "An unexpected internal error occurred. The operation was aborted and "
                    "no result was produced."
                ),
            },
        )


def _mount_frontend(app: FastAPI) -> None:
    candidates = [
        Path(__file__).resolve().parents[2] / "frontend" / "dist",
        Path.cwd() / "frontend" / "dist",
    ]
    for dist in candidates:
        if dist.is_dir() and (dist / "index.html").exists():
            app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
            logger.info("Serving frontend build from %s", dist)
            return


app = create_app()

__all__ = ["app", "create_app"]
