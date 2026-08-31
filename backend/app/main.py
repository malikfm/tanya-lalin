"""FastAPI application factory for the API and bundled web client."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.errors import (
    application_error_handler,
    problem_payload,
    rate_limit_error_handler,
    unexpected_error_handler,
    validation_error_handler,
)
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.config import Settings, get_settings
from app.dependencies import AppContainer
from app.exceptions import ApplicationError
from app.infrastructure.logging import configure_logging
from app.infrastructure.rate_limit import configure_rate_limits, limiter

API_PREFIX = "/api/v2"
APP_VERSION = "2.0.0"


def create_app(
    settings: Settings | None = None,
    container: AppContainer | None = None,
) -> FastAPI:
    """Create an application with optional injected configuration and dependencies."""

    runtime_settings = settings or get_settings()
    configure_logging(runtime_settings)
    configure_rate_limits(runtime_settings)
    owns_container = container is None

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        runtime_container = container or await AppContainer.build(runtime_settings)
        application.state.container = runtime_container
        logger.bind(
            corpus_version=runtime_container.corpus.version,
            chunk_count=len(runtime_container.corpus.chunks),
        ).info("application_started")
        try:
            yield
        finally:
            if owns_container:
                await runtime_container.close()
            logger.info("application_stopped")

    application = FastAPI(
        title="Tanya Lalin API",
        description="Evidence-grounded Indonesian traffic-law question answering.",
        version=APP_VERSION,
        lifespan=lifespan,
    )
    if container is not None:
        application.state.container = container
    application.state.limiter = limiter
    application.add_exception_handler(ApplicationError, application_error_handler)
    application.add_exception_handler(RequestValidationError, validation_error_handler)
    application.add_exception_handler(RateLimitExceeded, rate_limit_error_handler)
    application.add_exception_handler(Exception, unexpected_error_handler)
    application.add_middleware(SlowAPIMiddleware)
    application.add_middleware(GZipMiddleware, minimum_size=1000)

    if runtime_settings.parsed_cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=runtime_settings.parsed_cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "X-Request-ID"],
        )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id
        with logger.contextualize(request_id=request_id):
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        if request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif response.headers.get("content-type", "").startswith("text/html"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    application.include_router(chat_router, prefix=API_PREFIX)
    application.include_router(health_router, prefix=API_PREFIX)

    @application.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def unknown_api(request: Request, path: str) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=problem_payload(
                request,
                status=404,
                title="Not found",
                detail="The requested API endpoint does not exist.",
                error_code="not_found",
            ),
            media_type="application/problem+json",
        )

    static_dir = Path(runtime_settings.static_dir)
    if static_dir.is_dir() and (static_dir / "index.html").is_file():
        assets_dir = static_dir / "assets"
        if assets_dir.is_dir():
            application.mount(
                "/assets",
                StaticFiles(directory=assets_dir),
                name="assets",
            )

        @application.get("/{path:path}", include_in_schema=False)
        async def spa(path: str) -> FileResponse:
            requested = (static_dir / path).resolve()
            if requested.is_relative_to(static_dir) and requested.is_file():
                return FileResponse(requested)
            return FileResponse(static_dir / "index.html", media_type="text/html")
    else:

        @application.get("/", include_in_schema=False)
        async def api_root() -> dict[str, str]:
            return {
                "name": "Tanya Lalin API",
                "version": APP_VERSION,
                "docs": "/docs",
            }

    return application
