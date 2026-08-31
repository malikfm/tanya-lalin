"""RFC 7807-style problem response handlers."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi.errors import RateLimitExceeded

from app.api.schemas import ProblemDetails
from app.exceptions import ApplicationError


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def problem_payload(
    request: Request,
    *,
    status: int,
    title: str,
    detail: str,
    error_code: str,
) -> dict[str, Any]:
    return ProblemDetails(
        type=f"https://tanya-lalin.dev/problems/{error_code}",
        title=title,
        status=status,
        detail=detail,
        error_code=error_code,
        request_id=_request_id(request),
    ).model_dump(mode="json")


async def application_error_handler(request: Request, exc: ApplicationError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=problem_payload(
            request,
            status=exc.status_code,
            title=exc.title,
            detail=exc.detail,
            error_code=exc.error_code,
        ),
        media_type="application/problem+json",
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=problem_payload(
            request,
            status=422,
            title="Invalid request",
            detail="The request payload or path parameters are invalid.",
            error_code="invalid_request",
        ),
        media_type="application/problem+json",
    )


async def rate_limit_error_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content=problem_payload(
            request,
            status=429,
            title="Too many requests",
            detail="Terlalu banyak permintaan. Silakan coba lagi nanti.",
            error_code="rate_limited",
        ),
        media_type="application/problem+json",
        headers={"Retry-After": "60"},
    )


async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.opt(exception=exc).error("Unhandled request failure")
    return JSONResponse(
        status_code=500,
        content=problem_payload(
            request,
            status=500,
            title="Internal server error",
            detail="Terjadi kesalahan saat memproses permintaan.",
            error_code="internal_error",
        ),
        media_type="application/problem+json",
    )
