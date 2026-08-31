"""Public liveness and readiness routes."""

from fastapi import APIRouter, Depends

from app.api.schemas import LivenessResponse, ReadinessResponse
from app.dependencies import AppContainer, get_container
from app.exceptions import ProviderUnavailableError
from app.infrastructure.sessions import RedisSessionStore

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=LivenessResponse)
async def live() -> LivenessResponse:
    return LivenessResponse()


@router.get("/ready", response_model=ReadinessResponse)
async def ready(container: AppContainer = Depends(get_container)) -> ReadinessResponse:
    if not container.settings.openai_key:
        raise ProviderUnavailableError("The AI provider is not configured.")
    if not await container.sessions.ping():
        raise ProviderUnavailableError("The configured session store is unavailable.")
    return ReadinessResponse(
        status="ready",
        corpus_version=container.corpus.version,
        chunk_count=len(container.corpus.chunks),
        session_store=("redis" if isinstance(container.sessions, RedisSessionStore) else "memory"),
        provider_configured=True,
    )
