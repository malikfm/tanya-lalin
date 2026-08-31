"""Chat, streaming, and session history routes."""

import json
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from app.api.errors import problem_payload
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteSessionResponse,
    SessionHistoryResponse,
)
from app.application.chat_service import ChatApplicationService
from app.application.models import CompleteEvent, MetaEvent, StatusEvent
from app.dependencies import get_chat_service, get_session_store
from app.exceptions import ApplicationError, SessionNotFoundError
from app.infrastructure.rate_limit import chat_limit, delete_limit, history_limit, limiter
from app.infrastructure.sessions import SessionStore

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
@limiter.limit(chat_limit)
async def chat(
    request: Request,
    payload: ChatRequest,
    service: ChatApplicationService = Depends(get_chat_service),
) -> ChatResponse:
    result = await service.chat(
        message=payload.message,
        session_id=payload.session_id,
        request_id=UUID(request.state.request_id),
    )
    return ChatResponse.model_validate(result.model_dump())


@router.post("/stream")
@limiter.limit(chat_limit)
async def stream_chat(
    request: Request,
    payload: ChatRequest,
    service: ChatApplicationService = Depends(get_chat_service),
) -> EventSourceResponse:
    async def events():
        try:
            async for event in service.stream_chat(
                message=payload.message,
                session_id=payload.session_id,
                request_id=UUID(request.state.request_id),
            ):
                if isinstance(event, MetaEvent):
                    yield {"event": "meta", "data": event.model_dump_json()}
                elif isinstance(event, StatusEvent):
                    yield {
                        "event": "status",
                        "data": json.dumps({"stage": event.stage.value}),
                    }
                elif isinstance(event, CompleteEvent):
                    yield {"event": "complete", "data": event.result.model_dump_json()}
        except ApplicationError as exc:
            yield {
                "event": "error",
                "data": json.dumps(
                    problem_payload(
                        request,
                        status=exc.status_code,
                        title=exc.title,
                        detail=exc.detail,
                        error_code=exc.error_code,
                    )
                ),
            }
        except Exception as exc:
            logger.opt(exception=exc).error("Unhandled streaming failure")
            yield {
                "event": "error",
                "data": json.dumps(
                    problem_payload(
                        request,
                        status=500,
                        title="Internal server error",
                        detail="Terjadi kesalahan saat memproses permintaan.",
                        error_code="internal_error",
                    )
                ),
            }

    return EventSourceResponse(events())


@router.get("/{session_id}/history", response_model=SessionHistoryResponse)
@limiter.limit(history_limit)
async def history(
    request: Request,
    session_id: UUID,
    sessions: SessionStore = Depends(get_session_store),
) -> SessionHistoryResponse:
    session = await sessions.get(session_id)
    if session is None:
        raise SessionNotFoundError("Session does not exist or has expired.")
    return SessionHistoryResponse(
        session_id=session.id,
        messages=session.messages,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.delete("/{session_id}", response_model=DeleteSessionResponse)
@limiter.limit(delete_limit)
async def delete_session(
    request: Request,
    session_id: UUID,
    sessions: SessionStore = Depends(get_session_store),
) -> DeleteSessionResponse:
    if not await sessions.delete(session_id):
        raise SessionNotFoundError("Session does not exist or has expired.")
    return DeleteSessionResponse(session_id=session_id)
