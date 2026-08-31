"""Session-aware chat use case built on one pipeline event stream."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import suppress
from uuid import UUID, uuid4

from loguru import logger

from app.application.models import (
    ApplicationEvent,
    ChatResult,
    Citation,
    CompleteEvent,
    MetaEvent,
    Source,
    StatusEvent,
    StoredMessage,
)
from app.config import Settings
from app.infrastructure.sessions import SessionStore
from app.rag.models import ConversationMessage, ResultEvent, StageEvent
from app.rag.pipeline import RAGPipeline


class ChatApplicationService:
    """Coordinate sessions, RAG execution, persistence, and operational logs."""

    def __init__(
        self,
        *,
        pipeline: RAGPipeline,
        sessions: SessionStore,
        settings: Settings,
        corpus_version: str,
    ):
        self.pipeline = pipeline
        self.sessions = sessions
        self.settings = settings
        self.corpus_version = corpus_version

    @staticmethod
    def _operational_log(event: str, **context: object) -> None:
        """Emit audit metadata without allowing a logging sink to fail the request."""

        with suppress(Exception):
            logger.bind(audit=True, **context).info(event)

    async def stream_chat(
        self,
        *,
        message: str,
        session_id: UUID | None = None,
        request_id: UUID | None = None,
    ) -> AsyncIterator[ApplicationEvent]:
        started = time.monotonic()
        stage_started = started
        current_stage: str | None = None
        request_id = request_id or uuid4()
        session = await self.sessions.get_or_create(session_id)
        yield MetaEvent(request_id=request_id, session_id=session.id)

        history = [
            ConversationMessage(role=item.role, content=item.content)
            for item in session.messages[-self.settings.max_context_messages :]
        ]
        async for event in self.pipeline.stream(message, history):
            if isinstance(event, StageEvent):
                now = time.monotonic()
                if current_stage is not None:
                    self._operational_log(
                        "chat_stage_completed",
                        request_id=str(request_id),
                        session_id=str(session.id),
                        stage=current_stage,
                        latency_seconds=round(now - stage_started, 3),
                    )
                current_stage = event.stage.value
                stage_started = now
                yield StatusEvent(stage=event.stage)
                continue
            if not isinstance(event, ResultEvent):
                continue

            message_id = uuid4()
            result = ChatResult(
                request_id=request_id,
                session_id=session.id,
                message_id=message_id,
                status=event.result.status,
                answer=event.result.answer,
                citations=[
                    Citation.model_validate(item.model_dump()) for item in event.result.citations
                ],
                sources=[
                    Source(
                        source_id=item.source_id,
                        document_id=item.chunk.document_id,
                        document_title=item.chunk.document_title,
                        article_number=item.chunk.article_number,
                        paragraph_number=item.chunk.paragraph_number,
                        chunk_type=item.chunk.chunk_type,
                        excerpt=item.chunk.text,
                        official_url=item.chunk.official_url,
                        last_verified_at=item.chunk.last_verified_at,
                    )
                    for item in event.result.sources
                ],
            )
            session.messages.extend(
                [
                    StoredMessage(id=uuid4(), role="user", content=message),
                    StoredMessage(
                        id=message_id,
                        role="assistant",
                        content=result.answer,
                        status=result.status,
                        citations=result.citations,
                        sources=result.sources,
                        created_at=result.created_at,
                    ),
                ]
            )
            await self.sessions.save(session)
            if current_stage is not None:
                self._operational_log(
                    "chat_stage_completed",
                    request_id=str(request_id),
                    session_id=str(session.id),
                    stage=current_stage,
                    latency_seconds=round(time.monotonic() - stage_started, 3),
                )
            self._operational_log(
                "chat_completed",
                request_id=str(request_id),
                session_id=str(session.id),
                outcome=result.status.value,
                evidence_count=event.result.evidence_count,
                citation_count=len(result.citations),
                validation_score=event.result.validation_score,
                refusal_reason=(
                    event.result.refusal_reason.value if event.result.refusal_reason else None
                ),
                latency_seconds=round(time.monotonic() - started, 3),
                model=self.settings.chat_model,
                corpus_version=self.corpus_version,
            )
            yield CompleteEvent(result=result)
            return

        raise RuntimeError("RAG pipeline completed without a result event")

    async def chat(
        self,
        *,
        message: str,
        session_id: UUID | None = None,
        request_id: UUID | None = None,
    ) -> ChatResult:
        """Consume the canonical event stream and return its final result."""

        async for event in self.stream_chat(
            message=message,
            session_id=session_id,
            request_id=request_id,
        ):
            if isinstance(event, CompleteEvent):
                return event.result
        raise RuntimeError("Chat stream completed without a response")
