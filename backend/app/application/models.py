"""Application-level chat contracts shared by API and persistence adapters."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.rag.models import AnswerStatus, PipelineStage


def utc_now() -> datetime:
    return datetime.now(UTC)


class Citation(BaseModel):
    citation_id: str
    marker: str
    source_id: str


class Source(BaseModel):
    source_id: str
    document_id: str
    document_title: str
    article_number: int
    paragraph_number: int | None = None
    chunk_type: Literal["body", "elucidation"]
    excerpt: str
    official_url: str
    last_verified_at: date


class ChatResult(BaseModel):
    request_id: UUID
    session_id: UUID
    message_id: UUID
    status: AnswerStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class StoredMessage(BaseModel):
    id: UUID
    role: Literal["user", "assistant"]
    content: str
    status: AnswerStatus | None = None
    citations: list[Citation] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class ChatSession(BaseModel):
    id: UUID
    messages: list[StoredMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MetaEvent(BaseModel):
    kind: Literal["meta"] = "meta"
    request_id: UUID
    session_id: UUID


class StatusEvent(BaseModel):
    kind: Literal["status"] = "status"
    stage: PipelineStage


class CompleteEvent(BaseModel):
    kind: Literal["complete"] = "complete"
    result: ChatResult


ApplicationEvent = MetaEvent | StatusEvent | CompleteEvent
