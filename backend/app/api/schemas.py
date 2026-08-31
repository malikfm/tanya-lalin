"""Public HTTP and SSE schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.application.models import ChatResult, StoredMessage


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: UUID | None = None

    @field_validator("message")
    @classmethod
    def _strip_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Message cannot be blank")
        return stripped


class ChatResponse(ChatResult):
    """Validated chat response returned by both chat transports."""


class SessionHistoryResponse(BaseModel):
    session_id: UUID
    messages: list[StoredMessage]
    created_at: datetime
    updated_at: datetime


class DeleteSessionResponse(BaseModel):
    session_id: UUID
    deleted: bool = True


class LivenessResponse(BaseModel):
    status: str = "alive"


class ReadinessResponse(BaseModel):
    status: str
    corpus_version: str
    chunk_count: int
    session_store: str
    provider_configured: bool


class ProblemDetails(BaseModel):
    type: str
    title: str
    status: int
    detail: str
    error_code: str
    request_id: str
