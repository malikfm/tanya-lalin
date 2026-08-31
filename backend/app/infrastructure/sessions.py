"""Async in-memory and Redis chat session stores."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from redis.asyncio import Redis

from app.application.models import ChatSession


class SessionStore(ABC):
    """Storage contract for bounded chat sessions."""

    def __init__(self, *, ttl_seconds: int, max_messages: int):
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages

    @abstractmethod
    async def get(self, session_id: UUID) -> ChatSession | None: ...

    @abstractmethod
    async def save(self, session: ChatSession) -> None: ...

    @abstractmethod
    async def delete(self, session_id: UUID) -> bool: ...

    async def get_or_create(self, session_id: UUID | None) -> ChatSession:
        if session_id is not None:
            existing = await self.get(session_id)
            if existing is not None:
                return existing
        return ChatSession(id=uuid4())

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    def prepare(self, session: ChatSession) -> ChatSession:
        session.messages = session.messages[-self.max_messages :]
        session.updated_at = datetime.now(UTC)
        return session


class InMemorySessionStore(SessionStore):
    """Single-process session store with TTL and bounded history."""

    def __init__(self, *, ttl_seconds: int, max_messages: int):
        super().__init__(ttl_seconds=ttl_seconds, max_messages=max_messages)
        self._sessions: dict[UUID, ChatSession] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: UUID) -> ChatSession | None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if datetime.now(UTC) - session.updated_at > timedelta(seconds=self.ttl_seconds):
                self._sessions.pop(session_id, None)
                return None
            session.updated_at = datetime.now(UTC)
            return session.model_copy(deep=True)

    async def save(self, session: ChatSession) -> None:
        async with self._lock:
            prepared = self.prepare(session.model_copy(deep=True))
            self._sessions[prepared.id] = prepared

    async def delete(self, session_id: UUID) -> bool:
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None


class RedisSessionStore(SessionStore):
    """Redis-backed sessions with explicit availability semantics."""

    KEY_PREFIX = "tanya-lalin:session:"

    def __init__(
        self,
        redis_url: str,
        *,
        ttl_seconds: int,
        max_messages: int,
    ):
        super().__init__(ttl_seconds=ttl_seconds, max_messages=max_messages)
        self._redis = Redis.from_url(redis_url, decode_responses=True)

    def _key(self, session_id: UUID) -> str:
        return f"{self.KEY_PREFIX}{session_id}"

    async def get(self, session_id: UUID) -> ChatSession | None:
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            return None
        try:
            session = ChatSession.model_validate_json(raw)
        except Exception:
            await self._redis.delete(self._key(session_id))
            return None
        session.updated_at = datetime.now(UTC)
        await self.save(session)
        return session

    async def save(self, session: ChatSession) -> None:
        prepared = self.prepare(session.model_copy(deep=True))
        await self._redis.set(
            self._key(prepared.id),
            prepared.model_dump_json(),
            ex=self.ttl_seconds,
        )

    async def delete(self, session_id: UUID) -> bool:
        return bool(await self._redis.delete(self._key(session_id)))

    async def ping(self) -> bool:
        return bool(await self._redis.ping())

    async def close(self) -> None:
        await self._redis.aclose()


def create_session_store(
    redis_url: str,
    *,
    ttl_seconds: int,
    max_messages: int,
) -> SessionStore:
    """Choose Redis only when explicitly configured."""

    if redis_url:
        return RedisSessionStore(
            redis_url,
            ttl_seconds=ttl_seconds,
            max_messages=max_messages,
        )
    return InMemorySessionStore(
        ttl_seconds=ttl_seconds,
        max_messages=max_messages,
    )
