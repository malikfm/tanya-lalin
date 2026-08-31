"""Session-store contracts and application-service persistence tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.application.chat_service import ChatApplicationService
from app.application.models import CompleteEvent, StoredMessage
from app.config import Settings
from app.infrastructure.sessions import (
    InMemorySessionStore,
    RedisSessionStore,
    create_session_store,
)
from app.rag.models import (
    AnswerStatus,
    PipelineCitation,
    PipelineResult,
    PipelineSource,
    PipelineStage,
    ResultEvent,
    StageEvent,
)


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.closed = False

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex):
        self.values[key] = value

    async def delete(self, key):
        return int(self.values.pop(key, None) is not None)

    async def ping(self):
        return True

    async def aclose(self):
        self.closed = True


async def exercise_store(store):
    session = await store.get_or_create(None)
    assert await store.get(session.id) is None
    session.messages.append(StoredMessage(id=uuid4(), role="user", content="test"))
    await store.save(session)
    restored = await store.get(session.id)
    assert restored is not None
    assert restored.messages[0].content == "test"
    assert await store.delete(session.id) is True
    assert await store.delete(session.id) is False


async def test_in_memory_store_contract_and_factory():
    store = create_session_store("", ttl_seconds=60, max_messages=4)
    assert isinstance(store, InMemorySessionStore)
    await exercise_store(store)


async def test_redis_store_contract(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr("app.infrastructure.sessions.Redis.from_url", lambda *a, **k: fake)
    store = RedisSessionStore("redis://test", ttl_seconds=60, max_messages=4)
    await exercise_store(store)
    assert await store.ping() is True
    await store.close()
    assert fake.closed is True


async def test_session_ttl_cap_and_copy_isolation():
    store = InMemorySessionStore(ttl_seconds=60, max_messages=2)
    session = await store.get_or_create(None)
    session.messages = [
        StoredMessage(id=uuid4(), role="user", content=str(index)) for index in range(3)
    ]
    await store.save(session)
    restored = await store.get(session.id)
    assert [item.content for item in restored.messages] == ["1", "2"]
    restored.messages.clear()
    assert len((await store.get(session.id)).messages) == 2

    store._sessions[session.id].updated_at = datetime.now(UTC) - timedelta(seconds=61)
    assert await store.get(session.id) is None


class SuccessfulPipeline:
    def __init__(self, chunk):
        self.chunk = chunk
        self.histories = []

    async def stream(self, message, history):
        self.histories.append(history)
        yield StageEvent(stage=PipelineStage.VALIDATING)
        yield ResultEvent(
            result=PipelineResult(
                status=AnswerStatus.ANSWERED,
                answer="Denda paling banyak Rp500.000 [S1].",
                citations=[PipelineCitation(citation_id="C1", marker="[S1]", source_id="S1")],
                sources=[PipelineSource(source_id="S1", chunk=self.chunk)],
                evidence_count=1,
                validation_score=5,
            )
        )


async def test_service_reuses_canonical_stream_and_persists_complete_response(chunk):
    sessions = InMemorySessionStore(ttl_seconds=60, max_messages=10)
    pipeline = SuccessfulPipeline(chunk)
    service = ChatApplicationService(
        pipeline=pipeline,
        sessions=sessions,
        settings=Settings(environment="test"),
        corpus_version="test-v1",
    )
    request_id = uuid4()
    result = await service.chat(message="Apa dendanya?", request_id=request_id)
    assert result.request_id == request_id
    assert len(result.citations) == 1

    session = await sessions.get(result.session_id)
    assert [item.role for item in session.messages] == ["user", "assistant"]
    assert session.messages[-1].sources == result.sources

    events = [
        event
        async for event in service.stream_chat(
            message="Kalau diulangi?",
            session_id=result.session_id,
        )
    ]
    assert isinstance(events[-1], CompleteEvent)
    assert len(pipeline.histories[-1]) == 2


async def test_service_rejects_pipeline_without_result(chunk):
    class EmptyPipeline:
        async def stream(self, message, history):
            if False:
                yield

    service = ChatApplicationService(
        pipeline=EmptyPipeline(),
        sessions=InMemorySessionStore(ttl_seconds=60, max_messages=10),
        settings=Settings(environment="test"),
        corpus_version="test",
    )
    with pytest.raises(RuntimeError, match="without a result"):
        await service.chat(message="test")
