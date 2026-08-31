"""HTTP, SSE, health, error, and rate-limit contract tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import UUID

from fastapi.testclient import TestClient

from app.application.chat_service import ChatApplicationService
from app.config import Settings
from app.dependencies import AppContainer
from app.infrastructure.rate_limit import limiter
from app.infrastructure.sessions import InMemorySessionStore
from app.main import create_app
from app.rag.models import (
    AnswerStatus,
    PipelineCitation,
    PipelineResult,
    PipelineSource,
    PipelineStage,
    ResultEvent,
    StageEvent,
)


class ApiPipeline:
    def __init__(self, chunk, *, fail=False):
        self.chunk = chunk
        self.fail = fail

    async def stream(self, message, history):
        yield StageEvent(stage=PipelineStage.VALIDATING)
        yield StageEvent(stage=PipelineStage.ANALYZING)
        if self.fail:
            raise RuntimeError("boom")
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


def make_client(chunk, *, fail=False, key="test-key", chat_limit="1000/minute"):
    settings = Settings(
        environment="test",
        openai_api_key=key,
        chat_rate_limit=chat_limit,
        history_rate_limit="1000/minute",
        delete_rate_limit="1000/minute",
    )
    sessions = InMemorySessionStore(ttl_seconds=60, max_messages=10)
    corpus = SimpleNamespace(version="test-v1", chunks=[chunk])
    service = ChatApplicationService(
        pipeline=ApiPipeline(chunk, fail=fail),
        sessions=sessions,
        settings=settings,
        corpus_version=corpus.version,
    )
    container = AppContainer(settings=settings, corpus=corpus, sessions=sessions, chat=service)
    limiter.reset()
    return TestClient(create_app(settings, container)), sessions


def parse_sse(response_text):
    events = []
    current = {}
    for line in response_text.splitlines():
        if not line:
            if current:
                events.append(current)
                current = {}
        elif line.startswith("event: "):
            current["event"] = line[7:]
        elif line.startswith("data: "):
            current["data"] = json.loads(line[6:])
    if current:
        events.append(current)
    return events


def test_sync_sse_history_and_delete_share_one_contract(chunk):
    client, _ = make_client(chunk)
    sync = client.post("/api/v1/chat", json={"message": "Apa dendanya?"})
    assert sync.status_code == 200
    result = sync.json()
    assert result["request_id"] == sync.headers["X-Request-ID"]
    assert result["sources"][0]["source_id"] == "S1"

    stream = client.post("/api/v1/chat/stream", json={"message": "Apa dendanya?"})
    assert stream.status_code == 200
    events = parse_sse(stream.text)
    assert [event["event"] for event in events] == [
        "meta",
        "status",
        "status",
        "complete",
    ]
    assert all(event["event"] != "token" for event in events)
    assert events[-1]["data"]["answer"] == result["answer"]

    session_id = result["session_id"]
    history = client.get(f"/api/v1/chat/{session_id}/history")
    assert history.status_code == 200
    assert history.json()["messages"][-1]["sources"] == result["sources"]
    assert client.delete(f"/api/v1/chat/{session_id}").status_code == 200
    missing = client.get(f"/api/v1/chat/{session_id}/history")
    assert missing.status_code == 404
    assert missing.headers["content-type"].startswith("application/problem+json")


def test_public_health_and_api_not_found(chunk):
    client, _ = make_client(chunk)
    assert client.get("/api/v1/health/live").json() == {"status": "alive"}
    ready = client.get("/api/v1/health/ready")
    assert ready.status_code == 200
    assert ready.json()["corpus_version"] == "test-v1"
    missing = client.get("/api/not-real")
    assert missing.status_code == 404
    assert missing.json()["error_code"] == "not_found"


def test_readiness_and_validation_errors_are_problem_details(chunk):
    client, _ = make_client(chunk, key="")
    unavailable = client.get("/api/v1/health/ready")
    assert unavailable.status_code == 503
    assert unavailable.json()["error_code"] == "provider_unavailable"

    invalid = client.post("/api/v1/chat", json={"message": "   "})
    assert invalid.status_code == 422
    assert invalid.json()["request_id"] == invalid.headers["X-Request-ID"]
    assert invalid.json()["error_code"] == "invalid_request"
    assert client.get("/api/v1/chat/not-a-uuid/history").status_code == 422


def test_stream_reports_post_open_failure_as_error_event(chunk):
    client, _ = make_client(chunk, fail=True)
    response = client.post("/api/v1/chat/stream", json={"message": "test"})
    events = parse_sse(response.text)
    assert [event["event"] for event in events][-1] == "error"
    assert events[-1]["data"]["error_code"] == "internal_error"


def test_chat_rate_limit_uses_trusted_client_address(chunk):
    client, _ = make_client(chunk, chat_limit="1/minute")
    assert client.post("/api/v1/chat", json={"message": "first"}).status_code == 200
    limited = client.post("/api/v1/chat", json={"message": "second"})
    assert limited.status_code == 429
    assert limited.json()["error_code"] == "rate_limited"
    assert UUID(limited.json()["request_id"])
