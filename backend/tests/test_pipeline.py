"""Explicit RAG pipeline behavior and bounded-generation tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.config import Settings
from app.exceptions import ProviderUnavailableError
from app.rag.citations import validate_citations
from app.rag.models import (
    AnswerStatus,
    GeneratedAnswer,
    QueryAnalysis,
    RerankResult,
    ResultEvent,
    RetrievalResult,
    ValidationResult,
)
from app.rag.pipeline import RAGPipeline
from app.rag.query_analysis import PromptInjectionDetected


class FakeAnalyzer:
    def __init__(self, *, in_scope=True, injection=False, external=False):
        self.in_scope = in_scope
        self.injection = injection
        self.external = external
        self.history = None

    def validate_input(self, query):
        if self.injection:
            raise PromptInjectionDetected()

    async def analyze(self, query, history):
        self.history = history
        return QueryAnalysis(
            standalone_query=query,
            is_in_scope=self.in_scope,
            legal_query=query,
            requires_external_sources=self.external,
        )


class FakeRetriever:
    def __init__(self, evidence):
        self.evidence = evidence
        self.calls = 0

    async def retrieve(self, analysis):
        self.calls += 1
        return RetrievalResult(
            fused_candidates=self.evidence,
            candidates=self.evidence,
        )


class FakeGenerator:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = 0
        self.repair_reasons = []

    async def generate(self, query, evidence, history, repair_reason=None):
        self.calls += 1
        self.repair_reasons.append(repair_reason)
        value = next(self.answers)
        if isinstance(value, Exception):
            raise value
        return GeneratedAnswer(answer=value)


class FakeValidator:
    def __init__(self, verdicts=None):
        self.verdicts = iter(verdicts or [])

    async def validate(self, answer, evidence):
        citations = validate_citations(answer, evidence)
        try:
            valid = next(self.verdicts)
        except StopIteration:
            valid = citations.valid
        return citations, ValidationResult(
            valid=valid and citations.valid,
            score=5 if valid else 1,
            reason="unsupported" if not valid else "",
        )


@dataclass
class FakeReranker:
    calls: int = 0

    async def rerank(self, query, evidence):
        self.calls += 1
        evidence[0].reranker_score = 9.0
        return RerankResult(evidence=evidence)


async def collect(pipeline):
    return [event async for event in pipeline.stream("Apa dendanya?", [])]


def make_pipeline(evidence, generator, *, analyzer=None, validator=None, reranker=None):
    return RAGPipeline(
        settings=Settings(environment="test", final_evidence_top_k=5),
        analyzer=analyzer or FakeAnalyzer(),
        retriever=FakeRetriever(evidence),
        generator=generator,
        validator=validator or FakeValidator(),
        reranker=reranker,
    )


async def test_normal_success_generates_once_and_returns_cited_sources(evidence):
    generator = FakeGenerator(["Denda paling banyak Rp500.000 [S1]."])
    events = await collect(make_pipeline(evidence, generator))
    result = events[-1]
    assert isinstance(result, ResultEvent)
    assert result.result.status is AnswerStatus.ANSWERED
    assert generator.calls == 1
    assert result.result.analysis is not None
    assert result.result.evidence == evidence
    assert [source.source_id for source in result.result.sources] == ["S1"]
    assert [event.stage.value for event in events[:-1]] == [
        "validating",
        "analyzing",
        "retrieving",
        "generating",
        "verifying",
    ]


async def test_validation_failure_permits_one_repair(evidence):
    generator = FakeGenerator(["Draft tanpa sumber", "Denda berlaku [S1]."])
    events = await collect(make_pipeline(evidence, generator))
    assert events[-1].result.status is AnswerStatus.ANSWERED
    assert generator.calls == 2
    assert generator.repair_reasons == [None, "unsupported"]


async def test_failed_repair_never_exposes_rejected_answer(evidence):
    generator = FakeGenerator(["Draft salah", "Masih salah"])
    events = await collect(make_pipeline(evidence, generator))
    result = events[-1].result
    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert "Draft" not in result.answer
    assert generator.calls == 2


@pytest.mark.parametrize(
    ("analyzer", "retrieved", "status", "reason", "retrieval_calls"),
    [
        (
            FakeAnalyzer(injection=True),
            [object()],
            AnswerStatus.BLOCKED,
            "prompt_injection",
            0,
        ),
        (FakeAnalyzer(in_scope=False), [object()], AnswerStatus.BLOCKED, "off_topic", 0),
        (
            FakeAnalyzer(external=True),
            [object()],
            AnswerStatus.INSUFFICIENT_EVIDENCE,
            "external_sources_required",
            0,
        ),
        (FakeAnalyzer(), [], AnswerStatus.INSUFFICIENT_EVIDENCE, "empty_retrieval", 1),
    ],
)
async def test_safe_non_answer_outcomes(analyzer, retrieved, status, reason, retrieval_calls):
    generator = FakeGenerator([])
    pipeline = make_pipeline(retrieved, generator, analyzer=analyzer)
    events = await collect(pipeline)
    assert events[-1].result.status is status
    assert events[-1].result.refusal_reason.value == reason
    assert generator.calls == 0
    assert pipeline.retriever.calls == retrieval_calls
    assert "refusal_reason" not in events[-1].result.model_dump()


async def test_reranking_is_optional_and_observable(evidence):
    reranker = FakeReranker()
    pipeline = make_pipeline(
        evidence,
        FakeGenerator(["Jawaban [S1]."]),
        reranker=reranker,
    )
    events = await collect(pipeline)
    assert reranker.calls == 1
    assert "reranking" in [getattr(event, "stage", None) for event in events]


async def test_provider_failure_is_explicit(evidence):
    pipeline = make_pipeline(
        evidence,
        FakeGenerator([ProviderUnavailableError("unavailable")]),
    )
    with pytest.raises(ProviderUnavailableError):
        await collect(pipeline)
