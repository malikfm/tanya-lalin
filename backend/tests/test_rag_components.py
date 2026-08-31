"""Focused generation, validation, and reranking adapter tests."""

from __future__ import annotations

import pytest

from app.exceptions import ProviderUnavailableError
from app.infrastructure.ai.openai import SafeEmbeddingClient
from app.rag.generation import AnswerGenerator
from app.rag.models import ConversationMessage, ValidationResult
from app.rag.reranker import Reranker, RerankScores
from app.rag.validation import AnswerValidator


class FakeChain:
    def __init__(self, value):
        self.value = value
        self.payloads = []

    async def ainvoke(self, payload):
        self.payloads.append(payload)
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


async def test_generator_renders_evidence_history_and_repair_feedback(evidence):
    generator = object.__new__(AnswerGenerator)
    generator._chain = FakeChain("  Jawaban [S1].  ")
    result = await generator.generate(
        "Apa dendanya?",
        evidence,
        [ConversationMessage(role="user", content="Tentang lampu merah")],
        repair_reason="citation missing",
    )
    assert result.answer == "Jawaban [S1]."
    payload = generator._chain.payloads[0]
    assert "Pasal 287 ayat (2)" in payload["evidence"]
    assert "Pengguna: Tentang lampu merah" in payload["history"]
    assert "citation missing" in payload["query"]


async def test_generator_translates_provider_failure(evidence):
    generator = object.__new__(AnswerGenerator)
    generator._chain = FakeChain(RuntimeError("down"))
    with pytest.raises(ProviderUnavailableError):
        await generator.generate("test", evidence, [])


async def test_validator_supports_disabled_semantic_validation(evidence):
    validator = AnswerValidator(None)
    citations, result = await validator.validate("Jawaban [S1].", evidence)
    assert citations.valid is True
    assert result == ValidationResult(valid=True, score=5)

    citations, result = await validator.validate("Tanpa marker", evidence)
    assert citations.valid is False
    assert result.valid is False


async def test_validator_uses_semantic_result_and_fails_open_on_provider_error(evidence):
    validator = object.__new__(AnswerValidator)
    validator._chain = FakeChain(ValidationResult(valid=False, score=2, reason="unsupported claim"))
    _, result = await validator.validate("Jawaban [S1].", evidence)
    assert result.valid is False

    validator._chain = FakeChain(
        ValidationResult(valid=False, score=3, reason="non-material wording concern")
    )
    _, calibrated = await validator.validate("Jawaban [S1].", evidence)
    assert calibrated.valid is True

    validator._chain = FakeChain(RuntimeError("down"))
    _, fallback = await validator.validate("Jawaban [S1].", evidence)
    assert fallback.valid is True
    assert "unavailable" in fallback.reason


async def test_reranker_scores_and_falls_back(evidence):
    second = evidence[0].model_copy(deep=True)
    second.source_id = "S2"
    second.chunk.chunk_id = "second"
    items = [evidence[0], second]

    reranker = object.__new__(Reranker)
    reranker._chain = FakeChain(RerankScores(scores=[1.0, 9.0]))
    result = await reranker.rerank("query", items)
    assert [item.chunk.chunk_id for item in result.evidence] == [
        "second",
        evidence[0].chunk.chunk_id,
    ]
    assert [item.source_id for item in result.evidence] == ["S1", "S2"]
    assert result.failed is False

    reranker._chain = FakeChain(RerankScores(scores=[1.0]))
    fallback = await reranker.rerank("query", items)
    assert fallback.evidence == items
    assert fallback.failed is True
    assert (await reranker.rerank("query", items[:1])).evidence == items[:1]


async def test_reranker_uses_fusion_score_as_a_deterministic_tie_breaker(evidence):
    second = evidence[0].model_copy(deep=True)
    second.chunk.chunk_id = "higher-fusion"
    second.fusion_score = 0.2
    reranker = object.__new__(Reranker)
    reranker._chain = FakeChain(RerankScores(scores=[5.0, 5.0]))

    result = await reranker.rerank("query", [evidence[0], second])

    assert [item.chunk.chunk_id for item in result.evidence] == [
        "higher-fusion",
        evidence[0].chunk.chunk_id,
    ]


class FakeEmbeddingProvider:
    def __init__(self, value):
        self.value = value

    async def aembed_query(self, text):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value

    async def aembed_documents(self, texts):
        if isinstance(self.value, Exception):
            raise self.value
        return [self.value for _ in texts]


async def test_embedding_adapter_translates_provider_failures():
    client = SafeEmbeddingClient(FakeEmbeddingProvider([1.0, 0.0]))
    assert await client.aembed_query("test") == [1.0, 0.0]
    assert await client.aembed_documents(["one", "two"]) == [
        [1.0, 0.0],
        [1.0, 0.0],
    ]
    client = SafeEmbeddingClient(FakeEmbeddingProvider(RuntimeError("down")))
    with pytest.raises(ProviderUnavailableError):
        await client.aembed_query("test")
    with pytest.raises(ProviderUnavailableError):
        await client.aembed_documents(["test"])
