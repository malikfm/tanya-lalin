"""Hybrid retrieval, tracing, fusion, and selection tests."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.rag.models import Evidence, QueryAnalysis, SearchHit
from app.rag.retrieval import HybridRetriever, RetrievalProfile


class FakeEmbeddings:
    def __init__(self):
        self.queries: list[str] = []

    async def aembed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


class FakeIndex:
    def __init__(self, chunk):
        self.chunk = chunk
        self.vector_calls: list[str] = []
        self.bm25_queries: list[str] = []
        self.article_queries: list[int] = []

    def vector_search(self, embedding, *, search_name, top_k, min_relevance):
        self.vector_calls.append(search_name)
        return [
            SearchHit(
                chunk=self.chunk,
                search_name=search_name,
                rank=1,
                vector_relevance=0.8,
            ),
            SearchHit(
                chunk=self.chunk,
                search_name=search_name,
                rank=2,
                vector_relevance=0.7,
            ),
        ]

    def bm25_search(self, query, top_k):
        self.bm25_queries.append(query)
        return [SearchHit(chunk=self.chunk, search_name="bm25", rank=1, bm25_score=4.0)]

    def article_search(self, article_number, top_k):
        self.article_queries.append(article_number)
        return [
            SearchHit(
                chunk=self.chunk.model_copy(update={"article_number": article_number}),
                search_name=f"article_{article_number}",
                rank=1,
            )
        ]


async def test_balanced_retrieval_keeps_lists_isolated_and_normalizes_families(chunk):
    index = FakeIndex(chunk)
    embeddings = FakeEmbeddings()
    settings = Settings(_env_file=None, environment="test", retrieval_profile="balanced")
    retriever = HybridRetriever(index, embeddings, settings)
    analysis = QueryAnalysis(
        standalone_query="lampu merah",
        legal_query="Alat Pemberi Isyarat Lalu Lintas",
        key_phrases=["APILL"],
    )
    result = await retriever.retrieve(analysis)

    assert index.vector_calls == ["vector_legal", "vector_original", "vector_phrase_1"]
    assert embeddings.queries == [analysis.legal_query, analysis.standalone_query, "APILL"]
    assert index.bm25_queries == [analysis.standalone_query, analysis.legal_query]
    assert [trace.family for trace in result.ranked_lists] == [
        "vector",
        "vector",
        "vector",
        "bm25",
        "bm25",
    ]
    vector_weight = sum(trace.weight for trace in result.ranked_lists if trace.family == "vector")
    bm25_weight = sum(trace.weight for trace in result.ranked_lists if trace.family == "bm25")
    assert vector_weight == pytest.approx(1.0)
    assert bm25_weight == pytest.approx(1.0)
    assert len(result.candidates) == 1
    assert result.candidates[0].vector_relevance == 0.8
    assert result.candidates[0].bm25_rank == 1
    assert result.candidates[0].fusion_score == pytest.approx(2.0 / 61)


async def test_legacy_profile_preserves_previous_weighting(chunk):
    index = FakeIndex(chunk)
    retriever = HybridRetriever(
        index,
        FakeEmbeddings(),
        Settings(_env_file=None, environment="test"),
        RetrievalProfile.LEGACY,
    )
    result = await retriever.retrieve(
        QueryAnalysis(
            standalone_query="lampu merah",
            legal_query="APILL",
            key_phrases=["isyarat"],
        )
    )

    assert [trace.weight for trace in result.ranked_lists] == [2.0, 1.0, 0.8, 0.7]
    assert result.candidates[0].fusion_score == pytest.approx(4.5 / 61)


async def test_explicit_article_reference_adds_deterministic_ranked_list(chunk):
    index = FakeIndex(chunk)
    retriever = HybridRetriever(
        index,
        FakeEmbeddings(),
        Settings(_env_file=None, environment="test"),
    )
    result = await retriever.retrieve(
        QueryAnalysis(
            standalone_query="Menurut Pasal 229 dan Pasal 48, apa aturannya? UU 22/2009",
            legal_query="kecelakaan dan kendaraan",
        )
    )

    assert index.article_queries == [229, 48]
    article_traces = [trace for trace in result.ranked_lists if trace.family == "article"]
    assert [trace.weight for trace in article_traces] == [0.75, 0.75]
    assert HybridRetriever.extract_article_references("UU 22/2009 tanpa rujukan") == []


def test_deduplication_is_per_ranked_list(chunk):
    hits = [
        SearchHit(chunk=chunk, search_name="a", rank=1),
        SearchHit(chunk=chunk, search_name="a", rank=2),
    ]
    assert HybridRetriever._deduplicate(hits) == hits[:1]


def test_soft_diversity_reserves_three_unique_articles(chunk):
    evidence = [
        Evidence(
            source_id=f"S{index}",
            chunk=chunk.model_copy(
                update={"chunk_id": f"chunk-{index}", "article_number": article}
            ),
            fusion_score=1 / index,
        )
        for index, article in enumerate([10, 10, 10, 20, 30], 1)
    ]

    selected = HybridRetriever._soft_diversity(evidence)

    assert [item.chunk.article_number for item in selected[:3]] == [10, 20, 30]
    assert [item.chunk.article_number for item in selected[3:]] == [10, 10]
