"""Typed internal models for the RAG pipeline."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class PipelineStage(StrEnum):
    VALIDATING = "validating"
    ANALYZING = "analyzing"
    RETRIEVING = "retrieving"
    RERANKING = "reranking"
    GENERATING = "generating"
    VERIFYING = "verifying"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    BLOCKED = "blocked"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class RefusalReason(StrEnum):
    """Internal reason for a safe non-answer outcome."""

    PROMPT_INJECTION = "prompt_injection"
    OFF_TOPIC = "off_topic"
    EXTERNAL_SOURCES_REQUIRED = "external_sources_required"
    EMPTY_RETRIEVAL = "empty_retrieval"
    VALIDATION_FAILED = "validation_failed"


class CorpusChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    source: str
    article_number: int
    paragraph_number: int | None = None
    chunk_type: Literal["body", "elucidation"]
    text: str
    official_url: str
    last_verified_at: date


class SearchHit(BaseModel):
    chunk: CorpusChunk
    search_name: str
    rank: int
    vector_relevance: float | None = None
    bm25_score: float | None = None


class RankedListTrace(BaseModel):
    """One isolated retriever result list and its contribution to fusion."""

    search_name: str
    family: Literal["vector", "bm25", "article"]
    query: str
    weight: float
    hits: list[SearchHit] = Field(default_factory=list)


class Evidence(BaseModel):
    source_id: str
    chunk: CorpusChunk
    fusion_score: float
    vector_relevance: float | None = None
    bm25_rank: int | None = None
    reranker_score: float | None = None


class RetrievalResult(BaseModel):
    """Candidates plus diagnostic data produced by one retrieval execution."""

    ranked_lists: list[RankedListTrace] = Field(default_factory=list)
    fused_candidates: list[Evidence] = Field(default_factory=list)
    candidates: list[Evidence] = Field(default_factory=list)


class RerankResult(BaseModel):
    """Reranked evidence and operational metadata for one provider call."""

    evidence: list[Evidence] = Field(default_factory=list)
    failed: bool = False
    latency_seconds: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None


class QueryAnalysis(BaseModel):
    standalone_query: str
    is_in_scope: bool = True
    legal_query: str
    key_phrases: list[str] = Field(default_factory=list)
    requires_external_sources: bool = False
    external_source_reason: str = ""


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class GeneratedAnswer(BaseModel):
    answer: str


class ValidationResult(BaseModel):
    valid: bool
    score: int = Field(default=5, ge=1, le=5)
    reason: str = ""
    unsupported_claims: list[str] = Field(default_factory=list)


class PipelineCitation(BaseModel):
    citation_id: str
    marker: str
    source_id: str


class PipelineSource(BaseModel):
    source_id: str
    chunk: CorpusChunk


class PipelineResult(BaseModel):
    status: AnswerStatus
    answer: str
    citations: list[PipelineCitation] = Field(default_factory=list)
    sources: list[PipelineSource] = Field(default_factory=list)
    evidence_count: int = 0
    validation_score: int | None = None
    analysis: QueryAnalysis | None = Field(default=None, exclude=True)
    evidence: list[Evidence] = Field(default_factory=list, exclude=True)
    retrieval: RetrievalResult | None = Field(default=None, exclude=True)
    refusal_reason: RefusalReason | None = Field(default=None, exclude=True)


class StageEvent(BaseModel):
    kind: Literal["status"] = "status"
    stage: PipelineStage


class ResultEvent(BaseModel):
    kind: Literal["result"] = "result"
    result: PipelineResult


PipelineEvent = StageEvent | ResultEvent
