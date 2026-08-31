"""Explicit, typed RAG orchestration with one generation path."""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.config import Settings
from app.rag.generation import AnswerGenerator
from app.rag.models import (
    AnswerStatus,
    ConversationMessage,
    PipelineEvent,
    PipelineResult,
    PipelineStage,
    RefusalReason,
    ResultEvent,
    StageEvent,
)
from app.rag.query_analysis import PromptInjectionDetected, QueryAnalyzer
from app.rag.reranker import Reranker
from app.rag.retrieval import HybridRetriever
from app.rag.validation import AnswerValidator

BLOCKED_MESSAGE = (
    "Maaf, pertanyaan tersebut tidak dapat diproses. "
    "Silakan ajukan pertanyaan yang valid tentang aturan lalu lintas Indonesia."
)
OFF_TOPIC_MESSAGE = (
    "Maaf, pertanyaan tersebut berada di luar cakupan aturan lalu lintas dan "
    "angkutan jalan Indonesia."
)
INSUFFICIENT_MESSAGE = (
    "Maaf, sumber hukum yang tersedia belum cukup untuk memberikan jawaban "
    "yang dapat diverifikasi. "
    "Silakan ajukan pertanyaan yang lebih spesifik."
)


class RAGPipeline:
    """Run the complete legal-answer workflow and yield observable progress."""

    def __init__(
        self,
        *,
        settings: Settings,
        analyzer: QueryAnalyzer,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        validator: AnswerValidator,
        reranker: Reranker | None = None,
    ):
        self.settings = settings
        self.analyzer = analyzer
        self.retriever = retriever
        self.generator = generator
        self.validator = validator
        self.reranker = reranker

    async def stream(
        self,
        query: str,
        history: list[ConversationMessage],
    ) -> AsyncIterator[PipelineEvent]:
        yield StageEvent(stage=PipelineStage.VALIDATING)
        try:
            self.analyzer.validate_input(query)
        except PromptInjectionDetected:
            yield ResultEvent(
                result=PipelineResult(
                    status=AnswerStatus.BLOCKED,
                    answer=BLOCKED_MESSAGE,
                    refusal_reason=RefusalReason.PROMPT_INJECTION,
                )
            )
            return

        yield StageEvent(stage=PipelineStage.ANALYZING)
        analysis = await self.analyzer.analyze(query, history)
        if not analysis.is_in_scope:
            yield ResultEvent(
                result=PipelineResult(
                    status=AnswerStatus.BLOCKED,
                    answer=OFF_TOPIC_MESSAGE,
                    analysis=analysis,
                    refusal_reason=RefusalReason.OFF_TOPIC,
                )
            )
            return

        if analysis.requires_external_sources:
            yield ResultEvent(
                result=PipelineResult(
                    status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                    answer=INSUFFICIENT_MESSAGE,
                    analysis=analysis,
                    refusal_reason=RefusalReason.EXTERNAL_SOURCES_REQUIRED,
                )
            )
            return

        yield StageEvent(stage=PipelineStage.RETRIEVING)
        retrieval = await self.retriever.retrieve(analysis)
        evidence = retrieval.candidates
        if not evidence:
            yield ResultEvent(
                result=PipelineResult(
                    status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                    answer=INSUFFICIENT_MESSAGE,
                    analysis=analysis,
                    retrieval=retrieval,
                    refusal_reason=RefusalReason.EMPTY_RETRIEVAL,
                )
            )
            return

        if self.reranker is not None:
            yield StageEvent(stage=PipelineStage.RERANKING)
            rerank_result = await self.reranker.rerank(
                analysis.standalone_query,
                evidence[: self.settings.reranker_candidate_top_k],
            )
            evidence = rerank_result.evidence
        evidence = evidence[: self.settings.final_evidence_top_k]
        for index, item in enumerate(evidence, 1):
            item.source_id = f"S{index}"

        yield StageEvent(stage=PipelineStage.GENERATING)
        generated = await self.generator.generate(analysis.standalone_query, evidence, history)

        yield StageEvent(stage=PipelineStage.VERIFYING)
        citation_result, validation = await self.validator.validate(generated.answer, evidence)
        if not validation.valid:
            feedback = validation.reason or "; ".join(validation.unsupported_claims)
            yield StageEvent(stage=PipelineStage.GENERATING)
            generated = await self.generator.generate(
                analysis.standalone_query,
                evidence,
                history,
                repair_reason=feedback,
            )
            yield StageEvent(stage=PipelineStage.VERIFYING)
            citation_result, validation = await self.validator.validate(generated.answer, evidence)

        if not validation.valid or not citation_result.valid:
            yield ResultEvent(
                result=PipelineResult(
                    status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                    answer=INSUFFICIENT_MESSAGE,
                    evidence_count=len(evidence),
                    validation_score=validation.score,
                    analysis=analysis,
                    evidence=evidence,
                    retrieval=retrieval,
                    refusal_reason=RefusalReason.VALIDATION_FAILED,
                )
            )
            return

        yield ResultEvent(
            result=PipelineResult(
                status=AnswerStatus.ANSWERED,
                answer=generated.answer,
                citations=citation_result.citations,
                sources=citation_result.sources,
                evidence_count=len(evidence),
                validation_score=validation.score,
                analysis=analysis,
                evidence=evidence,
                retrieval=retrieval,
            )
        )
