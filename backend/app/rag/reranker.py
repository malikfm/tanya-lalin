"""Optional structured LLM reranker with observable fallback behavior."""

from __future__ import annotations

import time
from typing import Annotated, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from app.rag.models import Evidence, RerankResult


class RerankScores(BaseModel):
    scores: list[Annotated[float, Field(ge=0, le=10)]] = Field(default_factory=list)


RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Score each Indonesian legal excerpt for relevance to the question from 0 to 10. "
            "Consider the article, paragraph, excerpt type, and text. Return exactly one score "
            "per excerpt in the same order. Do not answer the question.",
        ),
        ("human", "Question: {query}\n\nExcerpts:\n{evidence}"),
    ]
)


class Reranker:
    """Rerank fused evidence while falling back safely on provider errors."""

    def __init__(self, model: BaseChatModel):
        structured = model.with_structured_output(RerankScores, include_raw=True)
        self._chain = RERANK_PROMPT | structured

    async def rerank(self, query: str, evidence: list[Evidence]) -> RerankResult:
        if len(evidence) <= 1:
            return RerankResult(evidence=evidence)
        rendered = "\n".join(
            self._render_item(index, item) for index, item in enumerate(evidence, 1)
        )
        started = time.monotonic()
        try:
            response = await self._chain.ainvoke({"query": query, "evidence": rendered})
            parsed, raw = self._parse_response(response)
            if len(parsed.scores) != len(evidence):
                raise ValueError("Reranker returned an unexpected number of scores")
            for item, score in zip(evidence, parsed.scores, strict=True):
                item.reranker_score = float(score)
            reranked = sorted(
                evidence,
                key=lambda item: (-(item.reranker_score or 0.0), -item.fusion_score),
            )
            for index, item in enumerate(reranked, 1):
                item.source_id = f"S{index}"
            input_tokens, output_tokens = self._usage(raw)
            return RerankResult(
                evidence=reranked,
                latency_seconds=time.monotonic() - started,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as exc:
            logger.warning("Reranking failed; preserving fused order: {}", exc)
            return RerankResult(
                evidence=evidence,
                failed=True,
                latency_seconds=time.monotonic() - started,
            )

    @staticmethod
    def _render_item(index: int, item: Evidence) -> str:
        chunk = item.chunk
        paragraph = f", paragraph {chunk.paragraph_number}" if chunk.paragraph_number else ""
        return (
            f"{index}. Article {chunk.article_number}{paragraph}; type={chunk.chunk_type}; "
            f"chunk_id={chunk.chunk_id}\n{chunk.text[:800]}"
        )

    @staticmethod
    def _parse_response(response: Any) -> tuple[RerankScores, AIMessage | None]:
        if isinstance(response, dict) and "parsed" in response:
            if response.get("parsing_error") is not None:
                raise ValueError("Reranker structured output could not be parsed")
            return RerankScores.model_validate(response["parsed"]), response.get("raw")
        return RerankScores.model_validate(response), None

    @staticmethod
    def _usage(raw: AIMessage | None) -> tuple[int | None, int | None]:
        usage = raw.usage_metadata if isinstance(raw, AIMessage) else None
        if not usage:
            return None, None
        return usage.get("input_tokens"), usage.get("output_tokens")
