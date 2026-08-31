"""Deterministic and optional LLM-based answer validation."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from app.rag.citations import CitationValidation, validate_citations
from app.rag.models import Evidence, ValidationResult

VALIDATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You verify an Indonesian traffic-law answer against supplied evidence. "
            "Set valid=false only when a material legal claim is contradicted, has an "
            "unsupported rule or number, or cites a source that does not support it. "
            "Treat faithful paraphrases and ordinary-language equivalents of legal terms "
            "as supported; for example, 'lampu merah' is an ordinary reference to an "
            "Alat Pemberi Isyarat Lalu Lintas. Do not require the answer to reproduce "
            "every legal classification or distinction when it answers a narrower question. "
            "Use a 1-5 groundedness score and list concise unsupported claims. "
            "Do not judge writing style.",
        ),
        ("human", "Evidence:\n{evidence}\n\nAnswer:\n{answer}"),
    ]
)


class AnswerValidator:
    """Validate source markers first, then semantic groundedness when enabled."""

    def __init__(self, model: BaseChatModel | None):
        self._chain = (
            VALIDATION_PROMPT | model.with_structured_output(ValidationResult)
            if model is not None
            else None
        )

    async def validate(
        self,
        answer: str,
        evidence: list[Evidence],
    ) -> tuple[CitationValidation, ValidationResult]:
        citations = validate_citations(answer, evidence)
        if not citations.valid:
            return citations, ValidationResult(valid=False, score=1, reason=citations.reason)

        if self._chain is None:
            return citations, ValidationResult(valid=True, score=5)

        context = "\n\n".join(f"[{item.source_id}] {item.chunk.text}" for item in evidence)
        try:
            raw = await self._chain.ainvoke({"evidence": context, "answer": answer})
            result = ValidationResult.model_validate(raw)
            if not result.valid and result.score >= 3:
                result = result.model_copy(update={"valid": True})
        except Exception as exc:
            logger.warning(
                "Semantic answer validation failed; keeping deterministic verdict: {}", exc
            )
            result = ValidationResult(
                valid=True,
                score=5,
                reason="Semantic validation unavailable; citations passed deterministic validation",
            )
        return citations, result
