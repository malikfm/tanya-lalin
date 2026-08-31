"""Grounded Indonesian answer generation."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.exceptions import ProviderUnavailableError
from app.rag.models import ConversationMessage, Evidence, GeneratedAnswer

SYSTEM_PROMPT = """Anda adalah asisten informasi aturan lalu lintas Indonesia.
Jawab hanya berdasarkan sumber hukum yang diberikan.

Rules:
1. Answer directly in formal, accessible Indonesian.
2. Every factual legal claim must end with one or more source markers such as [S1] or [S1][S2].
3. Use only source IDs that appear in the supplied evidence.
4. State imprisonment and maximum fines precisely when the evidence provides them.
5. Do not invent rules, article numbers, exceptions, or sanctions.
6. Use short paragraphs or bullets when useful.
7. Do not include a generic legal disclaimer; the application handles scope disclosure.
"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Evidence:\n{evidence}\n\nConversation context:\n{history}\n\nQuestion:\n{query}",
        ),
    ]
)


class AnswerGenerator:
    """Generate or repair one evidence-grounded answer."""

    def __init__(self, model: BaseChatModel):
        self._chain = ANSWER_PROMPT | model | StrOutputParser()

    async def generate(
        self,
        query: str,
        evidence: list[Evidence],
        history: list[ConversationMessage],
        repair_reason: str | None = None,
    ) -> GeneratedAnswer:
        evidence_text = "\n\n".join(
            f"[{item.source_id}] Pasal {item.chunk.article_number}"
            f"{f' ayat ({item.chunk.paragraph_number})' if item.chunk.paragraph_number else ''} "
            f"({'Penjelasan' if item.chunk.chunk_type == 'elucidation' else 'Isi'}):\n"
            f"{item.chunk.text}"
            for item in evidence
        )
        history_text = (
            "\n".join(
                f"{'Pengguna' if message.role == 'user' else 'Asisten'}: {message.content[:500]}"
                for message in history[-6:]
            )
            or "Tidak ada riwayat sebelumnya."
        )
        effective_query = query
        if repair_reason:
            effective_query += (
                "\n\nThe previous draft failed validation. Produce a corrected answer. "
                f"Validation feedback: {repair_reason}"
            )
        try:
            answer = await self._chain.ainvoke(
                {
                    "evidence": evidence_text,
                    "history": history_text,
                    "query": effective_query,
                }
            )
        except Exception as exc:
            raise ProviderUnavailableError(
                "The answer provider is temporarily unavailable."
            ) from exc
        return GeneratedAnswer(answer=answer.strip())
