"""Conversation-aware query analysis and deterministic fallback expansion."""

from __future__ import annotations

import re
import unicodedata

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from app.rag.models import ConversationMessage, QueryAnalysis

TERM_MAPPINGS: dict[str, list[str]] = {
    "lampu merah": ["Alat Pemberi Isyarat Lalu Lintas", "APILL"],
    "menerobos": ["melanggar aturan perintah atau larangan"],
    "ngebut": ["batas kecepatan", "kecepatan maksimal"],
    "sim": ["Surat Izin Mengemudi"],
    "stnk": ["Surat Tanda Nomor Kendaraan Bermotor"],
    "helm": ["helm standar nasional Indonesia"],
    "sabuk pengaman": ["sabuk keselamatan"],
    "parkir sembarangan": ["larangan parkir"],
    "trotoar": ["fasilitas Pejalan Kaki"],
    "menyalip": ["melewati Kendaraan", "mendahului"],
    "motor": ["Sepeda Motor", "Kendaraan Bermotor"],
    "mobil": ["Kendaraan Bermotor"],
    "tilang": ["pidana", "denda", "pelanggaran"],
    "plat nomor": ["Tanda Nomor Kendaraan Bermotor", "TNKB"],
    "lawan arah": ["melawan arus", "arah lalu lintas"],
    "zebra cross": ["tempat penyeberangan Pejalan Kaki"],
}


ZERO_WIDTH_PATTERN = re.compile(r"[\u200b-\u200f\u2060\ufeff]")


def normalize_for_policy(text: str) -> str:
    """Normalize user-controlled text before deterministic policy checks."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = ZERO_WIDTH_PATTERN.sub("", normalized)
    return " ".join(normalized.split())


INJECTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?|prompts?)",
        r"you\s+are\s+now\s+a",
        r"system\s*prompt",
        r"print\s+your\s+(?:instructions?|rules?|prompt)",
        r"forget\s+(?:everything|all)",
        r"\bjailbreak\b",
        r"disregard\s+(?:all|any|the)",
        r"override\s+(?:your|the|all)",
        r"\babaikan\b.{0,100}\b(?:instruksi|aturan|peraturan|konteks|sumber|kebijakan)\b.{0,80}\b(?:sebelumnya|sistem|developer|keselamatan|diberikan)\b",
        r"\babaikan\b.{0,80}\b(?:seluruh|semua)\b.{0,80}\b(?:instruksi|peraturan|konteks|sumber)\b",
        r"\bprioritaskan\b.{0,80}\binstruksi\b(?:\s+(?:ini|terakhir|utama|baru))?",
        r"\binstruksi\s+(?:sistem|utama|developer)\b",
        r"\b(?:tampilkan|keluarkan|cetak|ungkapkan)\b.{0,80}"
        r"\b(?:prompt|instruksi|konteks tersembunyi|seluruh konteks|sumber hukum)\b",
        r"\[(?:prompt injection|instruction manipulation|instruksi sistem|signature)\b[^]]*\]",
    )
)

FRESHNESS_PATTERN = re.compile(
    r"\b(?:saat ini|sekarang|terkini|terbaru|hari ini|tahun ini|masih berlaku|"
    r"berlaku saat ini|20(?:2[4-9]|[3-9]\d))\b",
    re.IGNORECASE,
)
EXTERNAL_INSTRUMENT_PATTERN = re.compile(
    r"\b(?:peraturan daerah|perda|peraturan gubernur|pergub|keputusan gubernur|"
    r"peraturan menteri|permen(?:hub)?)\b",
    re.IGNORECASE,
)
INSTRUMENT_CONTENT_PATTERN = re.compile(
    r"\b(?:berapa|apa saja|mana|tarif|biaya|jadwal|jam|ruas|zona|pengecualian|"
    r"mekanisme|prosedur|batas|kadar|ketentuan)\b",
    re.IGNORECASE,
)


ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You analyze questions for an Indonesian road-traffic law assistant. "
            "Use the conversation history to turn follow-up questions into standalone questions. "
            "Classify whether the question concerns Indonesian road traffic or road transport. "
            "Produce an Indonesian legal search query close to the terminology of "
            "Law No. 22 of 2009, "
            "plus concise key phrases. The application determines corpus freshness separately, "
            "so always leave requires_external_sources false and external_source_reason empty. "
            "Do not answer the legal question.",
        ),
        (
            "human",
            "Conversation history:\n{history}\n\nCurrent user question:\n{query}",
        ),
    ]
)


class PromptInjectionDetected(ValueError):
    """Raised when deterministic input checks detect instruction manipulation."""


class QueryAnalyzer:
    """Analyze user intent with one structured LLM call and a safe fallback."""

    def __init__(self, model: BaseChatModel):
        self._chain = ANALYSIS_PROMPT | model.with_structured_output(QueryAnalysis)

    @staticmethod
    def validate_input(query: str) -> None:
        query = normalize_for_policy(query)
        for pattern in INJECTION_PATTERNS:
            if pattern.search(query):
                raise PromptInjectionDetected("Prompt injection pattern detected")

    async def analyze(
        self,
        query: str,
        history: list[ConversationMessage],
    ) -> QueryAnalysis:
        history_text = (
            "\n".join(
                f"{'Pengguna' if item.role == 'user' else 'Asisten'}: {item.content[:500]}"
                for item in history[-10:]
            )
            or "No previous messages."
        )
        try:
            result = await self._chain.ainvoke({"query": query, "history": history_text})
            analysis = QueryAnalysis.model_validate(result)
            analysis = self._with_deterministic_terms(analysis, query)
            return self._with_corpus_scope(analysis, query)
        except Exception as exc:
            logger.warning("Query analysis failed; using deterministic fallback: {}", exc)
            return self.fallback(query)

    @staticmethod
    def _with_deterministic_terms(
        analysis: QueryAnalysis,
        original_query: str,
    ) -> QueryAnalysis:
        """Supplement successful model analysis with stable legal terminology."""

        searchable = f"{original_query} {analysis.standalone_query}".casefold()
        mapped_terms = [
            term
            for everyday, mapped in TERM_MAPPINGS.items()
            if everyday in searchable
            for term in mapped
        ]
        if not mapped_terms:
            return analysis

        existing_phrases = {phrase.casefold() for phrase in analysis.key_phrases}
        additional_phrases = [
            term for term in mapped_terms if term.casefold() not in existing_phrases
        ]
        legal_query_casefold = analysis.legal_query.casefold()
        legal_query_terms = [analysis.legal_query]
        legal_query_terms.extend(
            term for term in mapped_terms if term.casefold() not in legal_query_casefold
        )
        return analysis.model_copy(
            update={
                "legal_query": " ".join(legal_query_terms),
                "key_phrases": [*analysis.key_phrases, *additional_phrases][:5],
            }
        )

    @staticmethod
    def _with_corpus_scope(
        analysis: QueryAnalysis,
        original_query: str,
    ) -> QueryAnalysis:
        """Apply high-precision policy for requests the bundled corpus cannot verify."""

        searchable = normalize_for_policy(f"{original_query} {analysis.standalone_query}")
        if FRESHNESS_PATTERN.search(searchable):
            return analysis.model_copy(
                update={
                    "requires_external_sources": True,
                    "external_source_reason": "current_or_latest_information",
                }
            )
        if EXTERNAL_INSTRUMENT_PATTERN.search(searchable) and INSTRUMENT_CONTENT_PATTERN.search(
            searchable
        ):
            return analysis.model_copy(
                update={
                    "requires_external_sources": True,
                    "external_source_reason": "instrument_outside_corpus",
                }
            )
        return analysis.model_copy(
            update={"requires_external_sources": False, "external_source_reason": ""}
        )

    @staticmethod
    def fallback(query: str) -> QueryAnalysis:
        lowered = query.casefold()
        phrases = [
            term
            for everyday, mapped in TERM_MAPPINGS.items()
            if everyday in lowered
            for term in mapped
        ]
        legal_query = " ".join([query, *phrases]).strip()
        analysis = QueryAnalysis(
            standalone_query=query,
            is_in_scope=True,
            legal_query=legal_query,
            key_phrases=phrases[:5],
        )
        return QueryAnalyzer._with_corpus_scope(analysis, query)
