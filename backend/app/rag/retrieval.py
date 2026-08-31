"""Traceable hybrid retrieval with family-normalized reciprocal-rank fusion."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from enum import StrEnum

from app.config import Settings
from app.infrastructure.ai import EmbeddingClient
from app.infrastructure.corpus_index import StaticCorpusIndex
from app.rag.models import (
    Evidence,
    QueryAnalysis,
    RankedListTrace,
    RetrievalResult,
    SearchHit,
)

ARTICLE_REFERENCE_PATTERN = re.compile(r"\bpasal\s+(\d{1,3})\b", re.IGNORECASE)


class RetrievalProfile(StrEnum):
    """Fixed profiles used for reproducible retrieval experiments."""

    LEGACY = "legacy"
    BALANCED = "balanced"
    BALANCED_ARTICLE = "balanced_article"
    BALANCED_ARTICLE_DIVERSE = "balanced_article_diverse"


class HybridRetriever:
    """Search independent lists and retain the complete fusion trace."""

    def __init__(
        self,
        index: StaticCorpusIndex,
        embeddings: EmbeddingClient,
        settings: Settings,
        profile: RetrievalProfile | str | None = None,
    ):
        self.index = index
        self.embeddings = embeddings
        self.settings = settings
        self.profile = RetrievalProfile(profile or settings.retrieval_profile)

    async def retrieve(self, analysis: QueryAnalysis) -> RetrievalResult:
        traces = await self._search(analysis)
        fused = self._fuse([(trace.hits, trace.weight) for trace in traces])
        fused = fused[: self.settings.fusion_top_k]
        candidates = (
            self._soft_diversity(fused)
            if self.profile is RetrievalProfile.BALANCED_ARTICLE_DIVERSE
            else list(fused)
        )
        return RetrievalResult(
            ranked_lists=traces,
            fused_candidates=fused,
            candidates=candidates,
        )

    async def _search(self, analysis: QueryAnalysis) -> list[RankedListTrace]:
        if self.profile is RetrievalProfile.LEGACY:
            return await self._legacy_search(analysis)

        traces: list[RankedListTrace] = []
        vector_specs: list[tuple[str, str, float]] = [
            ("vector_legal", analysis.legal_query, 0.5),
        ]
        if analysis.standalone_query.casefold() != analysis.legal_query.casefold():
            vector_specs.append(("vector_original", analysis.standalone_query, 0.3))
        phrases = [phrase for phrase in analysis.key_phrases[:3] if phrase.strip()]
        phrase_share = 0.2 / len(phrases) if phrases else 0.0
        vector_specs.extend(
            (f"vector_phrase_{index}", phrase, phrase_share)
            for index, phrase in enumerate(phrases, 1)
        )
        traces.extend(
            await self._vector_traces(
                vector_specs,
                family_weight=self.settings.vector_rrf_weight,
            )
        )

        bm25_specs: list[tuple[str, str, float]] = [
            ("bm25_original", analysis.standalone_query, 0.6),
        ]
        if analysis.standalone_query.casefold() != analysis.legal_query.casefold():
            bm25_specs.append(("bm25_legal", analysis.legal_query, 0.4))
        traces.extend(self._bm25_traces(bm25_specs, family_weight=self.settings.bm25_rrf_weight))

        if self.profile in {
            RetrievalProfile.BALANCED_ARTICLE,
            RetrievalProfile.BALANCED_ARTICLE_DIVERSE,
        }:
            article_numbers = self.extract_article_references(analysis.standalone_query)
            article_weight = (
                self.settings.article_reference_rrf_weight / len(article_numbers)
                if article_numbers
                else 0.0
            )
            for article_number in article_numbers:
                traces.append(
                    RankedListTrace(
                        search_name=f"article_{article_number}",
                        family="article",
                        query=f"Pasal {article_number}",
                        weight=article_weight,
                        hits=self._deduplicate(
                            self.index.article_search(
                                article_number,
                                self.settings.vector_search_top_k,
                            )
                        ),
                    )
                )
        return traces

    async def _legacy_search(self, analysis: QueryAnalysis) -> list[RankedListTrace]:
        specs: list[tuple[str, str, float]] = [("legal", analysis.legal_query, 2.0)]
        if analysis.standalone_query.casefold() != analysis.legal_query.casefold():
            specs.append(("original", analysis.standalone_query, 1.0))
        specs.extend(
            (f"phrase_{index}", phrase, 0.8)
            for index, phrase in enumerate(analysis.key_phrases[:3], 1)
            if phrase.strip()
        )
        traces = await self._vector_traces(specs, family_weight=1.0, normalize=False)
        hits = self._deduplicate(
            self.index.bm25_search(analysis.legal_query, self.settings.bm25_search_top_k)
        )
        traces.append(
            RankedListTrace(
                search_name="bm25",
                family="bm25",
                query=analysis.legal_query,
                weight=0.7,
                hits=hits,
            )
        )
        return traces

    async def _vector_traces(
        self,
        specs: Sequence[tuple[str, str, float]],
        *,
        family_weight: float,
        normalize: bool = True,
    ) -> list[RankedListTrace]:
        total_share = sum(share for _, _, share in specs) or 1.0
        traces: list[RankedListTrace] = []
        for search_name, query, share in specs:
            embedding = await self.embeddings.aembed_query(query)
            hits = self.index.vector_search(
                embedding,
                search_name=search_name,
                top_k=self.settings.vector_search_top_k,
                min_relevance=self.settings.min_vector_relevance,
            )
            weight = family_weight * share / total_share if normalize else share
            traces.append(
                RankedListTrace(
                    search_name=search_name,
                    family="vector",
                    query=query,
                    weight=weight,
                    hits=self._deduplicate(hits),
                )
            )
        return traces

    def _bm25_traces(
        self,
        specs: Sequence[tuple[str, str, float]],
        *,
        family_weight: float,
    ) -> list[RankedListTrace]:
        total_share = sum(share for _, _, share in specs) or 1.0
        return [
            RankedListTrace(
                search_name=search_name,
                family="bm25",
                query=query,
                weight=family_weight * share / total_share,
                hits=self._rename_hits(
                    self.index.bm25_search(query, self.settings.bm25_search_top_k),
                    search_name,
                ),
            )
            for search_name, query, share in specs
        ]

    @staticmethod
    def _rename_hits(hits: Sequence[SearchHit], search_name: str) -> list[SearchHit]:
        return [hit.model_copy(update={"search_name": search_name}) for hit in hits]

    @staticmethod
    def extract_article_references(query: str) -> list[int]:
        """Extract unique article numbers only from explicit ``Pasal N`` syntax."""

        return list(dict.fromkeys(int(value) for value in ARTICLE_REFERENCE_PATTERN.findall(query)))

    @staticmethod
    def _deduplicate(hits: Sequence[SearchHit]) -> list[SearchHit]:
        seen: set[str] = set()
        result: list[SearchHit] = []
        for hit in hits:
            if hit.chunk.chunk_id not in seen:
                result.append(hit)
                seen.add(hit.chunk.chunk_id)
        return result

    def _fuse(
        self,
        ranked_lists: Sequence[tuple[Sequence[SearchHit], float]],
        rrf_constant: int = 60,
    ) -> list[Evidence]:
        scores: dict[str, float] = defaultdict(float)
        hits_by_id: dict[str, list[SearchHit]] = defaultdict(list)
        for hits, weight in ranked_lists:
            for rank, hit in enumerate(hits, 1):
                chunk_id = hit.chunk.chunk_id
                scores[chunk_id] += weight / (rrf_constant + rank)
                hits_by_id[chunk_id].append(hit)

        evidence: list[Evidence] = []
        for chunk_id, score in scores.items():
            hits = hits_by_id[chunk_id]
            vector_scores = [
                hit.vector_relevance for hit in hits if hit.vector_relevance is not None
            ]
            bm25_ranks = [hit.rank for hit in hits if hit.search_name.startswith("bm25")]
            evidence.append(
                Evidence(
                    source_id=f"S{len(evidence) + 1}",
                    chunk=hits[0].chunk,
                    fusion_score=score,
                    vector_relevance=max(vector_scores) if vector_scores else None,
                    bm25_rank=min(bm25_ranks) if bm25_ranks else None,
                )
            )

        evidence.sort(key=lambda item: (-item.fusion_score, item.chunk.chunk_id))
        for index, item in enumerate(evidence, 1):
            item.source_id = f"S{index}"
        return evidence

    @staticmethod
    def _soft_diversity(evidence: Sequence[Evidence], reserved_slots: int = 3) -> list[Evidence]:
        """Reserve early positions for unique articles, then preserve fused order."""

        selected: list[Evidence] = []
        seen_articles: set[int] = set()
        for item in evidence:
            if item.chunk.article_number not in seen_articles:
                selected.append(item)
                seen_articles.add(item.chunk.article_number)
                if len(selected) == min(reserved_slots, len(evidence)):
                    break
        selected_ids = {item.chunk.chunk_id for item in selected}
        selected.extend(item for item in evidence if item.chunk.chunk_id not in selected_ids)
        for index, item in enumerate(selected, 1):
            item.source_id = f"S{index}"
        return selected
