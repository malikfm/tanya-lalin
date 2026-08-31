"""Compare retrieval profiles on frozen development queries and conditionally rerank."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.infrastructure.ai import EmbeddingClient
from app.infrastructure.ai.openai import OpenAIProvider
from app.infrastructure.corpus_index import StaticCorpusIndex
from app.rag.models import CorpusChunk, Evidence, QueryAnalysis, RetrievalResult
from app.rag.reranker import Reranker
from app.rag.retrieval import HybridRetriever, RetrievalProfile
from eval.run_eval import RESULTS_DIR, load_ground_truth_payload, mean_metric, ranked_metrics

SOURCE_REPORT = RESULTS_DIR / "eval-20260831T075520Z.json"
EXPERIMENT_SCHEMA_VERSION = 1
PROFILES = (
    RetrievalProfile.LEGACY,
    RetrievalProfile.BALANCED,
    RetrievalProfile.BALANCED_ARTICLE,
    RetrievalProfile.BALANCED_ARTICLE_DIVERSE,
)


class CachedEmbeddings:
    """Cache repeated query embeddings across experimental profiles."""

    def __init__(self, delegate: EmbeddingClient):
        self.delegate = delegate
        self.cache: dict[str, list[float]] = {}

    async def aembed_query(self, text: str) -> list[float]:
        if text not in self.cache:
            self.cache[text] = await self.delegate.aembed_query(text)
        return self.cache[text]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.delegate.aembed_documents(texts)


def evidence_ids(evidence: list[Evidence]) -> tuple[list[str], list[str]]:
    return (
        [item.chunk.chunk_id for item in evidence],
        list(dict.fromkeys(str(item.chunk.article_number) for item in evidence)),
    )


def cutoff_metrics(case: dict[str, Any], evidence: list[Evidence], cutoff: int) -> dict[str, float]:
    chunk_ids, articles = evidence_ids(evidence)
    return {
        **ranked_metrics(
            case["expected_articles"],
            articles,
            prefix="article",
            cutoff=cutoff,
        ),
        **ranked_metrics(
            case["evidence_chunk_ids"],
            chunk_ids,
            prefix="evidence",
            cutoff=cutoff,
        ),
    }


def trace_payload(case: dict[str, Any], retrieval: RetrievalResult) -> list[dict[str, Any]]:
    return [
        {
            "search_name": trace.search_name,
            "family": trace.family,
            "query": trace.query,
            "weight": trace.weight,
            "chunk_ids": [hit.chunk.chunk_id for hit in trace.hits],
            "articles": list(dict.fromkeys(str(hit.chunk.article_number) for hit in trace.hits)),
            "vector_relevance": [hit.vector_relevance for hit in trace.hits],
            "bm25_scores": [hit.bm25_score for hit in trace.hits],
            "metrics_at_5": {
                **ranked_metrics(
                    case["expected_articles"],
                    list(dict.fromkeys(str(hit.chunk.article_number) for hit in trace.hits)),
                    prefix="article",
                ),
                **ranked_metrics(
                    case["evidence_chunk_ids"],
                    [hit.chunk.chunk_id for hit in trace.hits],
                    prefix="evidence",
                ),
            },
        }
        for trace in retrieval.ranked_lists
    ]


def profile_case_payload(
    case: dict[str, Any],
    retrieval: RetrievalResult,
    latency_seconds: float,
) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    for cutoff in (5, 10, 20):
        metrics.update(cutoff_metrics(case, retrieval.candidates, cutoff))
    top_five_articles = [item.chunk.article_number for item in retrieval.candidates[:5]]
    metrics["duplicate_article_concentration_at_5"] = (
        1.0 - len(set(top_five_articles)) / len(top_five_articles) if top_five_articles else 0.0
    )
    metrics["retrieval_latency_seconds"] = latency_seconds
    chunk_ids, articles = evidence_ids(retrieval.candidates)
    fused_ids, fused_articles = evidence_ids(retrieval.fused_candidates)
    return {
        "id": case["id"],
        "retrieved_chunk_ids": chunk_ids,
        "retrieved_articles": articles,
        "fused_chunk_ids": fused_ids,
        "fused_articles": fused_articles,
        "ranked_lists": trace_payload(case, retrieval),
        "metrics": metrics,
    }


def profile_aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = list(results[0]["metrics"]) if results else []
    return {
        "evaluated_cases": len(results),
        **{f"mean_{name}": mean_metric(results, name) for name in metric_names},
    }


def diversity_eligible(results: list[dict[str, Any]]) -> tuple[bool, dict[str, int | float]]:
    misses = [item for item in results if item["metrics"]["article_hit_rate_at_5"] == 0]
    crowded_recoveries = [
        item
        for item in misses
        if item["metrics"]["article_hit_rate_at_10"] == 1
        and item["metrics"]["duplicate_article_concentration_at_5"] > 0
    ]
    ratio = len(crowded_recoveries) / len(misses) if misses else 0.0
    return ratio >= 0.2, {
        "misses_at_5": len(misses),
        "crowded_recoveries_at_10": len(crowded_recoveries),
        "crowded_recovery_ratio": round(ratio, 4),
    }


def select_profile(
    profiles: dict[str, dict[str, Any]],
    *,
    include_diversity: bool,
) -> str:
    eligible = [
        profile
        for profile in profiles
        if include_diversity or profile != RetrievalProfile.BALANCED_ARTICLE_DIVERSE
    ]

    def score(profile: str) -> tuple[float, float, float, float]:
        aggregate = profiles[profile]["aggregate"]
        return (
            aggregate["mean_article_hit_rate_at_5"],
            aggregate["mean_evidence_recall_at_5"],
            aggregate["mean_article_mrr_at_5"],
            -aggregate["mean_retrieval_latency_seconds"],
        )

    return max(eligible, key=score)


def reranker_eligible(aggregate: dict[str, Any]) -> bool:
    return (
        aggregate["mean_article_hit_rate_at_10"] >= 0.9
        and aggregate["mean_article_hit_rate_at_5"] < 0.85
    )


def reranker_accepted(
    control_results: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
) -> bool:
    control = profile_aggregate(control_results)
    reranked = profile_aggregate(reranked_results)
    hit_gain_cases = round(
        (reranked["mean_article_hit_rate_at_5"] - control["mean_article_hit_rate_at_5"])
        * len(control_results)
    )
    evidence_regression = (
        control["mean_evidence_recall_at_5"] - reranked["mean_evidence_recall_at_5"]
    )
    return (
        hit_gain_cases >= 2
        and evidence_regression <= 0.02
        and not any(item["reranker_failed"] for item in reranked_results)
    )


async def evaluate_profile(
    *,
    profile: RetrievalProfile,
    cases: list[dict[str, Any]],
    analyses: dict[str, QueryAnalysis],
    index: StaticCorpusIndex,
    embeddings: EmbeddingClient,
    settings: Settings,
) -> list[dict[str, Any]]:
    retriever = HybridRetriever(index, embeddings, settings, profile)
    results: list[dict[str, Any]] = []
    for case in cases:
        started = time.monotonic()
        retrieval = await retriever.retrieve(analyses[case["id"]])
        results.append(profile_case_payload(case, retrieval, time.monotonic() - started))
    return results


async def evaluate_reranker(
    *,
    cases: list[dict[str, Any]],
    analyses: dict[str, QueryAnalysis],
    profile_results: list[dict[str, Any]],
    chunk_by_id: dict[str, CorpusChunk],
    reranker: Reranker,
) -> list[dict[str, Any]]:
    case_by_id = {case["id"]: case for case in cases}
    results: list[dict[str, Any]] = []
    for profile_result in profile_results:
        candidates = [
            Evidence(
                source_id=f"S{rank}",
                chunk=chunk_by_id[chunk_id],
                fusion_score=1.0 / rank,
            )
            for rank, chunk_id in enumerate(profile_result["retrieved_chunk_ids"][:10], 1)
        ]
        reranked = await reranker.rerank(
            analyses[profile_result["id"]].standalone_query,
            candidates,
        )
        case = case_by_id[profile_result["id"]]
        metrics = cutoff_metrics(case, reranked.evidence, 5)
        metrics["retrieval_latency_seconds"] = reranked.latency_seconds
        chunk_ids, articles = evidence_ids(reranked.evidence)
        results.append(
            {
                "id": case["id"],
                "retrieved_chunk_ids": chunk_ids,
                "retrieved_articles": articles,
                "metrics": metrics,
                "reranker_failed": reranked.failed,
                "input_tokens": reranked.input_tokens,
                "output_tokens": reranked.output_tokens,
            }
        )
    return results


async def run(source_report: Path, *, skip_reranker: bool = False) -> Path:
    source_text = await asyncio.to_thread(source_report.read_text, encoding="utf-8")
    source = json.loads(source_text)
    analyses = {
        item["id"]: QueryAnalysis.model_validate(item["query_analysis"])
        for item in source["results"]
        if item.get("query_analysis") is not None
    }
    cases = [
        case
        for case in load_ground_truth_payload()["cases"]
        if case.get("split") == "development" and case.get("cohort") == "answerable"
    ]
    missing = [case["id"] for case in cases if case["id"] not in analyses]
    if missing:
        raise ValueError(f"Source report is missing query analyses for: {', '.join(missing)}")

    settings = Settings().model_copy(
        update={
            "vector_search_top_k": 20,
            "bm25_search_top_k": 20,
            "fusion_top_k": 20,
        }
    )
    provider = OpenAIProvider(settings)
    index = StaticCorpusIndex(settings.corpus_dir, provider.embedding_identity)
    embeddings = CachedEmbeddings(provider.create_embeddings())
    profile_payloads: dict[str, dict[str, Any]] = {}
    retrieval_by_profile: dict[str, list[dict[str, Any]]] = {}
    for profile in PROFILES:
        results = await evaluate_profile(
            profile=profile,
            cases=cases,
            analyses=analyses,
            index=index,
            embeddings=embeddings,
            settings=settings,
        )
        retrieval_by_profile[profile] = results
        profile_payloads[profile] = {
            "aggregate": profile_aggregate(results),
            "results": results,
        }

    allow_diversity, diversity_diagnostics = diversity_eligible(
        retrieval_by_profile[RetrievalProfile.BALANCED_ARTICLE]
    )
    selected = select_profile(profile_payloads, include_diversity=allow_diversity)
    selected_results = retrieval_by_profile[selected]
    selected_aggregate = profile_payloads[selected]["aggregate"]

    reranker_payload: dict[str, Any] = {
        "eligible": reranker_eligible(selected_aggregate),
        "evaluated": False,
        "accepted": False,
    }
    if reranker_payload["eligible"] and not skip_reranker:
        chunk_by_id = {chunk.chunk_id: chunk for chunk in index.chunks}
        reranker = Reranker(
            provider.create_chat_model(
                model=settings.reranker_model,
                reasoning_effort=settings.analysis_reasoning_effort,
            )
        )
        reranked = await evaluate_reranker(
            cases=cases,
            analyses=analyses,
            profile_results=selected_results,
            chunk_by_id=chunk_by_id,
            reranker=reranker,
        )
        reranker_payload = {
            "eligible": True,
            "evaluated": True,
            "accepted": reranker_accepted(selected_results, reranked),
            "aggregate": profile_aggregate(reranked),
            "failure_count": sum(item["reranker_failed"] for item in reranked),
            "input_tokens": sum(item["input_tokens"] or 0 for item in reranked),
            "output_tokens": sum(item["output_tokens"] or 0 for item in reranked),
            "results": reranked,
        }

    report = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run_at": datetime.now(UTC).isoformat(),
        "mode": "development_retrieval_experiment",
        "source_report": str(source_report.relative_to(source_report.parents[2])),
        "dataset_version": load_ground_truth_payload().get("dataset_version"),
        "corpus_version": index.version,
        "configuration": {
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
            "vector_search_top_k": 20,
            "bm25_search_top_k": 20,
            "fusion_top_k": 20,
            "final_evidence_top_k": 5,
            "reranker_model": settings.reranker_model,
        },
        "diversity": {"eligible": allow_diversity, **diversity_diagnostics},
        "selected_profile": selected,
        "profiles": profile_payloads,
        "reranker": reranker_payload,
    }
    output = RESULTS_DIR / f"retrieval-experiment-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-report", type=Path, default=SOURCE_REPORT)
    parser.add_argument("--skip-reranker", action="store_true")
    args = parser.parse_args()
    output = asyncio.run(run(args.analysis_report, skip_reranker=args.skip_reranker))
    print(f"Retrieval experiment written to {output}")


if __name__ == "__main__":
    main()
