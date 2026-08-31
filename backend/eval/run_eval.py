"""Run the approved evaluation suite against lexical or configured AI providers."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from app.config import Settings
from app.dependencies import AppContainer
from app.infrastructure.ai import EmbeddingIdentity
from app.infrastructure.ai.openai import OpenAIProvider
from app.infrastructure.corpus_index import StaticCorpusIndex
from app.rag.models import AnswerStatus, Evidence, PipelineResult, ResultEvent, RetrievalResult

EVAL_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_PATH = EVAL_DIR / "ground_truth.json"
RESULTS_DIR = EVAL_DIR / "results"
REPORT_SCHEMA_VERSION = 2
RETRIEVAL_CUTOFF = 5

ARTICLE_METRIC_NAMES = (
    "article_hit_rate_at_5",
    "article_recall_at_5",
    "article_mrr_at_5",
    "article_precision_at_5",
    "article_ndcg_at_5",
)
EVIDENCE_METRIC_NAMES = (
    "evidence_hit_rate_at_5",
    "evidence_recall_at_5",
    "evidence_mrr_at_5",
    "evidence_precision_at_5",
    "evidence_ndcg_at_5",
)
RETRIEVAL_METRIC_NAMES = ARTICLE_METRIC_NAMES + EVIDENCE_METRIC_NAMES
CANDIDATE_METRIC_NAMES = tuple(
    f"candidate_{identifier}_{metric}_at_{cutoff}"
    for identifier in ("article", "evidence")
    for metric in ("hit_rate", "recall", "mrr", "precision", "ndcg")
    for cutoff in (10, 20)
)


FACT_COVERAGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You evaluate whether an Indonesian traffic-law answer covers required facts. "
            "Each numbered group contains alternative phrasings of one required fact. "
            "Mark a group covered when the answer communicates that fact faithfully, even "
            "with different wording. Be strict about legal conditions, negation, amounts, "
            "durations, and article numbers. Return exactly one verdict for every group, in "
            "the supplied order. Do not evaluate writing style or add facts.",
        ),
        (
            "human",
            "Question:\n{question}\n\nRequired fact groups:\n{fact_groups}\n\nAnswer:\n{answer}",
        ),
    ]
)


class FactGroupVerdict(BaseModel):
    """Semantic coverage verdict for one required-fact group."""

    index: int = Field(ge=0)
    covered: bool


class FactCoverageVerdict(BaseModel):
    """Structured fact coverage returned by the evaluation model."""

    groups: list[FactGroupVerdict]


class FactJudge:
    """Evaluate required-fact coverage without affecting application behavior."""

    def __init__(self, model: BaseChatModel):
        self._chain = FACT_COVERAGE_PROMPT | model.with_structured_output(FactCoverageVerdict)

    async def evaluate(
        self,
        *,
        question: str,
        answer: str,
        fact_groups: list[list[str]],
    ) -> FactCoverageVerdict | None:
        if not fact_groups:
            return FactCoverageVerdict(groups=[])
        formatted_groups = "\n".join(
            f"{index}. " + " OR ".join(group) for index, group in enumerate(fact_groups)
        )
        try:
            raw = await self._chain.ainvoke(
                {
                    "question": question,
                    "answer": answer,
                    "fact_groups": formatted_groups,
                }
            )
            verdict = FactCoverageVerdict.model_validate(raw)
            if [item.index for item in verdict.groups] != list(range(len(fact_groups))):
                raise ValueError("Fact judge returned incomplete or unordered group indexes")
            return verdict
        except Exception as exc:
            logger.warning("Required-fact evaluation failed: {}", exc)
            return None


def load_ground_truth_payload() -> dict[str, Any]:
    """Load the promoted suite and retain its version metadata."""

    payload = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {
            "schema_version": 1,
            "dataset_version": "legacy",
            "corpus_version": None,
            "cases": payload,
        }
    return payload


def load_ground_truth(
    limit: int | None = None,
    category: str | None = None,
    split: str | None = None,
    cohort: str | None = None,
) -> list[dict[str, Any]]:
    cases = load_ground_truth_payload()["cases"]
    if category:
        cases = [case for case in cases if case["category"] == category]
    if split:
        cases = [case for case in cases if case.get("split") == split]
    if cohort:
        cases = [case for case in cases if case.get("cohort") == cohort]
    return cases[:limit] if limit else cases


def unique_ranked(values: list[str]) -> list[str]:
    """Deduplicate a ranking while preserving the first occurrence."""

    return list(dict.fromkeys(values))


def ranked_metrics(
    expected: list[str],
    ranked: list[str],
    *,
    prefix: str,
    cutoff: int = RETRIEVAL_CUTOFF,
) -> dict[str, float | None]:
    """Calculate bounded binary-relevance metrics for one ranked identifier type."""

    expected_set = set(expected)
    names = {
        "hit": f"{prefix}_hit_rate_at_{cutoff}",
        "recall": f"{prefix}_recall_at_{cutoff}",
        "mrr": f"{prefix}_mrr_at_{cutoff}",
        "precision": f"{prefix}_precision_at_{cutoff}",
        "ndcg": f"{prefix}_ndcg_at_{cutoff}",
    }
    if not expected_set:
        return {name: None for name in names.values()}

    top_ranked = unique_ranked(ranked)[:cutoff]
    relevant = [item in expected_set for item in top_ranked]
    hit_count = len(expected_set.intersection(top_ranked))
    reciprocal_rank = next(
        (1.0 / rank for rank, is_relevant in enumerate(relevant, 1) if is_relevant),
        0.0,
    )
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, is_relevant in enumerate(relevant, 1) if is_relevant
    )
    ideal_count = min(len(expected_set), cutoff)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        names["hit"]: float(hit_count > 0),
        names["recall"]: hit_count / len(expected_set),
        names["mrr"]: reciprocal_rank,
        names["precision"]: sum(relevant) / max(len(top_ranked), 1),
        names["ndcg"]: dcg / ideal_dcg,
    }


def retrieval_metrics(expected: list[str], articles: list[str]) -> dict[str, float | None]:
    """Compatibility wrapper for unique-article retrieval metrics."""

    return ranked_metrics(expected, articles, prefix="article")


def case_retrieval_metrics(
    case: dict[str, Any],
    *,
    article_ranking: list[str],
    evidence_ranking: list[str],
) -> dict[str, float | None]:
    if case.get("cohort", "answerable") != "answerable":
        return {name: None for name in RETRIEVAL_METRIC_NAMES}
    return {
        **ranked_metrics(case.get("expected_articles", []), article_ranking, prefix="article"),
        **ranked_metrics(case.get("evidence_chunk_ids", []), evidence_ranking, prefix="evidence"),
    }


def candidate_metrics(case: dict[str, Any], evidence: list[Evidence]) -> dict[str, float | None]:
    """Measure whether relevant evidence survives in the pre-generation candidate pool."""

    if case.get("cohort", "answerable") != "answerable":
        return {}
    chunk_ids = [item.chunk.chunk_id for item in evidence]
    articles = unique_ranked([str(item.chunk.article_number) for item in evidence])
    metrics: dict[str, float | None] = {}
    for cutoff in (10, 20):
        metrics.update(
            ranked_metrics(
                case.get("expected_articles", []),
                articles,
                prefix="candidate_article",
                cutoff=cutoff,
            )
        )
        metrics.update(
            ranked_metrics(
                case.get("evidence_chunk_ids", []),
                chunk_ids,
                prefix="candidate_evidence",
                cutoff=cutoff,
            )
        )
    return metrics


def retrieval_trace_payload(
    case: dict[str, Any], retrieval: RetrievalResult | None
) -> list[dict[str, Any]] | None:
    """Serialize isolated ranked lists without leaking them through public contracts."""

    if retrieval is None:
        return None
    payload: list[dict[str, Any]] = []
    for trace in retrieval.ranked_lists:
        chunk_ids = [hit.chunk.chunk_id for hit in trace.hits]
        articles = unique_ranked([str(hit.chunk.article_number) for hit in trace.hits])
        payload.append(
            {
                "search_name": trace.search_name,
                "family": trace.family,
                "query": trace.query,
                "weight": trace.weight,
                "chunk_ids": chunk_ids,
                "articles": articles,
                "metrics_at_5": (
                    case_retrieval_metrics(
                        case,
                        article_ranking=articles,
                        evidence_ranking=chunk_ids,
                    )
                    if case.get("cohort") == "answerable"
                    else None
                ),
            }
        )
    return payload


def mean_metric(results: list[dict[str, Any]], name: str) -> float | None:
    values = [item["metrics"].get(name) for item in results]
    available = [float(value) for value in values if value is not None]
    return round(sum(available) / len(available), 4) if available else None


def has_metric(result: dict[str, Any], name: str) -> bool:
    return result["metrics"].get(name) is not None


def aggregate_retrieval(results: list[dict[str, Any]]) -> dict[str, float | int | None]:
    answerable = [item for item in results if item.get("cohort") == "answerable"]
    aggregate: dict[str, Any] = {
        "evaluated_cases": len(answerable),
        **{f"mean_{name}": mean_metric(answerable, name) for name in RETRIEVAL_METRIC_NAMES},
    }
    if any(has_metric(item, CANDIDATE_METRIC_NAMES[0]) for item in answerable):
        aggregate["candidate_pool"] = {
            f"mean_{name}": mean_metric(answerable, name) for name in CANDIDATE_METRIC_NAMES
        }
    return aggregate


def aggregate_status(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_cohort: dict[str, Any] = {}
    for cohort in ("answerable", "insufficient_evidence", "blocked"):
        cohort_results = [item for item in results if item.get("cohort") == cohort]
        if cohort_results:
            by_cohort[cohort] = {
                "evaluated_cases": len(cohort_results),
                "accuracy": mean_metric(cohort_results, "status_correct"),
            }
    labels = tuple(status.value for status in AnswerStatus)
    confusion_matrix = {
        expected: {
            predicted: sum(
                item.get("expected_status") == expected and item.get("status") == predicted
                for item in results
            )
            for predicted in (*labels, "error")
        }
        for expected in labels
    }
    classification: dict[str, dict[str, float | int]] = {}
    for label in (AnswerStatus.BLOCKED.value, AnswerStatus.INSUFFICIENT_EVIDENCE.value):
        true_positive = sum(
            item.get("expected_status") == label and item.get("status") == label for item in results
        )
        false_positive = sum(
            item.get("expected_status") != label and item.get("status") == label for item in results
        )
        false_negative = sum(
            item.get("expected_status") == label and item.get("status") != label for item in results
        )
        classification[label] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": round(true_positive / (true_positive + false_positive), 4)
            if true_positive + false_positive
            else 0.0,
            "recall": round(true_positive / (true_positive + false_negative), 4)
            if true_positive + false_negative
            else 0.0,
        }
    non_answerable = [
        item for item in results if item.get("expected_status") != AnswerStatus.ANSWERED
    ]
    answerable = [item for item in results if item.get("expected_status") == AnswerStatus.ANSWERED]
    return {
        "evaluated_cases": len(results),
        "accuracy": mean_metric(results, "status_correct"),
        "by_cohort": by_cohort,
        "confusion_matrix": confusion_matrix,
        "classification": classification,
        "false_answer_rate": round(
            sum(item.get("status") == AnswerStatus.ANSWERED for item in non_answerable)
            / len(non_answerable),
            4,
        )
        if non_answerable
        else None,
        "false_refusal_rate": round(
            sum(item.get("status") != AnswerStatus.ANSWERED for item in answerable)
            / len(answerable),
            4,
        )
        if answerable
        else None,
        "by_refusal_reason": {
            reason: sum(item.get("refusal_reason") == reason for item in results)
            for reason in sorted(
                {item["refusal_reason"] for item in results if item.get("refusal_reason")}
            )
        },
    }


def aggregate_answer(results: list[dict[str, Any]]) -> dict[str, float | int | None]:
    answerable = [item for item in results if item.get("cohort") == "answerable"]
    scored = [item for item in answerable if has_metric(item, "required_fact_recall")]
    semantically_judged = [
        item
        for item in answerable
        if item.get("status") == AnswerStatus.ANSWERED
        and item.get("fact_group_verdicts") is not None
    ]
    return {
        "evaluated_cases": len(answerable),
        "fact_scored_cases": len(scored),
        "semantic_judged_cases": len(semantically_judged),
        "non_answered_zero_scored_cases": sum(
            item.get("status") != AnswerStatus.ANSWERED for item in answerable
        ),
        "fact_judge_error_count": sum(
            bool(item["metrics"].get("fact_judge_error")) for item in answerable
        ),
        "mean_required_fact_recall": mean_metric(answerable, "required_fact_recall"),
        "complete_answer_rate": mean_metric(answerable, "complete_answer"),
    }


def aggregate_citations(results: list[dict[str, Any]]) -> dict[str, float | int | None]:
    answered = [item for item in results if item.get("status") == AnswerStatus.ANSWERED]
    recall_evaluated = [item for item in answered if has_metric(item, "cited_article_recall")]
    return {
        "answered_cases": len(answered),
        "recall_evaluated_cases": len(recall_evaluated),
        "structural_validity_rate": mean_metric(answered, "citation_structurally_valid"),
        "mean_cited_article_recall": mean_metric(answered, "cited_article_recall"),
        "mean_cited_evidence_recall": mean_metric(answered, "cited_evidence_recall"),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "retrieval": aggregate_retrieval(results),
        "status": aggregate_status(results),
        "answer": aggregate_answer(results),
        "citations": aggregate_citations(results),
        "reliability": {
            "crash_count": sum(item.get("status") == "error" for item in results),
            "malformed_count": sum(item.get("status") == "malformed" for item in results),
        },
        "mean_pipeline_latency_seconds": mean_metric(results, "latency_seconds"),
    }


def aggregate_by_split(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, Any] = {}
    for split in ("development", "test"):
        split_results = [item for item in results if item.get("split") == split]
        if split_results:
            by_split[split] = aggregate(split_results)
    return by_split


def citation_metrics(case: dict[str, Any], result: PipelineResult) -> dict[str, float | None]:
    if result.status is not AnswerStatus.ANSWERED:
        return {
            "citation_structurally_valid": None,
            "cited_article_recall": None,
            "cited_evidence_recall": None,
        }

    source_ids = {source.source_id for source in result.sources}
    citation_valid = bool(result.citations and result.sources) and all(
        citation.source_id in source_ids for citation in result.citations
    )
    expected_articles = set(case.get("expected_articles", []))
    expected_evidence = set(case.get("evidence_chunk_ids", []))
    cited_articles = {str(source.chunk.article_number) for source in result.sources}
    cited_evidence = {source.chunk.chunk_id for source in result.sources}
    return {
        "citation_structurally_valid": float(citation_valid),
        "cited_article_recall": (
            len(expected_articles & cited_articles) / len(expected_articles)
            if expected_articles
            else None
        ),
        "cited_evidence_recall": (
            len(expected_evidence & cited_evidence) / len(expected_evidence)
            if expected_evidence
            else None
        ),
    }


async def fact_metrics(
    case: dict[str, Any],
    result: PipelineResult,
    judge: FactJudge | None,
) -> tuple[dict[str, float | bool | None], list[dict[str, Any]] | None]:
    if case.get("cohort", "answerable") != "answerable":
        return {
            "required_fact_recall": None,
            "complete_answer": None,
            "fact_judge_error": False,
        }, None
    fact_groups = case.get("required_fact_groups", [])
    if result.status is not AnswerStatus.ANSWERED:
        return {
            "required_fact_recall": 0.0,
            "complete_answer": 0.0,
            "fact_judge_error": False,
        }, []
    if judge is None or not fact_groups:
        return {
            "required_fact_recall": None,
            "complete_answer": None,
            "fact_judge_error": False,
        }, None

    verdict = await judge.evaluate(
        question=case["question"],
        answer=result.answer,
        fact_groups=fact_groups,
    )
    if verdict is None:
        return {
            "required_fact_recall": None,
            "complete_answer": None,
            "fact_judge_error": True,
        }, None
    covered = sum(item.covered for item in verdict.groups)
    recall = covered / len(fact_groups)
    return {
        "required_fact_recall": recall,
        "complete_answer": float(covered == len(fact_groups)),
        "fact_judge_error": False,
    }, [item.model_dump() for item in verdict.groups]


async def evaluate_case(
    case: dict[str, Any],
    container: AppContainer,
    fact_judge: FactJudge | None,
) -> dict[str, Any]:
    """Evaluate one case from a single pipeline execution."""

    started = time.monotonic()
    pipeline_result: PipelineResult | None = None
    async for event in container.chat.pipeline.stream(case["question"], []):
        if isinstance(event, ResultEvent):
            pipeline_result = event.result
    if pipeline_result is None:
        raise ValueError("RAG pipeline completed without a result event")

    evidence_ids = [item.chunk.chunk_id for item in pipeline_result.evidence]
    articles = unique_ranked([str(item.chunk.article_number) for item in pipeline_result.evidence])
    metrics: dict[str, Any] = case_retrieval_metrics(
        case,
        article_ranking=articles,
        evidence_ranking=evidence_ids,
    )
    candidate_evidence = (
        pipeline_result.retrieval.candidates
        if pipeline_result.retrieval is not None
        else pipeline_result.evidence
    )
    metrics.update(candidate_metrics(case, candidate_evidence))
    metrics.update(
        status_correct=float(pipeline_result.status.value == case["expected_status"]),
        latency_seconds=round(time.monotonic() - started, 3),
        **citation_metrics(case, pipeline_result),
    )
    facts, fact_verdicts = await fact_metrics(case, pipeline_result, fact_judge)
    metrics.update(facts)

    return {
        "id": case["id"],
        "split": case.get("split", "legacy"),
        "cohort": case.get("cohort", "answerable"),
        "category": case["category"],
        "question": case["question"],
        "expected_status": case["expected_status"],
        "status": pipeline_result.status.value,
        "refusal_reason": (
            pipeline_result.refusal_reason.value if pipeline_result.refusal_reason else None
        ),
        "query_analysis": (
            pipeline_result.analysis.model_dump() if pipeline_result.analysis is not None else None
        ),
        "retrieved_chunk_ids": evidence_ids,
        "retrieved_articles": articles,
        "candidate_chunk_ids": [item.chunk.chunk_id for item in candidate_evidence],
        "candidate_articles": unique_ranked(
            [str(item.chunk.article_number) for item in candidate_evidence]
        ),
        "ranked_lists": retrieval_trace_payload(case, pipeline_result.retrieval),
        "cited_chunk_ids": [source.chunk.chunk_id for source in pipeline_result.sources],
        "cited_articles": unique_ranked(
            [str(source.chunk.article_number) for source in pipeline_result.sources]
        ),
        "answer_preview": pipeline_result.answer[:200],
        "validation_score": pipeline_result.validation_score,
        "fact_group_verdicts": fact_verdicts,
        "metrics": metrics,
    }


def baseline_case(case: dict[str, Any], index: StaticCorpusIndex) -> dict[str, Any]:
    hits = index.bm25_search(case["question"], top_k=RETRIEVAL_CUTOFF)
    chunk_ids = [hit.chunk.chunk_id for hit in hits]
    articles = unique_ranked([str(hit.chunk.article_number) for hit in hits])
    return {
        "id": case["id"],
        "split": case.get("split", "legacy"),
        "cohort": case.get("cohort", "answerable"),
        "category": case["category"],
        "question": case["question"],
        "retrieved_chunk_ids": chunk_ids,
        "retrieved_articles": articles,
        "metrics": case_retrieval_metrics(
            case,
            article_ranking=articles,
            evidence_ranking=chunk_ids,
        ),
    }


def archive_legacy_baseline(output: Path) -> None:
    if not output.exists():
        return
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload.get("schema_version") == REPORT_SCHEMA_VERSION:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_at = payload.get("run_at", "unknown").replace(":", "").replace("+", "-")
    archive = RESULTS_DIR / f"baseline-bm25-schema1-{run_at}.json"
    if not archive.exists():
        shutil.copyfile(output, archive)


def write_bm25_baseline(
    limit: int | None,
    category: str | None,
    split: str | None = None,
    cohort: str | None = None,
) -> Path:
    """Write a reproducible lexical baseline for answerable cases only."""

    settings = Settings()
    suite = load_ground_truth_payload()
    selected = load_ground_truth(limit, category, split, cohort)
    cases = [case for case in selected if case.get("cohort", "answerable") == "answerable"]
    index = StaticCorpusIndex(
        settings.corpus_dir,
        EmbeddingIdentity(
            provider=settings.embedding_provider,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        ),
    )
    results = [baseline_case(case, index) for case in cases]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_at": datetime.now(UTC).isoformat(),
        "mode": "bm25_only_secret_free_baseline",
        "release_gate": False,
        "suite_total_cases": len(selected),
        "evaluated_cases": len(results),
        "dataset_version": suite.get("dataset_version"),
        "corpus_version": index.version,
        "configuration": {
            "query": "original_question",
            "bm25_search_top_k": RETRIEVAL_CUTOFF,
        },
        "aggregate": {"retrieval": aggregate_retrieval(results)},
        "by_split": {
            split: {
                "retrieval": aggregate_retrieval(
                    [item for item in results if item["split"] == split]
                )
            }
            for split in ("development", "test")
            if any(item["split"] == split for item in results)
        },
        "results": results,
    }
    output = EVAL_DIR / "baseline-bm25.json"
    archive_legacy_baseline(output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def error_result(case: dict[str, Any], exc: Exception) -> dict[str, Any]:
    retrieval = (
        {name: 0.0 for name in RETRIEVAL_METRIC_NAMES}
        if case.get("cohort", "answerable") == "answerable"
        else {name: None for name in RETRIEVAL_METRIC_NAMES}
    )
    is_answerable = case.get("cohort", "answerable") == "answerable"
    return {
        "id": case["id"],
        "split": case.get("split", "legacy"),
        "cohort": case.get("cohort", "answerable"),
        "category": case["category"],
        "question": case["question"],
        "expected_status": case["expected_status"],
        "status": "error",
        "refusal_reason": None,
        "retrieved_chunk_ids": [],
        "retrieved_articles": [],
        "cited_chunk_ids": [],
        "cited_articles": [],
        "answer_preview": "",
        "fact_group_verdicts": None,
        "metrics": {
            **retrieval,
            "status_correct": 0.0,
            "required_fact_recall": 0.0 if is_answerable else None,
            "complete_answer": 0.0 if is_answerable else None,
            "fact_judge_error": False,
            "citation_structurally_valid": None,
            "cited_article_recall": None,
            "cited_evidence_recall": None,
            "latency_seconds": 0.0,
        },
        "error": type(exc).__name__,
    }


async def run(
    limit: int | None,
    category: str | None,
    *,
    fact_judge_enabled: bool = True,
    split: str | None = None,
    cohort: str | None = None,
) -> Path:
    settings = Settings()
    if not settings.openai_key:
        raise RuntimeError("OPENAI_API_KEY is required for the manual evaluation")

    suite = load_ground_truth_payload()
    provider = OpenAIProvider(settings)
    container = await AppContainer.build(settings, provider=provider)
    fact_judge = (
        FactJudge(
            provider.create_chat_model(
                model=settings.output_validation_model,
                reasoning_effort=settings.analysis_reasoning_effort,
            )
        )
        if fact_judge_enabled
        else None
    )
    cases = load_ground_truth(limit, category, split, cohort)
    results: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(cases, 1):
            logger.info("Evaluating {}/{}: {}", index, len(cases), case["id"])
            try:
                results.append(await evaluate_case(case, container, fact_judge))
            except Exception as exc:
                logger.opt(exception=exc).error("Evaluation case failed: {}", case["id"])
                results.append(error_result(case, exc))
    finally:
        await container.close()

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_at": datetime.now(UTC).isoformat(),
        "mode": "real_provider",
        "total_cases": len(results),
        "selection": {"split": split, "cohort": cohort, "category": category},
        "dataset_version": suite.get("dataset_version"),
        "corpus_version": container.corpus.version,
        "configuration": {
            "chat_model": settings.chat_model,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
            "analysis_reasoning_effort": settings.analysis_reasoning_effort,
            "generation_reasoning_effort": settings.generation_reasoning_effort,
            "reranker_enabled": settings.reranker_enabled,
            "reranker_candidate_top_k": settings.reranker_candidate_top_k,
            "retrieval_profile": settings.retrieval_profile,
            "vector_search_top_k": settings.vector_search_top_k,
            "bm25_search_top_k": settings.bm25_search_top_k,
            "fusion_top_k": settings.fusion_top_k,
            "final_evidence_top_k": settings.final_evidence_top_k,
            "fact_judge_enabled": fact_judge_enabled,
            "fact_judge_model": settings.output_validation_model if fact_judge_enabled else None,
        },
        "aggregate": aggregate(results),
        "by_split": aggregate_by_split(results),
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"eval-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category")
    parser.add_argument("--split", choices=("development", "test"))
    parser.add_argument("--cohort", choices=("answerable", "insufficient_evidence", "blocked"))
    parser.add_argument(
        "--bm25-baseline",
        action="store_true",
        help="Run the reproducible, secret-free BM25 retrieval baseline",
    )
    parser.add_argument(
        "--skip-fact-judge",
        action="store_true",
        help="Run the real-provider suite without semantic required-fact evaluation",
    )
    args = parser.parse_args()
    output = (
        write_bm25_baseline(args.limit, args.category, args.split, args.cohort)
        if args.bm25_baseline
        else asyncio.run(
            run(
                args.limit,
                args.category,
                fact_judge_enabled=not args.skip_fact_judge,
                split=args.split,
                cohort=args.cohort,
            )
        )
    )
    print(f"Evaluation report written to {output}")


if __name__ == "__main__":
    main()
