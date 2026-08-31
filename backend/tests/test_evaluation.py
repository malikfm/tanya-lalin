"""Schema-v2 evaluation metrics and single-pass execution tests."""

from __future__ import annotations

from types import SimpleNamespace

from app.rag.models import (
    AnswerStatus,
    PipelineCitation,
    PipelineResult,
    PipelineSource,
    QueryAnalysis,
    ResultEvent,
    RetrievalResult,
)
from eval.retrieval_experiment import diversity_eligible, reranker_accepted, select_profile
from eval.run_eval import (
    FactCoverageVerdict,
    FactGroupVerdict,
    aggregate_answer,
    aggregate_retrieval,
    aggregate_status,
    candidate_metrics,
    case_retrieval_metrics,
    evaluate_case,
    fact_metrics,
    retrieval_metrics,
)


def answerable_case() -> dict:
    return {
        "id": "eval-test",
        "split": "test",
        "cohort": "answerable",
        "category": "signals",
        "question": "Apa dendanya?",
        "expected_status": "answered",
        "expected_articles": ["287"],
        "evidence_chunk_ids": ["uu-22-2009:art287_p2_body"],
        "required_fact_groups": [["Denda paling banyak Rp500.000"]],
    }


def test_article_metrics_deduplicate_ranked_articles() -> None:
    metrics = retrieval_metrics(["106"], ["106", "106", "106", "999"])
    assert metrics["article_hit_rate_at_5"] == 1.0
    assert metrics["article_precision_at_5"] == 0.5
    assert metrics["article_ndcg_at_5"] == 1.0


def test_empty_expected_articles_are_not_applicable() -> None:
    metrics = retrieval_metrics([], ["106"])
    assert all(value is None for value in metrics.values())


def test_non_answerable_cases_do_not_receive_retrieval_scores() -> None:
    case = {"cohort": "blocked", "expected_articles": [], "evidence_chunk_ids": []}
    metrics = case_retrieval_metrics(
        case,
        article_ranking=["106"],
        evidence_ranking=["chunk-1"],
    )
    assert all(value is None for value in metrics.values())


def test_retrieval_aggregate_uses_answerable_cases_only() -> None:
    scored = case_retrieval_metrics(
        answerable_case(),
        article_ranking=["287"],
        evidence_ranking=["uu-22-2009:art287_p2_body"],
    )
    blocked = {name: None for name in scored}
    aggregate = aggregate_retrieval(
        [
            {"cohort": "answerable", "metrics": scored},
            {"cohort": "blocked", "metrics": blocked},
        ]
    )
    assert aggregate["evaluated_cases"] == 1
    assert aggregate["mean_article_hit_rate_at_5"] == 1.0


def test_candidate_metrics_distinguish_candidate_pool_from_final_evidence(evidence) -> None:
    missed = evidence[0].model_copy(deep=True)
    missed.chunk.article_number = 999
    missed.chunk.chunk_id = "irrelevant"
    metrics = candidate_metrics(answerable_case(), [missed, *evidence])
    assert metrics["candidate_article_hit_rate_at_10"] == 1.0
    assert metrics["candidate_evidence_mrr_at_10"] == 0.5


def experiment_result(
    *, hit: float, recall: float, mrr: float, latency: float = 1.0, hit_at_10: float | None = None
) -> dict:
    return {
        "metrics": {
            "article_hit_rate_at_5": hit,
            "article_hit_rate_at_10": hit if hit_at_10 is None else hit_at_10,
            "evidence_recall_at_5": recall,
            "article_mrr_at_5": mrr,
            "duplicate_article_concentration_at_5": 0.2,
            "retrieval_latency_seconds": latency,
        }
    }


def test_profile_selection_and_diversity_gate_are_deterministic() -> None:
    profiles = {
        "legacy": {
            "aggregate": {
                "mean_article_hit_rate_at_5": 0.7,
                "mean_evidence_recall_at_5": 0.5,
                "mean_article_mrr_at_5": 0.6,
                "mean_retrieval_latency_seconds": 1.0,
            }
        },
        "balanced": {
            "aggregate": {
                "mean_article_hit_rate_at_5": 0.8,
                "mean_evidence_recall_at_5": 0.5,
                "mean_article_mrr_at_5": 0.6,
                "mean_retrieval_latency_seconds": 1.0,
            }
        },
        "balanced_article_diverse": {
            "aggregate": {
                "mean_article_hit_rate_at_5": 0.9,
                "mean_evidence_recall_at_5": 0.5,
                "mean_article_mrr_at_5": 0.6,
                "mean_retrieval_latency_seconds": 1.0,
            }
        },
    }
    assert select_profile(profiles, include_diversity=False) == "balanced"
    assert select_profile(profiles, include_diversity=True) == "balanced_article_diverse"
    eligible, diagnostics = diversity_eligible(
        [experiment_result(hit=0.0, hit_at_10=1.0, recall=0.0, mrr=0.0)]
    )
    assert eligible is True
    assert diagnostics["crowded_recoveries_at_10"] == 1


def test_reranker_acceptance_requires_two_hit_gain_cases_and_bounded_recall_loss() -> None:
    control = [experiment_result(hit=0.0, recall=0.5, mrr=0.0) for _ in range(4)]
    reranked = [
        {**experiment_result(hit=float(index < 2), recall=0.49, mrr=1.0), "reranker_failed": False}
        for index in range(4)
    ]
    assert reranker_accepted(control, reranked) is True


def test_answer_aggregate_distinguishes_semantic_judging_from_zero_scores() -> None:
    aggregate = aggregate_answer(
        [
            {
                "cohort": "answerable",
                "status": "answered",
                "fact_group_verdicts": [{"index": 0, "covered": True}],
                "metrics": {
                    "required_fact_recall": 1.0,
                    "complete_answer": 1.0,
                    "fact_judge_error": False,
                },
            },
            {
                "cohort": "answerable",
                "status": "insufficient_evidence",
                "fact_group_verdicts": [],
                "metrics": {
                    "required_fact_recall": 0.0,
                    "complete_answer": 0.0,
                    "fact_judge_error": False,
                },
            },
        ]
    )
    assert aggregate["fact_scored_cases"] == 2
    assert aggregate["semantic_judged_cases"] == 1
    assert aggregate["non_answered_zero_scored_cases"] == 1


def test_status_aggregate_reports_confusion_and_safe_refusal_error_rates() -> None:
    results = [
        {
            "cohort": "answerable",
            "expected_status": "answered",
            "status": "answered",
            "refusal_reason": None,
            "metrics": {"status_correct": 1.0},
        },
        {
            "cohort": "insufficient_evidence",
            "expected_status": "insufficient_evidence",
            "status": "answered",
            "refusal_reason": None,
            "metrics": {"status_correct": 0.0},
        },
        {
            "cohort": "blocked",
            "expected_status": "blocked",
            "status": "blocked",
            "refusal_reason": "prompt_injection",
            "metrics": {"status_correct": 1.0},
        },
    ]

    aggregate = aggregate_status(results)

    assert aggregate["false_answer_rate"] == 0.5
    assert aggregate["false_refusal_rate"] == 0.0
    assert aggregate["classification"]["blocked"]["precision"] == 1.0
    assert aggregate["confusion_matrix"]["insufficient_evidence"]["answered"] == 1
    assert aggregate["by_refusal_reason"] == {"prompt_injection": 1}


class FakeFactJudge:
    def __init__(self, verdict):
        self.verdict = verdict

    async def evaluate(self, **kwargs):
        return self.verdict


async def test_fact_metrics_accept_semantic_judge_coverage() -> None:
    result = PipelineResult(status=AnswerStatus.ANSWERED, answer="Maksimal lima ratus ribu.")
    verdict = FactCoverageVerdict(groups=[FactGroupVerdict(index=0, covered=True)])
    metrics, groups = await fact_metrics(answerable_case(), result, FakeFactJudge(verdict))
    assert metrics["required_fact_recall"] == 1.0
    assert metrics["complete_answer"] == 1.0
    assert groups == [{"index": 0, "covered": True}]


async def test_fact_metrics_report_judge_failure_without_perfect_score() -> None:
    result = PipelineResult(status=AnswerStatus.ANSWERED, answer="Jawaban")
    metrics, groups = await fact_metrics(answerable_case(), result, FakeFactJudge(None))
    assert metrics["required_fact_recall"] is None
    assert metrics["complete_answer"] is None
    assert metrics["fact_judge_error"] is True
    assert groups is None


class CountingPipeline:
    def __init__(self, result):
        self.calls = 0
        self.result = result

    async def stream(self, query, history):
        self.calls += 1
        yield ResultEvent(result=self.result)


async def test_evaluate_case_uses_one_pipeline_execution(evidence) -> None:
    pipeline_result = PipelineResult(
        status=AnswerStatus.ANSWERED,
        answer="Denda paling banyak Rp500.000 [S1].",
        citations=[PipelineCitation(citation_id="C1", marker="[S1]", source_id="S1")],
        sources=[PipelineSource(source_id="S1", chunk=evidence[0].chunk)],
        analysis=QueryAnalysis(
            standalone_query="Apa dendanya?",
            legal_query="denda pelanggaran APILL",
        ),
        evidence=evidence,
        retrieval=RetrievalResult(
            fused_candidates=evidence,
            candidates=evidence,
        ),
        evidence_count=1,
        validation_score=5,
    )
    pipeline = CountingPipeline(pipeline_result)
    container = SimpleNamespace(chat=SimpleNamespace(pipeline=pipeline))
    result = await evaluate_case(answerable_case(), container, None)
    assert pipeline.calls == 1
    assert result["retrieved_chunk_ids"] == ["uu-22-2009:art287_p2_body"]
    assert result["metrics"]["citation_structurally_valid"] == 1.0
    assert result["metrics"]["cited_evidence_recall"] == 1.0
    assert result["metrics"]["candidate_article_hit_rate_at_10"] == 1.0
    assert "retrieval" not in pipeline_result.model_dump()
