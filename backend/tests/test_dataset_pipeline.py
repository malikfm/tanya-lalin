from __future__ import annotations

import json

from eval.dataset_pipeline import (
    CURATOR_OVERRIDES,
    GENERATOR_MODEL,
    REVIEWER_MODEL,
    Candidate,
    _article_numbers,
    build_generation_requests,
    sanitize_question,
    validate_candidate,
)
from eval.run_eval import retrieval_metrics


def test_sanitization_replaces_deterministic_identifiers() -> None:
    question = "Bagaimana status STNK nomor 123456789012345?"
    sanitized = sanitize_question(question)
    assert sanitized is not None
    assert "123456789012345" not in sanitized
    assert "[REDACTED]" in sanitized


def test_generation_payload_excludes_seed_answers_and_uses_snapshots() -> None:
    from eval.dataset_pipeline import PreparedSeed

    requests = build_generation_requests(
        [
            PreparedSeed(
                seed_id="seed-1",
                question="Apa kewajiban pengemudi?",
                source="hukumonline",
                article_hints=[106],
                evidence_chunk_ids=["c1"],
            )
        ],
        {
            "c1": {
                "chunk_id": "c1",
                "article_number": 106,
                "chunk_type": "body",
                "text": "Wajib penuh konsentrasi.",
            }
        },
    )
    payload = json.dumps(requests, ensure_ascii=False)
    assert '"answer"' not in payload
    assert '"context"' not in payload
    assert requests[0]["body"]["model"] == GENERATOR_MODEL
    assert requests[0]["body"]["store"] is False
    assert requests[-1]["body"]["model"] == GENERATOR_MODEL
    assert REVIEWER_MODEL not in payload


def test_candidate_validation_requires_official_evidence() -> None:
    candidate = Candidate(
        candidate_id="candidate-1",
        cohort="answerable",
        category="helm",
        difficulty="easy",
        question="Apakah helm wajib?",
        expected_status="answered",
        expected_articles=["106"],
        evidence_chunk_ids=["missing"],
        reference_answer="Ya.",
        required_fact_groups=[["wajib"]],
    )
    try:
        validate_candidate(candidate, {})
    except ValueError as exc:
        assert "evidence" in str(exc)
    else:
        raise AssertionError("invalid evidence must be rejected")


def test_retrieval_metrics_distinguish_hit_rate_from_recall() -> None:
    metrics = retrieval_metrics(["106", "291"], ["106", "999", "291"])
    assert metrics["article_hit_rate_at_5"] == 1.0
    assert metrics["article_recall_at_5"] == 1.0
    assert metrics["article_mrr_at_5"] == 1.0


def test_curator_overrides_are_explicit_and_reference_known_chunks() -> None:
    from eval.dataset_pipeline import load_chunks

    chunks = load_chunks()
    assert len(CURATOR_OVERRIDES) == 10
    for override in CURATOR_OVERRIDES.values():
        for chunk_id in override.get("evidence_add", []):
            assert chunk_id in chunks
        for chunk_id in override.get("expected_articles_add", []):
            assert chunk_id in chunks


def test_article_references_are_normalized_for_evaluation() -> None:
    chunks = {
        "c1": {"article_number": 106},
        "c2": {"article_number": 106},
    }
    assert _article_numbers(["c1", "c2", "291"], chunks) == ["106", "291"]
