"""Configuration and bundled corpus integrity tests."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest
from pydantic import ValidationError

from app.config import REPOSITORY_ROOT, Settings
from app.infrastructure.ai import EmbeddingIdentity
from app.infrastructure.corpus_index import CorpusIntegrityError, StaticCorpusIndex, tokenize
from app.rag.models import CorpusChunk


def make_chunk(identifier: str, article: int, text: str) -> CorpusChunk:
    return CorpusChunk(
        chunk_id=identifier,
        document_id="law",
        document_title="Test law",
        source="TEST",
        article_number=article,
        chunk_type="body",
        text=text,
        official_url="https://example.test/law",
        last_verified_at=date(2026, 8, 27),
    )


def identity(*, provider: str = "test-provider", model: str = "test-embedding", dimensions=2):
    return EmbeddingIdentity(provider=provider, model=model, dimensions=dimensions)


def test_settings_validate_ranges_and_paths(tmp_path):
    settings = Settings(
        _env_file=None,
        environment="test",
        corpus_dir=tmp_path,
        cors_origins="http://localhost:5173, https://example.test",
    )
    assert settings.corpus_dir == tmp_path.resolve()
    assert settings.parsed_cors_origins == ["http://localhost:5173", "https://example.test"]
    assert settings.json_logs_enabled is False
    assert Settings(_env_file=None, environment="production").json_logs_enabled is True
    assert (
        Settings(_env_file=None, corpus_dir="corpus").corpus_dir
        == (REPOSITORY_ROOT / "corpus").resolve()
    )
    with pytest.raises(ValidationError):
        Settings(_env_file=None, vector_search_top_k=0)


def test_tokenization_is_unicode_aware():
    assert tokenize("  LALU-LINTAS, Ayat (2)!  ") == ["lalu", "lintas", "ayat", "2"]


def test_exact_vector_and_bm25_search(corpus_factory):
    chunks = [
        make_chunk("a", 1, "helm standar nasional Indonesia"),
        make_chunk("b", 2, "lampu lalu lintas dan denda"),
        make_chunk("c", 3, "surat izin mengemudi"),
    ]
    corpus_dir = corpus_factory(
        chunks,
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
    )
    index = StaticCorpusIndex(corpus_dir, identity())

    vector_hits = index.vector_search([0.9, 0.1], search_name="query", top_k=2, min_relevance=0.5)
    assert [hit.chunk.chunk_id for hit in vector_hits] == ["a"]
    assert vector_hits[0].vector_relevance == pytest.approx(0.99388, rel=1e-4)
    assert index.vector_search([0.0, 0.0], search_name="query", top_k=2, min_relevance=0) == []
    assert index.bm25_search("lampu denda", top_k=2)[0].chunk.chunk_id == "b"
    assert index.article_search(1, top_k=2)[0].chunk.chunk_id == "a"


@pytest.mark.parametrize(
    "problem", ["schema", "provider", "model", "checksum", "count", "dimension"]
)
def test_manifest_compatibility_failures(corpus_factory, problem):
    chunk = make_chunk("a", 1, "aturan lalu lintas")
    corpus_dir = corpus_factory([chunk], np.array([[1.0, 0.0]]))
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = identity()
    if problem == "schema":
        manifest["schema_version"] = 1
    elif problem == "provider":
        expected = identity(provider="different-provider")
    elif problem == "model":
        expected = identity(model="different-model")
    elif problem == "checksum":
        manifest["vectors_sha256"] = "invalid"
    elif problem == "count":
        manifest["chunk_count"] = 3
    else:
        manifest["embedding_dimension"] = 3
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CorpusIntegrityError):
        StaticCorpusIndex(corpus_dir, expected)


def test_missing_and_invalid_corpus_files(tmp_path, corpus_factory):
    with pytest.raises(CorpusIntegrityError, match="manifest is missing"):
        StaticCorpusIndex(tmp_path, identity(model="test"))

    chunk = make_chunk("a", 1, "Cukup jelas.")
    corpus_dir = corpus_factory([chunk], np.array([[1.0, 0.0]]))
    with pytest.raises(CorpusIntegrityError, match="boilerplate"):
        StaticCorpusIndex(corpus_dir, identity())


def test_invalid_query_dimension(corpus_factory):
    chunk = make_chunk("a", 1, "aturan")
    index = StaticCorpusIndex(corpus_factory([chunk], np.array([[1.0, 0.0]])), identity())
    with pytest.raises(ValueError, match="dimension"):
        index.vector_search([1.0], search_name="x", top_k=1, min_relevance=0)
