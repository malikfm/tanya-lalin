"""Shared deterministic fixtures for backend tests."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from app.rag.models import CorpusChunk, Evidence


@pytest.fixture
def chunk() -> CorpusChunk:
    return CorpusChunk(
        chunk_id="uu-22-2009:art287_p2_body",
        document_id="uu-22-2009-llaj",
        document_title="Law No. 22 of 2009 on Road Traffic and Transportation",
        source="UU_22_2009_LLAJ",
        article_number=287,
        paragraph_number=2,
        chunk_type="body",
        text=(
            "Setiap orang yang mengemudikan Kendaraan Bermotor di Jalan yang melanggar "
            "aturan Alat Pemberi Isyarat Lalu Lintas dipidana dengan pidana kurungan "
            "paling lama dua bulan atau denda paling banyak Rp500.000."
        ),
        official_url="https://peraturan.bpk.go.id/Details/38654/Uu-No-22-Tahun-2009",
        last_verified_at=date(2026, 8, 27),
    )


@pytest.fixture
def evidence(chunk: CorpusChunk) -> list[Evidence]:
    return [Evidence(source_id="S1", chunk=chunk, fusion_score=0.1, vector_relevance=0.9)]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def corpus_factory(tmp_path: Path):
    def create(
        chunks: list[CorpusChunk],
        vectors: np.ndarray,
        *,
        provider: str = "test-provider",
        model: str = "test-embedding",
    ) -> Path:
        corpus_dir = tmp_path / f"corpus-{len(list(tmp_path.iterdir()))}"
        corpus_dir.mkdir()
        chunks_path = corpus_dir / "chunks.jsonl"
        chunks_path.write_text(
            "".join(chunk.model_dump_json() + "\n" for chunk in chunks),
            encoding="utf-8",
        )
        vectors_path = corpus_dir / "vectors.npy"
        np.save(vectors_path, vectors.astype(np.float32), allow_pickle=False)
        manifest = {
            "schema_version": 2,
            "corpus_version": "test-v1",
            "embedding_provider": provider,
            "embedding_model": model,
            "embedding_dimension": int(vectors.shape[1]),
            "chunk_count": len(chunks),
            "chunks_sha256": _digest(chunks_path),
            "vectors_sha256": _digest(vectors_path),
        }
        (corpus_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return corpus_dir

    return create
