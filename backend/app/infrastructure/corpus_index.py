"""Versioned static corpus index with exact vector and BM25 search."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from app.infrastructure.ai import EmbeddingIdentity
from app.rag.models import CorpusChunk, SearchHit

TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


class CorpusIntegrityError(RuntimeError):
    """Raised when bundled corpus artifacts are missing or inconsistent."""


def tokenize(text: str) -> list[str]:
    """Return deterministic Unicode-aware lowercase tokens."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return TOKEN_PATTERN.findall(normalized)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class StaticCorpusIndex:
    """Load and query immutable corpus artifacts bundled with the application."""

    def __init__(self, corpus_dir: Path, expected_embedding: EmbeddingIdentity):
        self.corpus_dir = corpus_dir
        self.manifest = self._load_manifest(corpus_dir / "manifest.json")
        self.chunks = self._load_chunks(corpus_dir / "chunks.jsonl")
        self.vectors = self._load_vectors(corpus_dir / "vectors.npy")
        self._validate(expected_embedding)
        self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in self.chunks])

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise CorpusIntegrityError(f"Corpus manifest is missing: {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CorpusIntegrityError(f"Corpus manifest is invalid: {exc}") from exc

    @staticmethod
    def _load_chunks(path: Path) -> list[CorpusChunk]:
        if not path.is_file():
            raise CorpusIntegrityError(f"Corpus chunks are missing: {path}")
        chunks: list[CorpusChunk] = []
        last_line_number = 0
        try:
            with path.open(encoding="utf-8") as source:
                for current_line_number, line in enumerate(source, 1):
                    last_line_number = current_line_number
                    if line.strip():
                        chunks.append(CorpusChunk.model_validate_json(line))
        except Exception as exc:
            raise CorpusIntegrityError(
                f"Corpus chunks are invalid at or before line {last_line_number}: {exc}"
            ) from exc
        return chunks

    @staticmethod
    def _load_vectors(path: Path) -> np.ndarray:
        if not path.is_file():
            raise CorpusIntegrityError(f"Corpus vectors are missing: {path}")
        try:
            vectors = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise CorpusIntegrityError(f"Corpus vectors are invalid: {exc}") from exc
        if vectors.dtype != np.float32 or vectors.ndim != 2:
            raise CorpusIntegrityError("Corpus vectors must be a two-dimensional float32 matrix")
        return vectors

    def _validate(self, expected_embedding: EmbeddingIdentity) -> None:
        chunks_path = self.corpus_dir / "chunks.jsonl"
        vectors_path = self.corpus_dir / "vectors.npy"
        checks = {
            "chunks_sha256": _sha256(chunks_path),
            "vectors_sha256": _sha256(vectors_path),
        }
        for field, actual in checks.items():
            if self.manifest.get(field) != actual:
                raise CorpusIntegrityError(f"Corpus checksum mismatch for {field}")

        if self.manifest.get("schema_version") != 2:
            raise CorpusIntegrityError("Unsupported corpus manifest schema version")

        expected_count = self.manifest.get("chunk_count")
        if expected_count != len(self.chunks) or len(self.chunks) != self.vectors.shape[0]:
            raise CorpusIntegrityError("Corpus chunk and vector counts do not match")

        expected_dimension = self.manifest.get("embedding_dimension")
        if (
            expected_dimension != self.vectors.shape[1]
            or expected_dimension != expected_embedding.dimensions
        ):
            raise CorpusIntegrityError("Corpus embedding dimension does not match the manifest")

        if self.manifest.get("embedding_provider") != expected_embedding.provider:
            raise CorpusIntegrityError(
                "Configured embedding provider does not match the bundled corpus index"
            )

        if self.manifest.get("embedding_model") != expected_embedding.model:
            raise CorpusIntegrityError(
                "Configured embedding model does not match the bundled corpus index"
            )

        ids = [chunk.chunk_id for chunk in self.chunks]
        if len(ids) != len(set(ids)):
            raise CorpusIntegrityError("Corpus contains duplicate chunk IDs")

        for chunk in self.chunks:
            normalized = " ".join(chunk.text.casefold().split()).rstrip(".")
            if not normalized or normalized == "cukup jelas":
                raise CorpusIntegrityError(f"Corpus contains boilerplate chunk {chunk.chunk_id}")

    @property
    def version(self) -> str:
        return str(self.manifest["corpus_version"])

    @property
    def embedding_dimension(self) -> int:
        return int(self.vectors.shape[1])

    def vector_search(
        self,
        embedding: list[float] | np.ndarray,
        *,
        search_name: str,
        top_k: int,
        min_relevance: float,
    ) -> list[SearchHit]:
        query = np.asarray(embedding, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.embedding_dimension:
            raise ValueError(
                f"Query embedding has dimension {query.shape}, expected {self.embedding_dimension}"
            )
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        scores = np.asarray(self.vectors @ (query / norm), dtype=np.float32)
        count = min(top_k, len(scores))
        if count == 0:
            return []
        indices = np.argpartition(scores, -count)[-count:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        hits: list[SearchHit] = []
        for index in indices:
            relevance = float(np.clip(scores[index], 0.0, 1.0))
            if relevance < min_relevance:
                continue
            hits.append(
                SearchHit(
                    chunk=self.chunks[int(index)],
                    search_name=search_name,
                    rank=len(hits) + 1,
                    vector_relevance=relevance,
                )
            )
        return hits

    def bm25_search(self, query: str, top_k: int) -> list[SearchHit]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)
        positive = np.flatnonzero(scores > 0)
        if not len(positive):
            return []
        count = min(top_k, len(positive))
        selected = positive[np.argpartition(scores[positive], -count)[-count:]]
        selected = selected[np.argsort(scores[selected])[::-1]]
        return [
            SearchHit(
                chunk=self.chunks[int(index)],
                search_name="bm25",
                rank=rank,
                bm25_score=float(scores[index]),
            )
            for rank, index in enumerate(selected, 1)
        ]

    def article_search(self, article_number: int, top_k: int) -> list[SearchHit]:
        """Return chunks for an explicitly referenced article in corpus order."""

        matching = [chunk for chunk in self.chunks if chunk.article_number == article_number]
        return [
            SearchHit(
                chunk=chunk,
                search_name=f"article_{article_number}",
                rank=rank,
            )
            for rank, chunk in enumerate(matching[:top_k], 1)
        ]
