"""Rebuild the committed vector matrix from canonical corpus chunks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.config import REPOSITORY_ROOT, Settings
from app.infrastructure.ai.openai import OpenAIProvider
from app.rag.models import CorpusChunk


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_chunks(path: Path) -> list[CorpusChunk]:
    """Load canonical chunks and reject unstable or unusable input."""

    chunks = [
        CorpusChunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [chunk.chunk_id for chunk in chunks]
    if not chunks:
        raise ValueError("Corpus contains no chunks")
    if len(ids) != len(set(ids)):
        raise ValueError("Corpus contains duplicate chunk IDs")
    if ids != sorted(ids):
        raise ValueError("Corpus chunks must be sorted by stable chunk ID")
    for chunk in chunks:
        normalized = " ".join(chunk.text.casefold().split()).rstrip(".")
        if not normalized or normalized == "cukup jelas":
            raise ValueError(f"Corpus contains boilerplate chunk {chunk.chunk_id}")
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=REPOSITORY_ROOT / "corpus")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


async def build() -> None:
    """Embed canonical chunks and atomically replace generated artifacts."""

    args = parse_args()
    settings = Settings(corpus_dir=args.corpus_dir)
    if not settings.openai_key:
        raise RuntimeError("OPENAI_API_KEY is required to rebuild corpus vectors")

    corpus_dir = args.corpus_dir.resolve()
    chunks_path = corpus_dir / "chunks.jsonl"
    vectors_path = corpus_dir / "vectors.npy"
    manifest_path = corpus_dir / "manifest.json"
    chunks = load_chunks(chunks_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    provider = OpenAIProvider(settings)
    client = provider.create_embeddings()
    identity = provider.embedding_identity
    embeddings: list[list[float]] = []
    for offset in range(0, len(chunks), args.batch_size):
        batch = chunks[offset : offset + args.batch_size]
        embeddings.extend(await client.aembed_documents([chunk.text for chunk in batch]))

    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if vectors.ndim != 2 or np.any(norms == 0):
        raise RuntimeError("Embedding provider returned an invalid vector matrix")
    if vectors.shape[1] != identity.dimensions:
        raise RuntimeError(
            f"Embedding provider returned {vectors.shape[1]} dimensions; "
            f"expected {identity.dimensions}"
        )
    vectors /= norms

    temporary_vectors = vectors_path.with_suffix(".tmp.npy")
    np.save(temporary_vectors, vectors, allow_pickle=False)
    temporary_vectors.replace(vectors_path)

    manifest.update(
        schema_version=2,
        corpus_version="2026.08.27-openai.1",
        created_at=datetime.now(UTC).isoformat(),
        embedding_provider=identity.provider,
        embedding_model=identity.model,
        embedding_dimension=int(vectors.shape[1]),
        chunk_count=len(chunks),
        chunks_sha256=sha256(chunks_path),
        vectors_sha256=sha256(vectors_path),
    )
    temporary_manifest = manifest_path.with_suffix(".tmp.json")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    print(f"Built {len(chunks)} normalized vectors with dimension {vectors.shape[1]}.")


def main() -> None:
    """Run the asynchronous provider-backed corpus build."""

    asyncio.run(build())


if __name__ == "__main__":
    main()
