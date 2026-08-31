"""Verify the integrity and runtime compatibility of bundled corpus artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import REPOSITORY_ROOT, Settings
from app.infrastructure.ai import EmbeddingIdentity
from app.infrastructure.corpus_index import StaticCorpusIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=REPOSITORY_ROOT / "corpus")
    return parser.parse_args()


def main() -> None:
    """Load the exact runtime index and print a concise verification result."""

    args = parse_args()
    settings = Settings(corpus_dir=args.corpus_dir)
    identity = EmbeddingIdentity(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    index = StaticCorpusIndex(args.corpus_dir.resolve(), identity)
    print(
        f"Corpus {index.version} verified: "
        f"{len(index.chunks)} chunks, {index.embedding_dimension} dimensions."
    )


if __name__ == "__main__":
    main()
