"""Extract traffic-law consultation records from the downloaded Parquet dataset."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

from app.config import REPOSITORY_ROOT

FILTER_TERMS = (
    "Undang-Undang Nomor 22 Tahun 2009",
    "UU LLAJ",
)
DATASET_NAME = "ShoAnn/legalqa_klinik_hukumonline"
SOURCE_FILES = (
    ("train", "train-00000-of-00001.parquet"),
    ("test", "test-00000-of-00001.parquet"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=REPOSITORY_ROOT / "corpus",
        help="Directory containing the downloaded Parquet files and output JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (defaults to corpus/hukumonline_llaj_seed.json).",
    )
    return parser.parse_args()


def extract_records(corpus_dir: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Read the source files in stable order and return matching records."""

    records: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    for split, filename in SOURCE_FILES:
        source_path = corpus_dir / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source Parquet file: {source_path}")
        rows = parquet.read_table(source_path).to_pylist()
        for row_index, row in enumerate(rows):
            answer = row.get("answer")
            if not isinstance(answer, str) or not any(term in answer for term in FILTER_TERMS):
                continue
            records.append(
                {
                    "seed_id": f"hukumonline-llaj-{split}-{row_index:05d}",
                    "source_split": split,
                    "source_file": filename,
                    "source_row_index": row_index,
                    **row,
                }
            )
            split_counts[split] += 1
    return records, split_counts


def build_payload(records: list[dict[str, Any]], split_counts: Counter[str]) -> dict[str, Any]:
    """Build the versioned, provenance-preserving JSON payload."""

    return {
        "schema_version": 1,
        "source_dataset": DATASET_NAME,
        "filter_terms": list(FILTER_TERMS),
        "record_count": len(records),
        "split_counts": {split: split_counts.get(split, 0) for split, _ in SOURCE_FILES},
        "records": records,
    }


def write_atomically(output_path: Path, payload: dict[str, Any]) -> None:
    """Write JSON beside the destination and replace it atomically."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main() -> None:
    """Extract and write the local Hukumonline LLAJ seed dataset."""

    args = parse_args()
    corpus_dir = args.corpus_dir.resolve()
    output_path = (args.output or corpus_dir / "hukumonline_llaj_seed.json").resolve()
    records, split_counts = extract_records(corpus_dir)
    payload = build_payload(records, split_counts)
    write_atomically(output_path, payload)
    print(
        f"Wrote {payload['record_count']} records to {output_path} "
        f"(train={split_counts['train']}, test={split_counts['test']})."
    )


if __name__ == "__main__":
    main()
