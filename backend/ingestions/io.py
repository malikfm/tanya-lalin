import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence, Optional

from pymupdf import Document

from ingestions.parser_core import LegalDocumentItem


def text_to_chunks(text: str) -> List[str]:
    """Split a full page text to non-empty."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def write_to_jsonl(legal_document: List[LegalDocumentItem], output_path: Path) -> None:
    """Write a JSON line file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in legal_document:
            f.write(json.dumps(asdict(item)) + "\n")


def chunks_by_page_from_pdf(document: Document) -> List[List[str]]:
    """Return chunked text per page."""
    return [text_to_chunks(page.get_text()) for page in document]
