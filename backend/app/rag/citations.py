"""Deterministic citation extraction and validation."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.rag.models import Evidence, PipelineCitation, PipelineSource

CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


class CitationValidation(BaseModel):
    valid: bool
    citations: list[PipelineCitation] = Field(default_factory=list)
    sources: list[PipelineSource] = Field(default_factory=list)
    reason: str = ""


def validate_citations(answer: str, evidence: list[Evidence]) -> CitationValidation:
    """Resolve markers to evidence and reject missing or unknown references."""

    available = {item.source_id: item for item in evidence}
    markers = [f"S{number}" for number in CITATION_PATTERN.findall(answer)]
    unique_markers = list(dict.fromkeys(markers))
    if not unique_markers:
        return CitationValidation(valid=False, reason="The answer contains no source markers")

    unknown = [marker for marker in unique_markers if marker not in available]
    if unknown:
        return CitationValidation(
            valid=False,
            reason=f"The answer references unknown sources: {', '.join(unknown)}",
        )

    citations = [
        PipelineCitation(
            citation_id=f"C{index}",
            marker=f"[{source_id}]",
            source_id=source_id,
        )
        for index, source_id in enumerate(unique_markers, 1)
    ]
    sources = [
        PipelineSource(source_id=source_id, chunk=available[source_id].chunk)
        for source_id in unique_markers
    ]
    return CitationValidation(valid=True, citations=citations, sources=sources)
