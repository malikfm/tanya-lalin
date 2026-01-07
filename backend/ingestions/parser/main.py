"""Main module for parsing legal PDF documents into structured data.

Handles the extraction and parsing of Indonesian legal PDF documents,
specifically separating the main body content from elucidation sections. It uses
PyMuPDF for PDF text extraction and applies regex-based parsing rules to identify
document structure including chapters, sections, articles, and paragraphs.

The parsing process:
1. Extracts text from specified page ranges of a PDF document.
2. Applies parsing rules to identify structural elements.
3. Separates body content from elucidation content.
4. Validates parsed results for completeness and consistency.
5. Outputs structured data to JSONL files.
"""
import argparse
from pathlib import Path

import pymupdf

from enums import LegalTextType
from ingestions.models import ParsingRules
from ingestions.io import write_to_jsonl
from ingestions.parser.core import LegalPDFParser
from ingestions.parser.pdf_patterns import (
    PAGE_PATTERN,
    CHAPTER_PATTERN,
    SECTION_PATTERN,
    SUBSECTION_PATTERN,
    ARTICLE_PATTERN,
    ARTICLE_WO_NUMBER_PATTERN,
    BODY_PARAGRAPH_PATTERN,
    ELUCIDATION_PARAGRAPH_PATTERN,
    BODY_ORDERED_LIST_PATTERN,
    ELUCIDATION_ORDERED_LIST_PATTERN,
    TRIPPLE_DOTS_PATTERN,
    TRIPPLE_DOT_SPACES_PATTERN,
    END_OF_BODY_MARKER,
    END_OF_ELUCIDATION_MARKER
)
from ingestions.parser.validation import validate_result
from logging_setup import setup_logger

logger = setup_logger()


def main() -> int:
    args_parser = argparse.ArgumentParser(
        description="Parse a legal PDF into body/elucidation JSON line outputs."
    )
    args_parser.add_argument(
        "--input-pdf-path", 
        required=True, 
        help="Path to the input PDF file to parse."
    )
    args_parser.add_argument(
        "--output-dir", 
        required=True, 
        help="Directory where output JSONL files will be saved."
    )
    args_parser.add_argument(
        "--body-start", 
        required=True, 
        type=int, 
        help="Starting page for body content."
    )
    args_parser.add_argument(
        "--body-end", 
        required=True, 
        type=int, 
        help="Ending page for body content."
    )
    args_parser.add_argument(
        "--elucidation-start", 
        required=True, 
        type=int, 
        help="Starting page for elucidation content."
    )
    args_parser.add_argument(
        "--header-lines-to-skip", 
        required=True, 
        type=int, 
        help="Number of header lines to skip on each page before parsing."
    )
    args_parser.add_argument(
        "--expected-total-articles", 
        required=True, 
        type=int, 
        help="Expected total number of articles in the document."
    )
    args = args_parser.parse_args()

    pdf_path = Path(args.input_pdf_path)
    output_dir = Path(args.output_dir)

    body_start = args.body_start - 1  # inclusive
    body_end = args.body_end  # exclusive
    elucidation_start = args.elucidation_start - 1  # inclusive

    document = pymupdf.open(pdf_path)
    pages = [page.get_text() for page in document]
    pages = [[line.strip() for line in page.splitlines() if line.strip()] for page in pages]
    body_pages = pages[body_start:body_end]
    elucidation_pages = pages[elucidation_start:]

    body_parsing_rules = ParsingRules(
        header_lines_to_skip=args.header_lines_to_skip,
        end_marker=END_OF_BODY_MARKER,
        paragraph_pattern=BODY_PARAGRAPH_PATTERN,
        ordered_list_pattern=BODY_ORDERED_LIST_PATTERN,
        article_pattern=ARTICLE_PATTERN,
        article_wo_number_pattern=ARTICLE_WO_NUMBER_PATTERN,
        section_marker_patterns=[CHAPTER_PATTERN, SECTION_PATTERN, SUBSECTION_PATTERN],
        skip_patterns=[PAGE_PATTERN, TRIPPLE_DOTS_PATTERN, TRIPPLE_DOT_SPACES_PATTERN]
    )
    elucidation_parsing_rules = ParsingRules(
        header_lines_to_skip=args.header_lines_to_skip,
        end_marker=END_OF_ELUCIDATION_MARKER,
        paragraph_pattern=ELUCIDATION_PARAGRAPH_PATTERN,
        ordered_list_pattern=ELUCIDATION_ORDERED_LIST_PATTERN,
        article_pattern=ARTICLE_PATTERN,
        article_wo_number_pattern=ARTICLE_WO_NUMBER_PATTERN,
        section_marker_patterns=[CHAPTER_PATTERN, SECTION_PATTERN, SUBSECTION_PATTERN],
        skip_patterns=[PAGE_PATTERN, TRIPPLE_DOTS_PATTERN, TRIPPLE_DOT_SPACES_PATTERN]
    )

    pdf_parser = LegalPDFParser()
    body = pdf_parser.parse(body_pages, body_parsing_rules)
    elucidation = pdf_parser.parse(elucidation_pages, elucidation_parsing_rules)

    body_report = validate_result(
        body,
        LegalTextType.BODY.value,
        expected_total_articles=args.expected_total_articles
    )
    elucidation_report = validate_result(
        elucidation,
        LegalTextType.ELUCIDATION.value,
        expected_total_articles=args.expected_total_articles
    )

    if not body_report.is_valid(args.expected_total_articles):
        logger.error(f"Invalid {LegalTextType.BODY.value}: {body_report}")
        return 1
    if not elucidation_report.is_valid(args.expected_total_articles):
        logger.error(f"Invalid {LegalTextType.ELUCIDATION.value}: {elucidation_report}")
        return 1

    logger.info(f"All is good, saving to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_to_jsonl(map(lambda legal_doc_item: legal_doc_item.model_dump_json(), body), output_dir.joinpath(f"{LegalTextType.BODY.value}.jsonl"))
    write_to_jsonl(map(lambda legal_doc_item: legal_doc_item.model_dump_json(), elucidation), output_dir.joinpath(f"{LegalTextType.ELUCIDATION.value}.jsonl"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
