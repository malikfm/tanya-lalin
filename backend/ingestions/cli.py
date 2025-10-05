import argparse
from pathlib import Path

import pymupdf

from ingestions.io import chunks_by_page_from_pdf, write_to_jsonl
from ingestions.parser_core import LegalPDFParser, ParsingRules
from ingestions.pdf_patterns import (
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
from ingestions.validation import validate_result
from logging_setup import setup_logger

class LegalTextType:
    BODY = "body"
    ELUDICADION = "elucidation"


def main() -> int:
    args_parser = argparse.ArgumentParser(description="Parse a legal PDF into body/elucidation JSON line outputs.")
    args_parser.add_argument("--input-pdf-path", required=True, help="Input PDF path.")
    args_parser.add_argument("--output-dir", required=True, help="Output directory for JSON line files.")
    args_parser.add_argument("--body-start", required=True, type=int, help="Page start for body (inclusive).")
    args_parser.add_argument("--body-end", required=True, type=int, help="Page end for body (exclusive).")
    args_parser.add_argument("--elucidation-start", required=True, type=int, help="Page start for elucidation (inclusive).")
    args_parser.add_argument("--header-lines-to-skip", required=True, type=int, help="Number of lines to skip before the actual content.")
    args_parser.add_argument("--expected-total-articles", required=True, type=int, help="Actual total articles.")
    args = args_parser.parse_args()

    logger = setup_logger()

    pdf_path = Path(args.input_pdf_path)
    output_dir = Path(args.output_dir)

    pdf_document = pymupdf.open(pdf_path)
    pages = chunks_by_page_from_pdf(pdf_document)
    body_pages = pages[args.body_start:args.body_end]
    elucidation_pages = pages[args.elucidation_start:]

    body_parsing_rules = ParsingRules(
        end_marker=END_OF_BODY_MARKER,
        paragraph_pattern=BODY_PARAGRAPH_PATTERN,
        ordered_list_pattern=BODY_ORDERED_LIST_PATTERN,
        article_pattern=ARTICLE_PATTERN,
        article_wo_number_pattern=ARTICLE_WO_NUMBER_PATTERN,
        section_marker_patterns=[CHAPTER_PATTERN, SECTION_PATTERN, SUBSECTION_PATTERN],
        skip_patterns=[PAGE_PATTERN, TRIPPLE_DOTS_PATTERN, TRIPPLE_DOT_SPACES_PATTERN]
    )
    elucidation_parsing_rules = ParsingRules(
        end_marker=END_OF_ELUCIDATION_MARKER,
        paragraph_pattern=ELUCIDATION_PARAGRAPH_PATTERN,
        ordered_list_pattern=ELUCIDATION_ORDERED_LIST_PATTERN,
        article_pattern=ARTICLE_PATTERN,
        article_wo_number_pattern=ARTICLE_WO_NUMBER_PATTERN,
        section_marker_patterns=[CHAPTER_PATTERN, SECTION_PATTERN, SUBSECTION_PATTERN],
        skip_patterns=[PAGE_PATTERN, TRIPPLE_DOTS_PATTERN, TRIPPLE_DOT_SPACES_PATTERN]
    )

    pdf_parser = LegalPDFParser(logger, args.header_lines_to_skip)
    body = pdf_parser.parse(body_pages, body_parsing_rules)
    elucidation = pdf_parser.parse(elucidation_pages, elucidation_parsing_rules)

    body_report = validate_result(
        logger,
        body,
        LegalTextType.BODY,
        expected_total_articles=args.expected_total_articles
    )
    elucidation_report = validate_result(
        logger,
        elucidation,
        LegalTextType.ELUDICADION,
        expected_total_articles=args.expected_total_articles
    )

    if not body_report.is_valid(args.expected_total_articles):
        logger.error(f"Invalid {LegalTextType.BODY}: {body_report}")
        return 1
    if not elucidation_report.is_valid(args.expected_total_articles):
        logger.error(f"Invalid {LegalTextType.ELUDICADION}: {elucidation_report}")
        return 1

    logger.info(f"All is good, saving to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_to_jsonl(body, output_dir.joinpath(f"{LegalTextType.BODY}.jsonl"))
    write_to_jsonl(elucidation, output_dir.joinpath(f"{LegalTextType.ELUDICADION}.jsonl"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
