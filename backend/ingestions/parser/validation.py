from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from loguru._logger import Logger

from ingestions.parser.core import LegalDocumentItem


@dataclass
class ValidationReport:
    total_articles: int
    missing_articles: List[int]
    missing_paragraphs: Dict[int, List[int]]
    duplicate_article_paragraphs: List[Tuple[int, int]]

    def is_valid(self, expected_total_articles: int) -> bool:
        return (
            self.total_articles == expected_total_articles
            and not self.missing_articles
            and not self.missing_paragraphs
            and not self.duplicate_article_paragraphs
        )


def check_total_articles(legal_document: List[LegalDocumentItem]) -> int:
    unique_articles = {item.article_number for item in legal_document if item.article_number}
    return len(unique_articles)


def check_missing_articles(
    legal_document: List[LegalDocumentItem],
    expected_total_articles: int
) -> List[int]:
    actual = {item.article_number for item in legal_document}
    expected = set(range(1, expected_total_articles + 1))
    missing = expected - actual
    return sorted(missing)


def check_missing_paragraphs(legal_document: List[LegalDocumentItem]) -> Dict[int, List[int]]:
    article_to_paragraphs = defaultdict(list)
    for item in legal_document:
        if item.paragraph_number:
            article_to_paragraphs[item.article_number].append(item.paragraph_number)

    if not article_to_paragraphs:
        raise ValueError("No article found. Unable to check missing paragraphs.")

    article_to_missing = defaultdict(list)
    for article_number, paragraph_numbers in article_to_paragraphs.items():
        if not paragraph_numbers:
            continue

        max_paragraph_number = max(paragraph_numbers)
        expected = set(range(1, max_paragraph_number + 1))
        actual = set(paragraph_numbers)
        missing = sorted(expected - actual)
        
        if missing:
            article_to_missing[article_number] = missing

    return dict(sorted(article_to_missing.items()))


def check_duplicate_article_paragraphs(
    legal_document: List[LegalDocumentItem],
) -> List[Tuple[int, int]]:
    counts: Dict[Tuple[int, int], int] = {}
    
    for item in legal_document:
        if item.article_number:
            key = (item.article_number, 0 if item.paragraph_number is None else item.paragraph_number)
            counts[key] = counts.get(key, 0) + 1
    duplicates = [key for key, count in counts.items() if count > 1]
    
    return sorted(duplicates)


def validate_result(
    logger: Logger,
    legal_document: List[LegalDocumentItem],
    legal_text_type: str,
    expected_total_articles: int
) -> ValidationReport:
    logger = logger
    total_articles = check_total_articles(legal_document)
    missing_articles = check_missing_articles(legal_document, expected_total_articles)
    missing_paragraphs = check_missing_paragraphs(legal_document)
    duplicate_article_paragraphs = check_duplicate_article_paragraphs(legal_document)

    logger.debug(f"Validating {legal_text_type}...")
    logger.debug(f"Total articles in {legal_text_type} text: {total_articles}")
    logger.debug(f"Missing articles in {legal_text_type} text: {missing_articles}")
    logger.debug(f"Missing paragraphs per article in {legal_text_type} text: {missing_paragraphs}")
    logger.debug(f"Duplicate article-paragraphs in {legal_text_type} text: {duplicate_article_paragraphs}")

    return ValidationReport(
        total_articles=total_articles,
        missing_articles=missing_articles,
        missing_paragraphs=missing_paragraphs,
        duplicate_article_paragraphs=duplicate_article_paragraphs,
    )
