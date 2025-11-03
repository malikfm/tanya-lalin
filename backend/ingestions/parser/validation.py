from collections import defaultdict
from typing import List, Tuple

from loguru import logger

from ingestions.models import ParsingValidationReport
from ingestions.parser.core import LegalDocumentItem


def check_total_articles(legal_document: List[LegalDocumentItem]) -> int:
    """Count the total number of unique articles in the legal document.
    
    This function counts unique articles by article number, ignoring
    any None values which might represent parsing artifacts.
    
    Args:
        legal_document (List[LegalDocumentItem]): List of parsed legal document items.
        
    Returns:
        int: Total count of unique articles.
    """
    unique_articles = {item.article_number for item in legal_document if item}
    return len(unique_articles)


def check_missing_articles(
    legal_document: List[LegalDocumentItem],
    expected_total_articles: int
) -> List[int]:
    """Check for missing articles in the legal document.
    
    This function compares the articles present in the parsed document
    against the expected range of articles (1 to expected_total_articles)
    and returns a list of any missing article numbers.
    
    Args:
        legal_document (List[LegalDocumentItem]): List of parsed legal document items.
        expected_total_articles (int): Expected total number of articles.
        
    Returns:
        List[int]: Sorted list of missing article numbers.
    """
    actual = {item.article_number for item in legal_document if item}
    expected = set(range(1, expected_total_articles + 1))
    missing = expected - actual
    return sorted(missing)


def check_missing_paragraphs(legal_document: List[LegalDocumentItem]) -> dict:
    """Check for missing paragraphs within articles.
    
    This function examines each article with paragraphs to ensure
    sequential numbering without gaps. It identifies any missing
    paragraph numbers within the expected range for each article.
    
    Args:
        legal_document (List[LegalDocumentItem]): List of parsed legal document items.
        
    Returns:
        dict: Dictionary mapping article numbers to lists of missing paragraph numbers.
        
    Raises:
        ValueError: If no articles with paragraphs are found.
    """
    article_to_paragraphs = defaultdict(list)
    for item in legal_document:
        if item and item.paragraph_number:
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
    """Check for duplicate article-paragraph combinations.
    
    This function identifies any duplicate combinations of article number
    and paragraph number, which would indicate parsing errors where the
    same content was parsed multiple times.
    
    Args:
        legal_document (List[LegalDocumentItem]): List of parsed legal document items.
        
    Returns:
        List[Tuple[int, int]]: List of duplicate (article_number, paragraph_number) tuples.
    """
    counts = {}
    for item in legal_document:
        if item:
            key = (item.article_number, 0 if item.paragraph_number is None else item.paragraph_number)
            counts[key] = counts.get(key, 0) + 1
    duplicates = [key for key, count in counts.items() if count > 1]
    
    return sorted(duplicates)


def validate_result(
    legal_document: List[LegalDocumentItem],
    legal_text_type: str,
    expected_total_articles: int
) -> ParsingValidationReport:
    """Validate the parsed legal document and generate a comprehensive validation report.
    
    This function performs all validation checks on the parsed legal document
    and returns a ParsingValidationReport containing the results. It checks for:
    1. Correct total article count
    2. Missing articles
    3. Missing paragraphs within articles
    4. Duplicate article-paragraph combinations
    
    Args:
        legal_document (List[LegalDocumentItem]): List of parsed legal document items.
        legal_text_type (str): Type of legal text ('body' or 'elucidation').
        expected_total_articles (int): Expected total number of articles.
        
    Returns:
        ParsingValidationReport: Comprehensive validation results.
    """
    total_articles = check_total_articles(legal_document)
    missing_articles = check_missing_articles(legal_document, expected_total_articles)
    missing_paragraphs = check_missing_paragraphs(legal_document)
    duplicate_article_paragraphs = check_duplicate_article_paragraphs(legal_document)

    logger.info(f"Validating {legal_text_type}...")
    logger.info(f"Total articles in {legal_text_type} text: {total_articles}")
    logger.info(f"Missing articles in {legal_text_type} text: {missing_articles}")
    logger.info(f"Missing paragraphs per article in {legal_text_type} text: {missing_paragraphs}")
    logger.info(f"Duplicate article-paragraphs in {legal_text_type} text: {duplicate_article_paragraphs}")

    return ParsingValidationReport(
        total_articles=total_articles,
        missing_articles=missing_articles,
        missing_paragraphs=missing_paragraphs,
        duplicate_article_paragraphs=duplicate_article_paragraphs,
    )
