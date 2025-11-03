from re import Pattern
from typing import List, Optional, Tuple

from pydantic import BaseModel


class LegalDocumentItem(BaseModel):
    article_number: int
    paragraph_number: Optional[int]
    text: str


class LegalTextChunkDTO(BaseModel):
    source: str
    article_number: int
    paragraph_number: Optional[int]
    chunk_index: int
    chunk_type: str
    text: str
    embedding: List[float]
    token_count: int


class ParsingState(BaseModel):
    article_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    text: str = ""
    in_article_section: bool = False
    prev_was_article_wo_number: bool = False


class ParsingValidationReport(BaseModel):
    total_articles: int
    missing_articles: List[int]
    missing_paragraphs: dict
    duplicate_article_paragraphs: List[Tuple[int, int]]

    def is_valid(self, expected_total_articles: int) -> bool:
        """Check if the validation report indicates a valid parsed document.
        
        A document is considered valid if it contains the expected number
        of articles, has no missing articles or paragraphs, and contains
        no duplicate article-paragraph combinations.
        
        Args:
            expected_total_articles (int): Expected total number of articles.
            
        Returns:
            bool: True if the document passes all validation checks.
        """
        return (
            self.total_articles == expected_total_articles
            and not self.missing_articles
            and not self.missing_paragraphs
            and not self.duplicate_article_paragraphs
        )


class ParsingRules(BaseModel):
    header_lines_to_skip: int
    end_marker: str
    paragraph_pattern: Pattern
    ordered_list_pattern: Pattern
    article_pattern: Pattern
    article_wo_number_pattern: Pattern
    section_marker_patterns: List[Pattern]
    skip_patterns: List[Pattern]
