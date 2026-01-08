from typing import List

from loguru import logger
from tqdm import tqdm

from .models import LegalDocumentItem, ParsingState, ParsingRules


class LegalPDFParser:
    @staticmethod
    def _validate_article_marker_for_article_wo_number(chunk: str, state: ParsingState) -> bool:
        """Determine whether the current text chunk represents an article number following an article header
        without a number (e.g., "Pasal" followed by "3" on the next line).

        This validation helps handle cases where article markers are split across lines.

        Logic:
            - Treat as a new article only if the chunk is a digit and sequentially follows the previous article number.
            For example:
                - Previous article is 2, chunk must be 3
                - No previous article, chunk must be 1

        Args:
            chunk (str): The current text chunk.
            state (ParseState): The parser state.

        Returns:
            bool: True if the chunk is a valid article marker, False otherwise.
        """
        return (
            chunk.isdigit()
            and (
                (state.article_number is not None and int(chunk) == state.article_number + 1)
                or state.article_number is None and int(chunk) == 1
            )
        )

    @staticmethod
    def _validate_paragraph_marker(paragraph_number_candidate: int, state: ParsingState) -> bool:
        """Validate whether a detected paragraph number indicates the start of a new paragraph section.

        This ensures the parser only treats sequential numbers (e.g., (1), (2), (3))
        as true paragraph markers and ignores accidental matches in the text body.

        Logic:
            - If the current paragraph number is None, candidate must be 1.
            - Otherwise, candidate must equal the previous paragraph number + 1.

        Args:
            paragraph_number_candidate (int): The detected paragraph number.
            state (ParseState): The current parsing state.

        Returns:
            bool: True if the candidate marks a new paragraph, False otherwise.
        """
        return (
            (state.paragraph_number is None and paragraph_number_candidate == 1)
            or (state.paragraph_number is not None and paragraph_number_candidate == state.paragraph_number + 1)
        )

    @staticmethod
    def _append_text(buffer: str, chunk: str, is_ordered_list: bool) -> str:
        """Append text to the buffer with appropriate formatting.
        
        For ordered list items, separates lines with newlines, otherwise joins
        with spaces. This maintains proper formatting for different types of content.
        
        Args:
            buffer (str): Existing text buffer.
            chunk (str): Text chunk to append.
            is_ordered_list (bool): True if the chunk is part of an ordered list.
        
        Returns:
            str: Combined text with appropriate formatting.
        """
        if not chunk:
            return ""
        
        if is_ordered_list:
            return f"{buffer}\n{chunk}" if buffer else chunk
        return f"{buffer} {chunk}" if buffer else chunk

    def _flush_buffer(self, legal_document: List[LegalDocumentItem], state: ParsingState) -> None:
        """Flush the current parsing state to a new LegalDocumentItem and add to document.
        
        This method creates a new LegalDocumentItem from the current state variables
        and appends it to the legal document list if the text is not empty.
        
        Args:
            legal_document (List[LegalDocumentItem]): The list to append the item to.
            state (ParsingState): Current parsing state containing article/paragraph info.
        """
        if state.text.strip():
            logger.debug(f"Flush article: {state.article_number}, paragraph: {state.paragraph_number}")
            legal_document.append(
                LegalDocumentItem(
                    article_number=state.article_number,
                    paragraph_number=state.paragraph_number,
                    text=state.text.strip()
                )
            )

    def parse(self, pages: List[List[str]], parsing_rules: ParsingRules) -> List[LegalDocumentItem]:
        """Parse a list of page texts into structured legal document items.
        
        This is the main parsing method that processes pages of text according to
        the provided parsing rules, identifying structural elements like articles,
        paragraphs, and sections, and organizing them into LegalDocumentItem objects.
        
        Args:
            pages (List[List[str]]): List of pages, each containing a list of text lines.
            parsing_rules (ParsingRules): Rules defining how to parse the document structure.
        
        Returns:
            List[LegalDocumentItem]: List of structured legal document items.
        """
        state = ParsingState()
        legal_document = []
        
        for page_chunks in tqdm(pages, "Pages"):
            for chunk in page_chunks[parsing_rules.header_lines_to_skip:]:
                if chunk == parsing_rules.end_marker:
                    logger.debug(f"End of pages: {chunk}")
                    break

                if any(pattern.match(chunk) for pattern in parsing_rules.skip_patterns):
                    logger.debug(f"Skip: {chunk}")
                    continue

                if any(pattern.match(chunk) for pattern in parsing_rules.section_marker_patterns):
                    logger.debug(f"Out of section: {chunk}")
                    state.in_article_section = False
                    continue

                if parsing_rules.article_pattern.match(chunk):
                    if state.article_number:  # No need to flush when processing the first article
                        self._flush_buffer(legal_document, state)
                    state.article_number = int(chunk.split()[1])
                    state.paragraph_number = None
                    state.text = ""
                    state.in_article_section = True
                    logger.debug(f"New article section")
                    logger.debug(f"Current article: {state.article_number}")
                    continue

                # Edge case: "Pasal" (article) without number — next chunk might contain the article number
                if parsing_rules.article_wo_number_pattern.match(chunk):
                    logger.debug(f"Article without number detected: {chunk}")
                    state.prev_was_article_wo_number = True
                    continue

                if state.prev_was_article_wo_number:
                    state.prev_was_article_wo_number = False
                    if self._validate_article_marker_for_article_wo_number(chunk, state):
                        if state.article_number:  # No need to flush when processing the first article
                            self._flush_buffer(legal_document, state)
                        state.article_number = int(chunk)
                        state.paragraph_number = None
                        state.text = ""
                        state.in_article_section = True
                        logger.debug(f"New article section")
                        logger.debug(f"Current article: {state.article_number}")
                        continue
                    else:
                        # The previous chunk (i.e. "Pasal") and the current chunk are still treated as part of the current article’s content
                        logger.debug(f"Still the part of current article")
                        chunk = f"Pasal {chunk}"
                
                paragraph_match = parsing_rules.paragraph_pattern.match(chunk)
                if paragraph_match:
                    paragraph_number_candidate = int(paragraph_match[0].replace("Ayat ", "").replace("(", "").replace(")", ""))
                    
                    if self._validate_paragraph_marker(paragraph_number_candidate, state):
                        if state.paragraph_number:  # No need to flush when processing the first paragraph
                            self._flush_buffer(legal_document, state)
                        state.paragraph_number = paragraph_number_candidate
                        state.text = ""
                        logger.debug(f"New paragraph section")
                        logger.debug(f"Current article: {chunk}, current paragraph: {state.paragraph_number}")
                    
                    continue

                # Append the text if it's part of the article content
                if state.in_article_section:
                    logger.debug("Append chunk")
                    state.text = self._append_text(state.text, chunk, bool(parsing_rules.ordered_list_pattern.match(chunk)))

        # Flush the last item
        self._flush_buffer(legal_document, state)

        return legal_document
    