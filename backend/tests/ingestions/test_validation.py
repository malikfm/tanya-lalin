import unittest

from ingestions import validation
from ingestions.parser_core import LegalDocumentItem
from ingestions.validation import ValidationReport
from logging_setup import setup_logger


class TestValidation(unittest.TestCase):
    def test_check_total_articles(self):
        empty_list = []
        empty_items = [LegalDocumentItem(None, None, None),LegalDocumentItem(None, None, None)]
        partial_empty_items = [
            LegalDocumentItem(None, None, None),
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text"),
            LegalDocumentItem(None, None, None)
        ]
        valid_legal_document = [
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text1 text2 text3 text4 text5 text6 text7 text8"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text1 text2"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text1"),
            LegalDocumentItem(article_number=3, paragraph_number=1, text="text1"),
            LegalDocumentItem(article_number=4, paragraph_number=None, text="text1 text2 text3"),
            LegalDocumentItem(article_number=5, paragraph_number=1, text="text1 text2:\n1. ordered list 1\n2. ordered list 2\n3. ordered list 3 text3"),
            LegalDocumentItem(article_number=5, paragraph_number=2, text="text1:\na. ordered list 1;\nb. ordered list 2; dan\nc. ordered list 3 text2"),
            LegalDocumentItem(article_number=6, paragraph_number=None, text="text1 text2"),
            LegalDocumentItem(article_number=7, paragraph_number=1, text="text1 text2"),
            LegalDocumentItem(article_number=8, paragraph_number=None, text="Yang diatur oleh Pasal ini adalah:\n1. ordered list 1\n2. ordered list 2 text1\n3. ordered list 3")
        ]

        self.assertEqual(validation.check_total_articles(empty_list), 0)
        self.assertEqual(validation.check_total_articles(empty_items), 0)
        self.assertEqual(validation.check_total_articles(partial_empty_items), 2)
        self.assertEqual(validation.check_total_articles(valid_legal_document), 8)

    def test_check_missing_articles(self):
        empty_list = []
        empty_items = [LegalDocumentItem(None, None, None),LegalDocumentItem(None, None, None)]
        partial_empty_items = [
            LegalDocumentItem(None, None, None),
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text"),
            LegalDocumentItem(None, None, None)
        ]
        missing_articles = [
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text"),
            LegalDocumentItem(article_number=4, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=6, paragraph_number=1, text="text")
        ]
        valid_legal_document = [
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text1 text2 text3 text4 text5 text6 text7 text8"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text1 text2"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text1"),
            LegalDocumentItem(article_number=3, paragraph_number=1, text="text1"),
            LegalDocumentItem(article_number=4, paragraph_number=None, text="text1 text2 text3"),
            LegalDocumentItem(article_number=5, paragraph_number=1, text="text1 text2:\n1. ordered list 1\n2. ordered list 2\n3. ordered list 3 text3"),
            LegalDocumentItem(article_number=5, paragraph_number=2, text="text1:\na. ordered list 1;\nb. ordered list 2; dan\nc. ordered list 3 text2"),
            LegalDocumentItem(article_number=6, paragraph_number=None, text="text1 text2"),
            LegalDocumentItem(article_number=7, paragraph_number=1, text="text1 text2"),
            LegalDocumentItem(article_number=8, paragraph_number=None, text="Yang diatur oleh Pasal ini adalah:\n1. ordered list 1\n2. ordered list 2 text1\n3. ordered list 3")
        ]

        self.assertEqual(validation.check_missing_articles(empty_list, 3), [1,2,3])
        self.assertEqual(validation.check_missing_articles(empty_items, 2), [1,2])
        self.assertEqual(validation.check_missing_articles(partial_empty_items, 3), [3])
        self.assertEqual(validation.check_missing_articles(missing_articles, 6), [3,5])
        self.assertEqual(validation.check_missing_articles(valid_legal_document, 8), [])
    
    def test_check_missing_paragraphs(self):
        empty_list = []
        empty_items = [LegalDocumentItem(None, None, None),LegalDocumentItem(None, None, None)]
        partial_empty_items = [
            LegalDocumentItem(None, None, None),
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text"),
            LegalDocumentItem(None, None, None)
        ]
        missing_paragraphs_in_partial_empty_items = [
            LegalDocumentItem(None, None, None),
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=6, text="text"),
            LegalDocumentItem(None, None, None)
        ]
        missing_paragraphs = [
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=5, text="text"),
            LegalDocumentItem(article_number=4, paragraph_number=5, text="text"),
            LegalDocumentItem(article_number=5, paragraph_number=1, text="text")
        ]
        valid_legal_document = [
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text1 text2 text3 text4 text5 text6 text7 text8"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text1 text2"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text1"),
            LegalDocumentItem(article_number=3, paragraph_number=1, text="text1"),
            LegalDocumentItem(article_number=4, paragraph_number=None, text="text1 text2 text3"),
            LegalDocumentItem(article_number=5, paragraph_number=1, text="text1 text2:\n1. ordered list 1\n2. ordered list 2\n3. ordered list 3 text3"),
            LegalDocumentItem(article_number=5, paragraph_number=2, text="text1:\na. ordered list 1;\nb. ordered list 2; dan\nc. ordered list 3 text2"),
            LegalDocumentItem(article_number=6, paragraph_number=None, text="text1 text2"),
            LegalDocumentItem(article_number=7, paragraph_number=1, text="text1 text2"),
            LegalDocumentItem(article_number=8, paragraph_number=None, text="Yang diatur oleh Pasal ini adalah:\n1. ordered list 1\n2. ordered list 2 text1\n3. ordered list 3")
        ]

        self.assertRaises(ValueError, validation.check_missing_paragraphs, empty_list)
        self.assertRaises(ValueError, validation.check_missing_paragraphs, empty_items)
        self.assertEqual(validation.check_missing_paragraphs(partial_empty_items), {})
        self.assertEqual(validation.check_missing_paragraphs(missing_paragraphs_in_partial_empty_items), {2:[3,4,5]})
        self.assertEqual(validation.check_missing_paragraphs(missing_paragraphs), {3:[2,3,4],4:[1,2,3,4]})
        self.assertEqual(validation.check_missing_paragraphs(valid_legal_document), {})
    
    def test_check_duplicate_article_paragraphs(self):
        empty_list = []
        empty_items = [LegalDocumentItem(None, None, None),LegalDocumentItem(None, None, None)]
        partial_empty_items = [
            LegalDocumentItem(None, None, None),
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text"),
            LegalDocumentItem(None, None, None)
        ]
        duplicates_in_partial_empty_items = [
            LegalDocumentItem(None, None, None),
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=1, text="text"),
            LegalDocumentItem(None, None, None)
        ]
        duplicates = [
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=2, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=3, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=3, text="text"),
            LegalDocumentItem(article_number=4, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=4, paragraph_number=2, text="text"),
            LegalDocumentItem(article_number=4, paragraph_number=3, text="text")
        ]
        valid_legal_document = [
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text1 text2 text3 text4 text5 text6 text7 text8"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text1 text2"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text1"),
            LegalDocumentItem(article_number=3, paragraph_number=1, text="text1"),
            LegalDocumentItem(article_number=4, paragraph_number=None, text="text1 text2 text3"),
            LegalDocumentItem(article_number=5, paragraph_number=1, text="text1 text2:\n1. ordered list 1\n2. ordered list 2\n3. ordered list 3 text3"),
            LegalDocumentItem(article_number=5, paragraph_number=2, text="text1:\na. ordered list 1;\nb. ordered list 2; dan\nc. ordered list 3 text2"),
            LegalDocumentItem(article_number=6, paragraph_number=None, text="text1 text2"),
            LegalDocumentItem(article_number=7, paragraph_number=1, text="text1 text2"),
            LegalDocumentItem(article_number=8, paragraph_number=None, text="Yang diatur oleh Pasal ini adalah:\n1. ordered list 1\n2. ordered list 2 text1\n3. ordered list 3")
        ]

        self.assertEqual(validation.check_duplicate_article_paragraphs(empty_list), [])
        self.assertEqual(validation.check_duplicate_article_paragraphs(empty_items), [])
        self.assertEqual(validation.check_duplicate_article_paragraphs(partial_empty_items), [])
        self.assertEqual(validation.check_duplicate_article_paragraphs(duplicates_in_partial_empty_items), [(3,1)])
        self.assertEqual(validation.check_duplicate_article_paragraphs(duplicates), [(1,0),(3,1),(3,3)])
        self.assertEqual(validation.check_duplicate_article_paragraphs(valid_legal_document), [])
    
    def test_validate_result(self):
        logger = setup_logger()

        empty_list = []
        empty_items = [LegalDocumentItem(None, None, None),LegalDocumentItem(None, None, None)]
        partial_empty_items = [
            LegalDocumentItem(None, None, None),
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text"),
            LegalDocumentItem(None, None, None)
        ]
        missing_articles = [
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text"),
            LegalDocumentItem(article_number=4, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=6, paragraph_number=1, text="text")
        ]
        missing_paragraphs_in_partial_empty_items = [
            LegalDocumentItem(None, None, None),
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=6, text="text"),
            LegalDocumentItem(None, None, None)
        ]
        missing_paragraphs = [
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=5, text="text"),
            LegalDocumentItem(article_number=4, paragraph_number=5, text="text"),
            LegalDocumentItem(article_number=5, paragraph_number=1, text="text")
        ]
        duplicates_in_partial_empty_items = [
            LegalDocumentItem(None, None, None),
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=1, text="text"),
            LegalDocumentItem(None, None, None)
        ]
        duplicates = [
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=2, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=3, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=3, text="text"),
            LegalDocumentItem(article_number=4, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=4, paragraph_number=2, text="text"),
            LegalDocumentItem(article_number=4, paragraph_number=3, text="text")
        ]
        valid_legal_document = [
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text1 text2 text3 text4 text5 text6 text7 text8"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text1 text2"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text1"),
            LegalDocumentItem(article_number=3, paragraph_number=1, text="text1"),
            LegalDocumentItem(article_number=4, paragraph_number=None, text="text1 text2 text3"),
            LegalDocumentItem(article_number=5, paragraph_number=1, text="text1 text2:\n1. ordered list 1\n2. ordered list 2\n3. ordered list 3 text3"),
            LegalDocumentItem(article_number=5, paragraph_number=2, text="text1:\na. ordered list 1;\nb. ordered list 2; dan\nc. ordered list 3 text2"),
            LegalDocumentItem(article_number=6, paragraph_number=None, text="text1 text2"),
            LegalDocumentItem(article_number=7, paragraph_number=1, text="text1 text2"),
            LegalDocumentItem(article_number=8, paragraph_number=None, text="Yang diatur oleh Pasal ini adalah:\n1. ordered list 1\n2. ordered list 2 text1\n3. ordered list 3")
        ]

        # Empty list and items, should raise ValueError in check_missing_paragraphs
        self.assertRaises(ValueError, validation.validate_result, logger, empty_list, "test", 3)
        self.assertRaises(ValueError, validation.validate_result, logger, empty_items, "test", 2)
        
        # Partially empty items:
        #   - total_articles = 2, only 2 non-null unique articles exist, though expected_total_articles = 3
        #   - missing_articles = [3], since only articles 1 and 2 exist, article 3 is missing
        #   - missing_paragraphs = {}, no missing paragraphs in the existing non-null articles
        #   - duplicate_article_paragraphs = [], no duplicate article–paragraph combinations among non-null articles
        self.assertEqual(
            validation.validate_result(logger, partial_empty_items, "test", 3),
            ValidationReport(total_articles=2, missing_articles=[3], missing_paragraphs={}, duplicate_article_paragraphs=[])
        )

        # Total articles mismatch (expected 10, but 8 are given):
        #   - total_articles = 8, only 8 unique articles exist, though expected_total_articles = 10
        #   - missing_articles = [9,10], articles 1–8 exist, but 9 and 10 are missing
        #   - missing_paragraphs = {}, no missing paragraphs in any article
        #   - duplicate_article_paragraphs = [], no duplicate article–paragraph combinations
        self.assertEqual(
            validation.validate_result(logger, valid_legal_document, "test", 10),
            ValidationReport(total_articles=8, missing_articles=[9,10], missing_paragraphs={}, duplicate_article_paragraphs=[])
        )

        # Missing articles (which also causes a total article count mismatch):
        #   - total_articles = 4, only 4 unique articles exist, though expected_total_articles = 6
        #   - missing_articles = [3,5], articles 3 and 5 are missing
        #   - missing_paragraphs = {}, no missing paragraphs in existing articles
        #   - duplicate_article_paragraphs = [], no duplicate article–paragraph combinations in existing articles
        self.assertEqual(
            validation.validate_result(logger, missing_articles, "test", 6),
            ValidationReport(total_articles=4, missing_articles=[3,5], missing_paragraphs={}, duplicate_article_paragraphs=[])
        )

        # Missing paragraphs in partially empty items:
        #   - total_articles = 2, only 2 non-null unique articles exist, though expected_total_articles = 3
        #   - missing_articles = [3], since only articles 1 and 2 exist, article 3 is missing
        #   - missing_paragraphs = {2:[3,4,5]}, paragraphs 3, 4, and 5 are missing in article 2 (based on its min/max paragraph range)
        #   - duplicate_article_paragraphs = [], no duplicate article–paragraph combinations among non-null articles
        self.assertEqual(
            validation.validate_result(logger, missing_paragraphs_in_partial_empty_items, "test", 3),
            ValidationReport(total_articles=2, missing_articles=[3], missing_paragraphs={2:[3,4,5]}, duplicate_article_paragraphs=[])
        )

        # Missing paragraphs:
        #   - total_articles = 5, 5 unique articles exist, matching expected_total_articles
        #   - missing_articles = [], all expected articles (1–5) exist
        #   - missing_paragraphs = {3:[2,3,4], 4:[1,2,3,4]}, paragraphs 2–4 missing in article 3, and paragraphs 1–4 missing in article 4
        #   - duplicate_article_paragraphs = [], no duplicate article–paragraph combinations
        self.assertEqual(
            validation.validate_result(logger, missing_paragraphs, "test", 5),
            ValidationReport(total_articles=5, missing_articles=[], missing_paragraphs={3:[2,3,4],4:[1,2,3,4]}, duplicate_article_paragraphs=[])
        )

        # Duplicates in partially empty items:
        #   - total_articles = 3, only 3 unique articles exist, though expected_total_articles = 4
        #   - missing_articles = [4], since only articles 1–3 exist, article 4 is missing
        #   - missing_paragraphs = {}, no missing paragraphs in existing non-null articles
        #   - duplicate_article_paragraphs = [(3,1)], article–paragraph (3,1) appears multiple times
        self.assertEqual(
            validation.validate_result(logger, duplicates_in_partial_empty_items, "test", 4),
            ValidationReport(total_articles=3, missing_articles=[4], missing_paragraphs={}, duplicate_article_paragraphs=[(3,1)])
        )

        # Duplicates exist:
        #   - total_articles = 4, 4 unique articles exist, matching expected_total_articles
        #   - missing_articles = [], all expected articles (1–4) exist
        #   - missing_paragraphs = {}, no missing paragraphs in any article
        #   - duplicate_article_paragraphs = [(1,0), (3,1), (3,3)], these article–paragraph combinations appear multiple times
        self.assertEqual(
            validation.validate_result(logger, duplicates, "test", 4),
            ValidationReport(total_articles=4, missing_articles=[], missing_paragraphs={}, duplicate_article_paragraphs=[(1,0),(3,1),(3,3)])
        )
        
        # Valid case:
        #   - total_articles = 8, 8 unique articles exist, matching expected_total_articles
        #   - missing_articles = [], all expected articles (1–8) exist
        #   - missing_paragraphs = {}, no missing paragraphs in any article
        #   - duplicate_article_paragraphs = [], no duplicate article–paragraph combinations
        self.assertEqual(
            validation.validate_result(logger, valid_legal_document, "test", 8),
            ValidationReport(total_articles=8, missing_articles=[], missing_paragraphs={}, duplicate_article_paragraphs=[])
        )


if __name__ == "__main__":
    unittest.main()
