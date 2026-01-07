import unittest

from ingestions.parser.core import LegalDocumentItem, LegalPDFParser, ParsingState, ParsingRules
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
from logging_setup import setup_logger


class TestCore(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.pdf_parser = LegalPDFParser()

    def test_validate_article_marker_for_article_wo_number(self):
        self.assertTrue(
            self.pdf_parser._validate_article_marker_for_article_wo_number("2", ParsingState(article_number=1, in_article_section=True))
        )
        self.assertTrue(
            self.pdf_parser._validate_article_marker_for_article_wo_number("1", ParsingState(article_number=None, in_article_section=True))
        )
        self.assertTrue(
            self.pdf_parser._validate_article_marker_for_article_wo_number("2", ParsingState(article_number=1, in_article_section=False))
        )
        self.assertTrue(
            self.pdf_parser._validate_article_marker_for_article_wo_number("1", ParsingState(article_number=None, in_article_section=False))
        )
        self.assertFalse(
            self.pdf_parser._validate_article_marker_for_article_wo_number("2", ParsingState(article_number=2, in_article_section=True))
        )
        self.assertFalse(
            self.pdf_parser._validate_article_marker_for_article_wo_number("text", ParsingState(article_number=2, in_article_section=True))
        )
        self.assertFalse(
            self.pdf_parser._validate_article_marker_for_article_wo_number("2", ParsingState(article_number=2, in_article_section=False))
        )
        self.assertFalse(
            self.pdf_parser._validate_article_marker_for_article_wo_number("text", ParsingState(article_number=2, in_article_section=False))
        )

    def test_validate_paragraph_marker(self):
        self.assertTrue(self.pdf_parser._validate_paragraph_marker(1, ParsingState(paragraph_number=None)))
        self.assertTrue(self.pdf_parser._validate_paragraph_marker(2, ParsingState(paragraph_number=1)))
        self.assertFalse(self.pdf_parser._validate_paragraph_marker(2, ParsingState(paragraph_number=2)))

    def test_flush_buffer(self):
        legal_document_1 = []
        legal_document_2 = [LegalDocumentItem(article_number=1, paragraph_number=None, text="text")]

        # Act, should append to legal_document_1
        self.pdf_parser._flush_buffer(legal_document_1, ParsingState(article_number=1, paragraph_number=None, text="text"))
        self.assertEqual(legal_document_1, legal_document_2)

        # Act, should append to legal_document_2
        self.pdf_parser._flush_buffer(legal_document_2, ParsingState(article_number=2, paragraph_number=1, text="text"))
        self.assertEqual(
            legal_document_2,
            [
                LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
                LegalDocumentItem(article_number=2, paragraph_number=1, text="text")
            ]
        )

    def test_append_text(self):
        text_1 = "text 1"
        text_2 = "text 2"

        self.assertEqual(self.pdf_parser._append_text(text_1, text_2, True), f"{text_1}\n{text_2}")
        self.assertEqual(self.pdf_parser._append_text(text_1, text_2, False), f"{text_1} {text_2}")
        self.assertEqual(self.pdf_parser._append_text(None, text_2, True), text_2)
        self.assertEqual(self.pdf_parser._append_text(None, text_2, False), text_2)
        self.assertEqual(self.pdf_parser._append_text(text_1, None, True), "")
        self.assertEqual(self.pdf_parser._append_text(text_1, None, False), "")
        self.assertEqual(self.pdf_parser._append_text(None, None, True), "")
        self.assertEqual(self.pdf_parser._append_text(None, None, False), "")

    def test_parse_body(self):
        parsing_rules = ParsingRules(
            header_lines_to_skip=3,
            end_marker=END_OF_BODY_MARKER,
            paragraph_pattern=BODY_PARAGRAPH_PATTERN,
            ordered_list_pattern=BODY_ORDERED_LIST_PATTERN,
            article_pattern=ARTICLE_PATTERN,
            article_wo_number_pattern=ARTICLE_WO_NUMBER_PATTERN,
            section_marker_patterns=[CHAPTER_PATTERN, SECTION_PATTERN, SUBSECTION_PATTERN],
            skip_patterns=[PAGE_PATTERN, TRIPPLE_DOTS_PATTERN, TRIPPLE_DOT_SPACES_PATTERN]
        )

        body_pages = [
            [
                "PRESIDEN",
                "REPUBLIK INDONESIA",
                "- 2 -",
                "text4 . . .",
                "Mengingat",
                ":",
                "Pasal 5 ayat (1) serta Pasal 20 ayat (1) dan ayat (2) Undang-",
                "UNDANG-UNDANG TENTANG LALU LINTAS DAN ANGKUTAN",
                "JALAN.",
                "BAB I",
                "KETENTUAN UMUM",
                "Pasal 1",
                "text1",
                "text2",
                "text3"
            ],
            [
                "PRESIDEN",
                "REPUBLIK INDONESIA",
                "- 3 -",
                "text6 . . .",
                "text4",
                "text5",
            ],
            [
                "PRESIDEN",
                "REPUBLIK INDONESIA",
                "- 4 -",
                "Pasal 2 . . .",
                "text6",
                "text7",
                "text8",
            ],
            [
                "PRESIDEN",
                "REPUBLIK INDONESIA",
                "- 5 -",
                "(2) . . .",
                "Pasal 2",
                "(1)",
                "text1",
                "text2",
            ],
            [
                "PRESIDEN",
                "REPUBLIK INDONESIA",
                "- 5 -",
                "Pasal 4 . . .",
                "(2)",
                "text1",
                "BAB II",
                "JUDUL BAB",
                "Bagian Kesatu",
                "Judul Bagian Kesatu",
                "Pasal 3",
                "(1)",
                "text1",
            ],
            [
                "PRESIDEN",
                "REPUBLIK INDONESIA",
                "- 6 -",
                "text3 . . .",
                "Pasal 4",
                "text1",
                "text2"
            ],
            [
                "PRESIDEN",
                "REPUBLIK INDONESIA",
                "- 7 -",
                "text1 . . .",
                "text3",
                "Bagian Kedua",
                "Judul Bagian Kedua",
                "Pasal 5",
                "(1)"
            ],
            [
                "PRESIDEN",
                "REPUBLIK INDONESIA",
                "- 8 -",
                "Pasal 6 . . .",
                "text1",
                "text2:",
                "1. ordered list 1",
                "2.",
                "ordered list 2",
                "3.",
                "ordered list 3",
                "text3",
                "(2)",
                "text1:",
                "a.",
                "ordered list 1;",
                "b. ordered list 2; dan",
                "c.",
                "ordered list 3",
                "text2"
            ],
            [
                "PRESIDEN",
                "REPUBLIK INDONESIA",
                "- 9 -",
                "Pasal 9 . . .",
                "Bagian Ketiga",
                "Judul Bagian Ketiga",
                "Pasal",
                "6",
                "text1",
                "text2",
                "Pasal",
                "7",
                "(1)",
                "text1",
                "text2",
                "Pasal 8",
                "Yang diatur oleh",
                "Pasal",
                "ini adalah:",
                "1.",
                "ordered list 1",
                "2. ordered list 2",
                "text1",
                "3. ordered list 3"
            ]
        ]

        self.assertEqual(
            self.pdf_parser.parse(body_pages, parsing_rules),
            [
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
        )


    def test_parse_elucidation(self):
        parsing_rules = ParsingRules(
            header_lines_to_skip=3,
            end_marker=END_OF_ELUCIDATION_MARKER,
            paragraph_pattern=ELUCIDATION_PARAGRAPH_PATTERN,
            ordered_list_pattern=ELUCIDATION_ORDERED_LIST_PATTERN,
            article_pattern=ARTICLE_PATTERN,
            article_wo_number_pattern=ARTICLE_WO_NUMBER_PATTERN,
            section_marker_patterns=[CHAPTER_PATTERN, SECTION_PATTERN, SUBSECTION_PATTERN],
            skip_patterns=[PAGE_PATTERN, TRIPPLE_DOTS_PATTERN, TRIPPLE_DOT_SPACES_PATTERN]
        )

        elucidation_pages = [
            [
                "PRESIDEN",
                "REPUBLIK INDONESIA",
                "- 7 -",
                "text2 . . .",
                "II. PASAL DEMI PASAL",
                "Pasal 1",
                "Cukup jelas.",
                "Pasal 2",
                "Huruf a text1 text2",
                "Huruf b",
                "text1 text2",
                "Huruf c",
                "text1",
            ],
            [
                "PRESIDEN",
                "REPUBLIK INDONESIA",
                "- 8 -",
                "Ayat (3) . . .",
                "text2",
                "Pasal 3",
                "Ayat (1)",
                "Cukup jelas.",
                "Ayat (2)",
                "text1 text2",
                "text3"
            ],
            [
                "PRESIDEN",
                "REPUBLIK INDONESIA",
                "- 9 -",
                "Pasal 5 . . .",
                "Ayat (3)",
                "Cukup jelas.",
                "Pasal 4",
                "Ayat (1)",
                "Huruf a",
                "text1 text2",
                "Huruf b text1",
                "text2",
                "Ayat (2)",
                "Cukup jelas."
            ],
            [
                "PRESIDEN",
                "REPUBLIK INDONESIA",
                "- 10 -",
                "Huruf b . . .",
                "Pasal 5",
                "Yang dimaksud:",
                "1. text1",
                "text2",
                "2.",
                "text1 text2",
                "Pasal 6",
                "Ayat (1)",
                "Yang dimaksud:",
                "a.",
                "text1 text2",
                "b. text1",
                "text2",
                "Ayat (2)",
                "Huruf a",
                "yaitu:",
                "1.",
                "text1 text2",
                "2. text1 text2"
            ],
            [
                "PRESIDEN",
                "REPUBLIK INDONESIA",
                "- 11 -",
                "Pasal 7 . . .",
                "Huruf b",
                "yaitu:",
                "1. text1",
                "di antaranya:",
                "a. text1 text2",
                "b.",
                "text1",
                "2.",
                "text1 text2",
                "Huruf c",
                "text1 text2"
            ]
        ]

        self.assertEqual(
            self.pdf_parser.parse(elucidation_pages, parsing_rules),
            [
                LegalDocumentItem(article_number=1, paragraph_number=None, text="Cukup jelas."),
                LegalDocumentItem(article_number=2, paragraph_number=None, text="Huruf a text1 text2\nHuruf b text1 text2\nHuruf c text1 text2"),
                LegalDocumentItem(article_number=3, paragraph_number=1, text="Cukup jelas."),
                LegalDocumentItem(article_number=3, paragraph_number=2, text="text1 text2 text3"),
                LegalDocumentItem(article_number=3, paragraph_number=3, text="Cukup jelas."),
                LegalDocumentItem(article_number=4, paragraph_number=1, text="Huruf a text1 text2\nHuruf b text1 text2"),
                LegalDocumentItem(article_number=4, paragraph_number=2, text="Cukup jelas."),
                LegalDocumentItem(article_number=5, paragraph_number=None, text="Yang dimaksud:\n1. text1 text2\n2. text1 text2"),
                LegalDocumentItem(article_number=6, paragraph_number=1, text="Yang dimaksud:\na. text1 text2\nb. text1 text2"),
                LegalDocumentItem(
                    article_number=6,
                    paragraph_number=2,
                    text="Huruf a yaitu:\n1. text1 text2\n2. text1 text2\nHuruf b yaitu:\n1. text1 di antaranya:\na. text1 text2\nb. text1\n2. text1 text2\nHuruf c text1 text2"
                )
            ]
        )


if __name__ == "__main__":
    unittest.main()
