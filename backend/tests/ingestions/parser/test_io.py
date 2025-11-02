import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from ingestions import io
from ingestions.parser.core import LegalDocumentItem


class TestIO(unittest.TestCase):
    def test_write_to_jsonl(self):
        legal_document = [
            LegalDocumentItem(1, None, "text"),
            LegalDocumentItem(2, 1, "text"),
            LegalDocumentItem(2, 2, "text"),
            LegalDocumentItem(2, 3, "text"),
            LegalDocumentItem(3, None, "text")
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir).joinpath("test.jsonl")
            io.write_to_jsonl(legal_document, output_path)

            # Assert file exists
            self.assertTrue(output_path.exists())

            # Read back and check content
            with open(output_path) as f:
                lines = f.readlines()
            
            self.assertEqual(len(lines), len(legal_document))
            
            for writen, original in zip(lines, legal_document):
                self.assertEqual(json.loads(writen), asdict(original))

    def test_text_to_chunks(self):
        single_line_text = "text1 text2 text3"
        multi_line_text = "text1 text2\ntext3\ntext4 text5 text6"
        multi_line_text_trailing_whitespace = "text1\n\ntext2 text3  \ntext4 "

        self.assertEqual(io.text_to_chunks(single_line_text), ["text1 text2 text3"])
        self.assertEqual(io.text_to_chunks(multi_line_text), ["text1 text2", "text3", "text4 text5 text6"])
        self.assertEqual(io.text_to_chunks(multi_line_text_trailing_whitespace), ["text1", "text2 text3", "text4"])


if __name__ == "__main__":
    unittest.main()
    