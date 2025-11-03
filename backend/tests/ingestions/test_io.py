import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from ingestions import io
from ingestions.models import LegalDocumentItem


class TestIO(unittest.TestCase):
    def test_write_to_jsonl(self):
        legal_document = [
            LegalDocumentItem(article_number=1, paragraph_number=None, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=1, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=2, text="text"),
            LegalDocumentItem(article_number=2, paragraph_number=3, text="text"),
            LegalDocumentItem(article_number=3, paragraph_number=None, text="text")
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir).joinpath("test.jsonl")
            io.write_to_jsonl(map(lambda item: item.model_dump_json(), legal_document), output_path)

            # Assert file exists
            self.assertTrue(output_path.exists())

            # Read back and check content
            with open(output_path) as f:
                lines = f.readlines()
            
            self.assertEqual(len(lines), len(legal_document))
            
            for writen, original in zip(lines, legal_document):
                self.assertEqual(LegalDocumentItem.model_validate_json(writen), original)


if __name__ == "__main__":
    unittest.main()
    