"""Module for input/output operations related to text processing.

This module contains utility functions for handling file operations,
particularly for working with JSONL (JSON Lines) files that are commonly
used for storing serialized data records.
"""

from pathlib import Path
from typing import Iterable


def write_to_jsonl(text_lines: Iterable[str], output_path: Path) -> None:
    """Write a list of JSON-formatted strings to a JSONL file.
    
    This function creates a JSONL file at the specified path, ensuring that
    the parent directory exists. Each string in the input list is written
    as a separate line in the file, forming a valid JSONL format.
    
    JSONL is useful for storing large amounts of structured data in a
    line-delimited format that can be processed incrementally.
    
    Args:
        text_lines (List[str]): List of JSON-formatted strings to write.
        output_path (Path): Path where the JSONL file will be created.
        
    Returns:
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in text_lines:
            f.write(item + "\n")
