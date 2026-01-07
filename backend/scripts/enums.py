"""Enum definitions for the script modules."""
from enum import Enum


class LegalTextType(str, Enum):
    """Types of legal text content."""
    BODY = "body"
    ELUCIDATION = "elucidation"
