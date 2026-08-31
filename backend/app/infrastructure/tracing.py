"""Optional LangSmith environment configuration."""

from __future__ import annotations

import os

from app.config import Settings


def configure_tracing(settings: Settings) -> bool:
    """Enable LangSmith only when explicitly requested and configured."""

    if not settings.langsmith_tracing or not settings.langsmith_key:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    return True
