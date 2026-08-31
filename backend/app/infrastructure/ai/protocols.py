"""Provider-neutral contracts used by application composition and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from langchain_core.language_models.chat_models import BaseChatModel

ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    """Identify the exact embedding space required by a corpus."""

    provider: str
    model: str
    dimensions: int


class EmbeddingClient(Protocol):
    """Minimal embedding behavior required by runtime and corpus tooling."""

    async def aembed_query(self, text: str) -> list[float]: ...

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...


class AIProvider(Protocol):
    """Create provider-backed models without exposing their SDK to the RAG layer."""

    @property
    def embedding_identity(self) -> EmbeddingIdentity: ...

    def create_chat_model(
        self,
        *,
        model: str,
        reasoning_effort: ReasoningEffort,
    ) -> BaseChatModel: ...

    def create_embeddings(self) -> EmbeddingClient: ...
