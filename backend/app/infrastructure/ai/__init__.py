"""Provider-neutral AI interfaces."""

from app.infrastructure.ai.protocols import AIProvider, EmbeddingClient, EmbeddingIdentity

__all__ = ["AIProvider", "EmbeddingClient", "EmbeddingIdentity"]
