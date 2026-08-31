"""OpenAI implementation of the provider-neutral AI contracts."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import Settings
from app.exceptions import ProviderUnavailableError
from app.infrastructure.ai.protocols import EmbeddingClient, EmbeddingIdentity, ReasoningEffort


class SafeEmbeddingClient:
    """Translate embedding failures into the stable application error contract."""

    def __init__(self, client: Embeddings):
        self._client = client

    async def aembed_query(self, text: str) -> list[float]:
        try:
            return await self._client.aembed_query(text)
        except Exception as exc:
            raise ProviderUnavailableError(
                "The embedding provider is temporarily unavailable."
            ) from exc

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return await self._client.aembed_documents(texts)
        except Exception as exc:
            raise ProviderUnavailableError(
                "The embedding provider is temporarily unavailable."
            ) from exc


class OpenAIProvider:
    """Construct OpenAI models behind provider-neutral interfaces."""

    def __init__(self, settings: Settings):
        self._settings = settings

    @property
    def embedding_identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity(
            provider=self._settings.embedding_provider,
            model=self._settings.embedding_model,
            dimensions=self._settings.embedding_dimensions,
        )

    def create_chat_model(
        self,
        *,
        model: str,
        reasoning_effort: ReasoningEffort,
    ) -> BaseChatModel:
        model_options: dict[str, object] = {}
        if model.startswith(("gpt-5", "o")):
            model_options["reasoning"] = {"effort": reasoning_effort}

        return ChatOpenAI(
            model=model,
            api_key=self._settings.openai_key,
            use_responses_api=True,
            store=False,
            temperature=0,
            timeout=self._settings.ai_request_timeout_seconds,
            max_retries=self._settings.ai_max_retries,
            **model_options,
        )

    def create_embeddings(self) -> EmbeddingClient:
        return SafeEmbeddingClient(
            OpenAIEmbeddings(
                model=self._settings.embedding_model,
                dimensions=self._settings.embedding_dimensions,
                api_key=self._settings.openai_key,
                request_timeout=self._settings.ai_request_timeout_seconds,
                max_retries=self._settings.ai_max_retries,
            )
        )
