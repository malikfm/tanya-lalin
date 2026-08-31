"""OpenAI adapter configuration tests that never call the provider network."""

from __future__ import annotations

import numpy as np
import pytest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import Settings
from app.dependencies import AppContainer
from app.exceptions import ProviderUnavailableError
from app.infrastructure.ai import EmbeddingIdentity
from app.infrastructure.ai.openai import OpenAIProvider, SafeEmbeddingClient


def test_openai_provider_configures_responses_and_embeddings():
    settings = Settings(
        _env_file=None,
        environment="test",
        openai_api_key="test-key",
        chat_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        ai_request_timeout_seconds=12,
        ai_max_retries=1,
    )
    provider = OpenAIProvider(settings)

    model = provider.create_chat_model(model=settings.chat_model, reasoning_effort="low")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o-mini"
    assert model.use_responses_api is True
    assert model.store is False
    assert model.temperature == 0
    assert model.reasoning is None
    assert model.request_timeout == 12
    assert model.max_retries == 1

    embeddings = provider.create_embeddings()
    assert isinstance(embeddings, SafeEmbeddingClient)
    assert isinstance(embeddings._client, OpenAIEmbeddings)
    assert embeddings._client.model == "text-embedding-3-small"
    assert embeddings._client.dimensions == 1536
    assert embeddings._client.request_timeout == 12
    assert embeddings._client.max_retries == 1
    assert provider.embedding_identity == EmbeddingIdentity(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
    )


def test_openai_provider_keeps_reasoning_for_supported_models():
    settings = Settings(_env_file=None, openai_api_key="test-key")
    provider = OpenAIProvider(settings)

    model = provider.create_chat_model(model="gpt-5-mini", reasoning_effort="low")

    assert isinstance(model, ChatOpenAI)
    assert model.reasoning == {"effort": "low"}


def test_settings_do_not_expose_api_key_in_repr():
    settings = Settings(_env_file=None, openai_api_key="super-secret")
    assert settings.openai_key == "super-secret"
    assert "super-secret" not in repr(settings)


async def test_container_without_key_exposes_stable_provider_failure(corpus_factory, chunk):
    corpus_dir = corpus_factory(
        [chunk],
        np.array([[1.0, 0.0]], dtype=np.float32),
        provider="openai",
        model="text-embedding-3-small",
    )
    settings = Settings(
        _env_file=None,
        environment="test",
        corpus_dir=corpus_dir,
        openai_api_key="",
        embedding_dimensions=2,
    )
    container = await AppContainer.build(settings)
    try:
        with pytest.raises(ProviderUnavailableError, match="not configured"):
            await container.chat.chat(message="Apa dendanya?")
    finally:
        await container.close()
