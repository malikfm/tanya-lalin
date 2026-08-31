"""Application composition root and FastAPI dependency accessors."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import Request

from app.application.chat_service import ChatApplicationService
from app.config import Settings
from app.exceptions import ProviderUnavailableError
from app.infrastructure.ai import AIProvider, EmbeddingIdentity
from app.infrastructure.ai.openai import OpenAIProvider
from app.infrastructure.corpus_index import StaticCorpusIndex
from app.infrastructure.sessions import SessionStore, create_session_store
from app.infrastructure.tracing import configure_tracing
from app.rag.generation import AnswerGenerator
from app.rag.models import ConversationMessage, PipelineEvent, PipelineStage, StageEvent
from app.rag.pipeline import RAGPipeline
from app.rag.query_analysis import QueryAnalyzer
from app.rag.reranker import Reranker
from app.rag.retrieval import HybridRetriever
from app.rag.validation import AnswerValidator


class UnavailablePipeline:
    """Keep health endpoints available when the provider is not configured."""

    async def stream(
        self,
        query: str,
        history: list[ConversationMessage],
    ) -> AsyncIterator[PipelineEvent]:
        yield StageEvent(stage=PipelineStage.VALIDATING)
        raise ProviderUnavailableError("The AI provider is not configured.")


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    corpus: StaticCorpusIndex
    sessions: SessionStore
    chat: ChatApplicationService

    @classmethod
    async def build(
        cls,
        settings: Settings,
        *,
        provider: AIProvider | None = None,
    ) -> AppContainer:
        configure_tracing(settings)
        embedding_identity = (
            provider.embedding_identity
            if provider is not None
            else EmbeddingIdentity(
                provider=settings.embedding_provider,
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
            )
        )
        corpus = StaticCorpusIndex(settings.corpus_dir, embedding_identity)
        sessions = create_session_store(
            settings.redis_url,
            ttl_seconds=settings.session_ttl_seconds,
            max_messages=settings.session_max_messages,
        )
        if not await sessions.ping():
            raise RuntimeError("Configured session store is unavailable")

        if not settings.openai_key and provider is None:
            chat = ChatApplicationService(
                pipeline=UnavailablePipeline(),
                sessions=sessions,
                settings=settings,
                corpus_version=corpus.version,
            )
            return cls(settings=settings, corpus=corpus, sessions=sessions, chat=chat)

        provider = provider or OpenAIProvider(settings)
        analysis_model = provider.create_chat_model(
            model=settings.chat_model,
            reasoning_effort=settings.analysis_reasoning_effort,
        )
        generation_model = provider.create_chat_model(
            model=settings.chat_model,
            reasoning_effort=settings.generation_reasoning_effort,
        )
        validation_model = (
            provider.create_chat_model(
                model=settings.output_validation_model,
                reasoning_effort=settings.analysis_reasoning_effort,
            )
            if settings.output_validation_enabled
            else None
        )
        reranker = (
            Reranker(
                provider.create_chat_model(
                    model=settings.reranker_model,
                    reasoning_effort=settings.analysis_reasoning_effort,
                )
            )
            if settings.reranker_enabled
            else None
        )
        pipeline = RAGPipeline(
            settings=settings,
            analyzer=QueryAnalyzer(analysis_model),
            retriever=HybridRetriever(corpus, provider.create_embeddings(), settings),
            generator=AnswerGenerator(generation_model),
            validator=AnswerValidator(validation_model),
            reranker=reranker,
        )
        chat = ChatApplicationService(
            pipeline=pipeline,
            sessions=sessions,
            settings=settings,
            corpus_version=corpus.version,
        )
        return cls(settings=settings, corpus=corpus, sessions=sessions, chat=chat)

    async def close(self) -> None:
        await self.sessions.close()


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_chat_service(request: Request) -> ChatApplicationService:
    return get_container(request).chat


def get_session_store(request: Request) -> SessionStore:
    return get_container(request).sessions
