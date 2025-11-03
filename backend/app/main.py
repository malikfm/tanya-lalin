"""Main FastAPI application for the Indonesian Traffic Law Q&A RAG system."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.core.ollama_client import init_ollama_client
from logging_setup import setup_logger

ROUTER_PREFIX = "/api/v1"

logger = setup_logger()
ollama_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ollama_client = await init_ollama_client()
    yield
    await app.state.ollama_client._client.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Tanya Lalin - Indonesian Traffic Law Chat API",
        description="RAG-based API for answering questions about Indonesian traffic law",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routers
    app.include_router(chat_router, prefix=ROUTER_PREFIX, tags=["chat"])
    app.include_router(health_router, prefix=ROUTER_PREFIX, tags=["health"])

    return app


app = create_app()


@app.get("/")
async def root():
    return {"message": "Tanya Lalin - Indonesian Traffic Law Chat API", "status": "running"}
