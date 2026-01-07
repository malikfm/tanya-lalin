"""Main FastAPI application for Tanya Lalin - Indonesian Traffic Law Q&A."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.core.vector_store import get_vector_store
from app.core.session_store import get_session_store
from config import settings
from logging_setup import setup_logger


ROUTER_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    setup_logger()
    logger.info("Starting Tanya Lalin API...")
    
    # Initialize vector store
    vector_store = get_vector_store()
    logger.info(f"Vector store initialized with {vector_store.count()} documents")
    
    # Initialize session store
    get_session_store()
    logger.info("Session store initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Tanya Lalin API...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Tanya Lalin - Indonesian Traffic Law Chat API",
        description=(
            "RAG-based API for answering questions about Indonesian traffic law "
            "(UU No. 22 Tahun 2009 tentang Lalu Lintas dan Angkutan Jalan)"
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # Parse CORS origins from settings
    if settings.cors_origins == "*":
        cors_origins = ["*"]
    else:
        cors_origins = [origin.strip() for origin in settings.cors_origins.split(",")]

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
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
    """Root endpoint with API information."""
    return {
        "name": "Tanya Lalin",
        "description": "Indonesian Traffic Law Chat API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }
