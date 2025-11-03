"""Main FastAPI application for the Indonesian Traffic Law Q&A RAG system."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from logging_setup import setup_logger

logger = setup_logger()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Tanya Lalin - Indonesian Traffic Law Chat API",
        description="RAG-based API for answering questions about Indonesian traffic law",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
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
    app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
    app.include_router(health_router, prefix="/api/v1", tags=["health"])

    return app


app = create_app()


@app.get("/")
async def root():
    return {"message": "Tanya Lalin - Indonesian Traffic Law Chat API", "status": "running"}
