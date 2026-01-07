"""Health check API endpoints."""
from fastapi import APIRouter, Depends

from app.schemas import HealthResponse
from app.core.vector_store import get_vector_store, VectorStore


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Check the health status of the API.
    
    Returns:
        HealthResponse with service status and document count
    """
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        vector_store_count=vector_store.count()
    )
