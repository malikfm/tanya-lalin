from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    """Simple health check endpoint.
    
    Returns:
        dict: Health status information.
    """
    return {"status": "healthy", "message": "Indonesian Traffic Law RAG API is running"}


@router.get("/ready")
def readiness_check() -> dict:
    """Readiness check endpoint.
    
    Returns:
        dict: Readiness status information.
    """
    # In a production system, you might check database connectivity, cache availability, etc
    return {"status": "ready", "message": "Service is ready to accept requests"}
