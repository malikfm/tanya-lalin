"""Chat API endpoints."""
from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from app.schemas import (
    ChatRequest, 
    ChatResponse, 
    SessionHistoryResponse,
    RetrievedChunk,
    ChatMessage,
    ErrorResponse
)
from app.services.chat import ChatService
from app.core.gemini_client import get_gemini_client, GeminiClient
from app.core.vector_store import get_vector_store, VectorStore
from app.core.session_store import get_session_store, SessionStore
from app.constants import ResponseMessages


router = APIRouter()


def get_chat_service(
    vector_store: VectorStore = Depends(get_vector_store),
    gemini_client: GeminiClient = Depends(get_gemini_client),
    session_store: SessionStore = Depends(get_session_store)
) -> ChatService:
    """Dependency injection for ChatService."""
    return ChatService(vector_store, gemini_client, session_store)


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}}
)
async def chat_endpoint(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Process a chat message and return AI-generated response.
    
    This endpoint:
    1. Takes a user query and optional session ID
    2. Rewrites the query with legal terminology
    3. Retrieves relevant legal document chunks using hybrid search
    4. Generates a response using LLM with retrieved context
    5. Returns the response with source citations
    
    Args:
        request: Chat request with message and optional parameters
        
    Returns:
        ChatResponse with generated response and retrieved chunks
    """
    try:
        result = await chat_service.chat(
            message=request.message,
            session_id=request.session_id,
            top_k=request.top_k,
            min_similarity=request.min_similarity
        )
        
        # Convert retrieved chunks to response format
        retrieved_chunks = [
            RetrievedChunk(
                source=chunk.get("source", ""),
                article_number=chunk.get("article_number"),
                paragraph_number=chunk.get("paragraph_number"),
                chunk_type=chunk.get("chunk_type", "body"),
                text=chunk.get("text", ""),
                similarity_score=chunk.get("similarity_score", 0)
            )
            for chunk in result.get("retrieved_chunks", [])
        ]
        
        # Don't show sources if response is a "not found" message
        response_text = result["response"]
        not_found_responses = (
            ResponseMessages.NOT_FOUND,
            ResponseMessages.NO_RELEVANT_CHUNKS,
            ResponseMessages.ERROR
        )
        if response_text in not_found_responses:
            retrieved_chunks = []
        
        return ChatResponse(
            session_id=result["session_id"],
            query=result["query"],
            response=response_text,
            retrieved_chunks=retrieved_chunks
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan saat memproses permintaan. Silakan coba lagi."
        )


@router.get(
    "/chat/{session_id}/history",
    response_model=SessionHistoryResponse,
    responses={404: {"model": ErrorResponse}}
)
async def get_session_history(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Get conversation history for a session.
    
    Args:
        session_id: The session ID to retrieve history for
        
    Returns:
        SessionHistoryResponse with all messages in the session
    """
    history = chat_service.get_session_history(session_id)
    
    if history is None:
        raise HTTPException(
            status_code=404,
            detail="Session tidak ditemukan atau sudah kadaluarsa."
        )
    
    # Convert messages to response format
    messages = [
        ChatMessage(
            id=msg["id"],
            role=msg["role"],
            content=msg["content"],
            retrieved_chunks=[
                RetrievedChunk(
                    source=chunk.get("source", ""),
                    article_number=chunk.get("article_number"),
                    paragraph_number=chunk.get("paragraph_number"),
                    chunk_type=chunk.get("chunk_type", "body"),
                    text=chunk.get("text", ""),
                    similarity_score=chunk.get("similarity_score", 0)
                )
                for chunk in msg.get("retrieved_chunks", [])
            ],
            created_at=msg["created_at"]
        )
        for msg in history.get("messages", [])
    ]
    
    return SessionHistoryResponse(
        session_id=history["id"],
        messages=messages,
        created_at=history["created_at"],
        updated_at=history["updated_at"]
    )


@router.delete(
    "/chat/{session_id}",
    responses={404: {"model": ErrorResponse}}
)
async def delete_session(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Delete a chat session and its history.
    
    Args:
        session_id: The session ID to delete
        
    Returns:
        Success message
    """
    deleted = chat_service.delete_session(session_id)
    
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Session tidak ditemukan."
        )
    
    return {"message": "Session berhasil dihapus.", "session_id": session_id}