import ollama
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.ollama_client import get_ollama_client
from app.schemas import ChatRequest, ChatResponse, RetrievedChunk
from app.services.retrieval import RetrievalService
from app.services.llm import LLMService

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db),
    ollama_client: ollama.AsyncClient = Depends(get_ollama_client)
):
    """Chat endpoint that processes user queries and returns relevant legal information.
    
    This endpoint:
    1. Takes a user query and retrieves relevant legal text chunks.
    2. Uses an LLM to generate a response based on the relevant chunks.
    3. Returns both the response and the chunks used for context.
    
    Args:
        request (ChatRequest): The user's question and retrieval parameters.
        db (Session): Database session dependency.
        
    Returns:
        ChatResponse: The generated response with relevant chunks.
        
    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        # Initialize services
        retrieval_service = RetrievalService(db)
        llm_service = LLMService(ollama_client)
        
        # Retrieve relevant chunks based on the query
        relevant_chunks_with_scores = retrieval_service.retrieve_relevant_chunks_with_scores(
            query=request.message,
            top_k=request.top_k,
            min_similarity=request.min_similarity
        )
        
        # Generate response using LLM
        response_text = await llm_service.generate_response(request.message, relevant_chunks_with_scores)
        
        formatted_chunks = []
        for chunk in relevant_chunks_with_scores:
            formatted_chunks.append(RetrievedChunk(
                source=chunk["source"],
                article_number=chunk["article_number"],
                paragraph_number=chunk["paragraph_number"],
                chunk_type=chunk["chunk_type"],
                text=chunk["text"],
                similarity_score=chunk["similarity_score"]
            ))
        
        response = ChatResponse(
            query=request.message,
            response=response_text,
            retrieved_chunks=formatted_chunks
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred. Please try again later.")


@router.post("/simple-chat")
async def simple_chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """Simple chat endpoint that returns retrieved chunks without LLM processing.
    
    This endpoint retrieves relevant chunks but doesn't generate an LLM response,
    useful for getting raw document chunks.
    
    Args:
        request (ChatRequest): The user's question and retrieval parameters.
        db (Session): Database session dependency.
        
    Returns:
        dict: The retrieved chunks with minimal processing.
    """
    try:
        retrieval_service = RetrievalService(db)
        
        relevant_chunks_with_scores = retrieval_service.retrieve_relevant_chunks_with_scores(
            query=request.message,
            top_k=request.top_k,
            min_similarity=request.min_similarity
        )
        
        formatted_chunks = []
        for chunk in relevant_chunks_with_scores:
            formatted_chunks.append(RetrievedChunk(
                source=chunk["source"],
                article_number=chunk["article_number"],
                paragraph_number=chunk["paragraph_number"],
                chunk_type=chunk["chunk_type"],
                text=chunk["text"],
                similarity_score=chunk["similarity_score"]
            ))
        
        response = ChatResponse(
            query=request.message,
            retrieved_chunks=formatted_chunks
        )

        return response
    
    except Exception as e:
        logger.error(f"Error processing simple chat request: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred. Please try again later.")
    