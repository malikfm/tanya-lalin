"""API request and response schemas."""
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """Request schema for the chat endpoint."""
    
    session_id: Optional[str] = Field(
        default=None, 
        description="Session ID for conversation continuity. Null for new session."
    )
    message: str = Field(
        ..., 
        description="The user's question or message",
        min_length=1,
        max_length=2000
    )
    top_k: int = Field(
        default=5, 
        ge=1, 
        le=20, 
        description="Number of top similar documents to retrieve"
    )
    min_similarity: float = Field(
        default=0.3, 
        ge=0.0, 
        le=1.0, 
        description="Minimum similarity threshold for document retrieval"
    )


class RetrievedChunk(BaseModel):
    """Schema for a retrieved document chunk."""
    
    source: str = Field(description="Source document identifier")
    article_number: Optional[int] = Field(default=None, description="Article number (Pasal)")
    paragraph_number: Optional[int] = Field(default=None, description="Paragraph number (Ayat)")
    chunk_type: str = Field(default="body", description="Type of chunk: 'body' or 'elucidation'")
    text: str = Field(description="The text content of the chunk")
    similarity_score: float = Field(description="Similarity score (0-1)")


class ChatResponse(BaseModel):
    """Response schema for the chat endpoint."""
    
    session_id: str = Field(description="Session ID for conversation continuity")
    query: str = Field(description="Original user query")
    response: str = Field(description="Generated response")
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="List of relevant document chunks used to generate the response"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )


class ChatMessage(BaseModel):
    """Schema for a chat message in history."""
    
    id: str = Field(description="Message ID")
    role: str = Field(description="Message role: 'user' or 'assistant'")
    content: str = Field(description="Message content")
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Retrieved chunks (for assistant messages)"
    )
    created_at: str = Field(description="Message creation timestamp")


class SessionHistoryResponse(BaseModel):
    """Response schema for session history endpoint."""
    
    session_id: str = Field(description="Session ID")
    messages: list[ChatMessage] = Field(
        default_factory=list,
        description="List of messages in the session"
    )
    created_at: str = Field(description="Session creation timestamp")
    updated_at: str = Field(description="Session last update timestamp")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    vector_store_count: int = Field(description="Number of documents in vector store")
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Response schema for error responses."""
    
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    timestamp: datetime = Field(default_factory=datetime.now)
