"""Data schemas for API requests and responses."""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ChatRequest(BaseModel):
    """Request schema for the chat endpoint.
    
    Attributes:
        message (str): User's input message/question.
        top_k (int): Number of top similar documents to retrieve. Defaults to 5.
        min_similarity (float): Minimum similarity threshold for document retrieval. Defaults to 0.3.
    """
    message: str = Field(..., description="The user's question or message", min_length=1)
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top similar documents to retrieve")
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum similarity threshold for document retrieval")


class RetrievedChunk(BaseModel):
    """Schema for a retrieved document chunk.
    
    Attributes:
        source (str): Source document identifier.
        article_number (int): Legal article number (Pasal).
        paragraph_number (Optional[int]): Legal paragraph number (Ayat), if applicable.
        chunk_type (str): Type of chunk ('body' or 'elucidation').
        text (str): The actual text content of the chunk.
        similarity_score (float): Similarity score between the query and this chunk.
    """
    source: str
    article_number: int
    paragraph_number: Optional[int]
    chunk_type: str
    text: str
    similarity_score: float


class ChatResponse(BaseModel):
    """Response schema for the chat endpoint.
    
    Attributes:
        query (str): Original user query.
        response (str): Generated response based on retrieved documents.
        retrieved_chunks (List[RetrievedChunk]): List of relevant document chunks used to generate the response.
        timestamp (datetime): Time when the response was generated.
    """
    query: str
    response: str
    retrieved_chunks: List[RetrievedChunk]
    timestamp: datetime = Field(default_factory=datetime.now)
