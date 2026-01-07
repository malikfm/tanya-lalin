"""Chat service orchestrating the RAG pipeline."""
from loguru import logger

from app.core.gemini_client import GeminiClient
from app.core.vector_store import VectorStore
from app.core.session_store import ChatSession, SessionStore
from app.services.retrieval import RetrievalService
from app.services.llm import LLMService
from config import settings


class ChatService:
    """Main service for handling chat interactions."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        gemini_client: GeminiClient,
        session_store: SessionStore
    ):
        self.vector_store = vector_store
        self.gemini_client = gemini_client
        self.session_store = session_store
        self.retrieval_service = RetrievalService(vector_store, gemini_client)
        self.llm_service = LLMService(gemini_client)
    
    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None
    ) -> dict:
        """Process a chat message and return response.
        
        Args:
            message: User message
            session_id: Optional session ID for conversation continuity
            top_k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary with response, session info, and retrieved chunks
        """
        # Get or create session
        session = self.session_store.get_or_create_session(session_id)
        
        # Add user message to session
        session.add_message(role="user", content=message)
        
        # Get conversation history for context
        history = session.get_context_messages(settings.max_context_messages)
        # Remove the current message from history (it's the last one)
        history = history[:-1] if history else []
        
        logger.info(f"Processing chat message in session {session.id}")
        
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = await self.retrieval_service.retrieve(
            query=message,
            top_k=top_k,
            min_similarity=min_similarity,
            use_query_rewriting=True
        )
        
        # Step 2: Generate response
        response_text = await self.llm_service.generate_response(
            query=message,
            retrieved_chunks=retrieved_chunks,
            conversation_history=history
        )
        
        # Add assistant response to session
        session.add_message(
            role="assistant",
            content=response_text,
            retrieved_chunks=retrieved_chunks
        )
        
        return {
            "session_id": session.id,
            "query": message,
            "response": response_text,
            "retrieved_chunks": retrieved_chunks
        }
    
    def get_session_history(self, session_id: str) -> dict | None:
        """Get conversation history for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data with messages, or None if not found
        """
        session = self.session_store.get_session(session_id)
        if session:
            return session.to_dict()
        return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        return self.session_store.delete_session(session_id)
