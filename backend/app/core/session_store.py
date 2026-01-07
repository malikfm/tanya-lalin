"""In-memory session store for chat conversations."""
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4
from loguru import logger

from config import settings


class ChatMessage:
    """Represents a single chat message."""
    
    def __init__(
        self,
        role: str,
        content: str,
        retrieved_chunks: list[dict] | None = None
    ):
        self.id = str(uuid4())
        self.role = role  # "user" or "assistant"
        self.content = content
        self.retrieved_chunks = retrieved_chunks or []
        self.created_at = datetime.now()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "retrieved_chunks": self.retrieved_chunks,
            "created_at": self.created_at.isoformat()
        }


class ChatSession:
    """Represents a chat session with message history."""
    
    def __init__(self, session_id: str | None = None):
        self.id = session_id or str(uuid4())
        self.messages: list[ChatMessage] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_message(
        self,
        role: str,
        content: str,
        retrieved_chunks: list[dict] | None = None
    ) -> ChatMessage:
        """Add a message to the session.
        
        Args:
            role: Message role ("user" or "assistant")
            content: Message content
            retrieved_chunks: Retrieved document chunks (for assistant messages)
            
        Returns:
            The created ChatMessage
        """
        message = ChatMessage(role, content, retrieved_chunks)
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def get_context_messages(self, max_messages: int | None = None) -> list[dict]:
        """Get recent messages for context.
        
        Args:
            max_messages: Maximum number of messages to return
            
        Returns:
            List of message dictionaries
        """
        limit = max_messages or settings.max_context_messages
        recent_messages = self.messages[-limit:]
        return [msg.to_dict() for msg in recent_messages]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class SessionStore:
    """In-memory session store with TTL-based expiration."""
    
    def __init__(self):
        self._sessions: dict[str, ChatSession] = {}
        self._ttl_hours = settings.session_ttl_hours
    
    def create_session(self) -> ChatSession:
        """Create a new chat session.
        
        Returns:
            New ChatSession instance
        """
        session = ChatSession()
        self._sessions[session.id] = session
        logger.debug(f"Created new session: {session.id}")
        return session
    
    def get_session(self, session_id: str) -> ChatSession | None:
        """Get a session by ID.
        
        Args:
            session_id: Session ID to retrieve
            
        Returns:
            ChatSession if found and not expired, None otherwise
        """
        session = self._sessions.get(session_id)
        
        if session:
            # Check if session has expired
            expiry_time = session.updated_at + timedelta(hours=self._ttl_hours)
            if datetime.now() > expiry_time:
                self.delete_session(session_id)
                return None
        
        return session
    
    def get_or_create_session(self, session_id: str | None = None) -> ChatSession:
        """Get existing session or create new one.
        
        Args:
            session_id: Optional session ID to retrieve
            
        Returns:
            ChatSession (existing or new)
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        return self.create_session()
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if session was deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Deleted session: {session_id}")
            return True
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """Remove all expired sessions.
        
        Returns:
            Number of sessions removed
        """
        now = datetime.now()
        expired_ids = []
        
        for session_id, session in self._sessions.items():
            expiry_time = session.updated_at + timedelta(hours=self._ttl_hours)
            if now > expiry_time:
                expired_ids.append(session_id)
        
        for session_id in expired_ids:
            del self._sessions[session_id]
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired sessions")
        
        return len(expired_ids)


# Singleton instance
_session_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    """Get or create session store singleton."""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store
