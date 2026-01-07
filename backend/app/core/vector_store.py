"""ChromaDB vector store for document retrieval."""
import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from pathlib import Path
from typing import Any

from config import settings


class VectorStore:
    """ChromaDB-based vector store for legal document chunks."""
    
    def __init__(self):
        """Initialize ChromaDB client and collection."""
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"ChromaDB initialized with {self.collection.count()} documents")
    
    def add_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]]
    ) -> None:
        """Add documents to the vector store.
        
        Args:
            ids: Unique document IDs
            embeddings: Document embedding vectors
            documents: Document text content
            metadatas: Document metadata
        """
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Added {len(ids)} documents to vector store")
    
    def search_by_vector(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict | None = None
    ) -> list[dict]:
        """Search documents by vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            where: Optional filter conditions
            
        Returns:
            List of matching documents with scores
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to list of dicts with similarity scores
        documents = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distance, convert to similarity (1 - distance for cosine)
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance
                
                documents.append({
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity_score": similarity
                })
        
        return documents
    
    def search_by_text(
        self,
        query_text: str,
        top_k: int = 10,
        where: dict | None = None,
        where_document: dict | None = None
    ) -> list[dict]:
        """Search documents by text content (keyword search).
        
        Args:
            query_text: Search query text
            top_k: Number of results to return  
            where: Optional metadata filter conditions
            where_document: Optional document content filter
            
        Returns:
            List of matching documents
        """
        # ChromaDB's where_document uses $contains for text search
        if where_document is None:
            where_document = {"$contains": query_text}
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where,
                where_document=where_document,
                include=["documents", "metadatas", "distances"]
            )
            
            documents = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0
                    similarity = 1 - distance
                    
                    documents.append({
                        "id": doc_id,
                        "text": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "similarity_score": similarity
                    })
            
            return documents
            
        except Exception as e:
            logger.warning(f"Text search failed: {e}")
            return []
    
    def get_all_documents(self) -> list[dict]:
        """Get all documents from the collection.
        
        Returns:
            List of all documents with metadata
        """
        results = self.collection.get(
            include=["documents", "metadatas"]
        )
        
        documents = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                documents.append({
                    "id": doc_id,
                    "text": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {}
                })
        
        return documents
    
    def count(self) -> int:
        """Get total number of documents in collection."""
        return self.collection.count()
    
    def delete_all(self) -> None:
        """Delete all documents from collection."""
        # Get all IDs first
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)
        logger.info("Deleted all documents from vector store")


# Singleton instance
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get or create vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
