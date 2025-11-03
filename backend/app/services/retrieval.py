from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.services.embeddings import EmbeddingService


class RetrievalService:
    def __init__(self, db: Session):
        self.db = db
        self.embedding_service = EmbeddingService()
    
    def retrieve_relevant_chunks_with_scores(
        self, 
        query: str, 
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[dict]:
        """Retrieve the most relevant text chunks for the given query with similarity scores.
        
        Args:
            query (str): User query to match against document chunks.
            top_k (int): Number of top similar documents to return. Defaults to 5.
            min_similarity (float): Minimum similarity threshold for results. Defaults to 0.1.
            
        Returns:
            List[dics]: List of dictionaries containing document chunks and their similarity scores.
        """
        # Generate embedding for the query
        query_embedding = self.embedding_service.get_embedding(query)
        
        # Use raw SQL to take advantage of pgvector operators
        query_sql = text("""
            SELECT
                source,
                article_number,
                paragraph_number,
                chunk_type,
                text,
                (1 - (embedding <=> CAST(:query_embedding AS vector))) AS cosine_similarity
            FROM legal_text_chunks
            ORDER BY embedding <=> CAST(:query_embedding AS vector)
            LIMIT :top_k
        """)
        
        result = self.db.execute(query_sql, {
            "query_embedding": query_embedding,
            "top_k": top_k
        })
        
        rows = result.fetchall()
        
        chunks_with_scores = []
        for row in rows:
            if row.cosine_similarity >= min_similarity:
                chunk = {
                    "source": row.source,
                    "article_number": row.article_number,
                    "paragraph_number": row.paragraph_number,
                    "chunk_type": row.chunk_type,
                    "text": row.text,
                    "similarity_score": row.cosine_similarity
                }
                chunks_with_scores.append(chunk)
        
        return chunks_with_scores
