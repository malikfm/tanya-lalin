"""Hybrid retrieval service combining vector and keyword search."""
from loguru import logger

from app.core.gemini_client import GeminiClient
from app.core.vector_store import VectorStore
from app.services.query_rewriter import QueryRewriter
from config import settings


class RetrievalService:
    """Service for hybrid document retrieval using vector and keyword search."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        gemini_client: GeminiClient
    ):
        self.vector_store = vector_store
        self.gemini_client = gemini_client
        self.query_rewriter = QueryRewriter(gemini_client)
    
    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_similarity: float | None = None,
        use_query_rewriting: bool = True
    ) -> list[dict]:
        """Retrieve relevant document chunks using hybrid search.
        
        Args:
            query: User query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            use_query_rewriting: Whether to use query rewriting
            
        Returns:
            List of relevant document chunks with scores
        """
        top_k = top_k or settings.final_top_k
        min_similarity = min_similarity or settings.min_similarity
        
        all_results = []
        seen_ids = set()
        
        # Step 1: Query Rewriting
        if use_query_rewriting:
            rewrite_result = await self.query_rewriter.rewrite_query(query)
            legal_search_query = self.query_rewriter.build_expanded_query(rewrite_result)
            additional_queries = self.query_rewriter.get_additional_queries(rewrite_result)
            
            logger.info(f"Original query: {query}")
            logger.info(f"Legal search query: {legal_search_query}")
            logger.info(f"Additional queries: {additional_queries}")
        else:
            legal_search_query = query
            additional_queries = []
        
        # Step 2: Primary Vector Search with legal query (most important!)
        if legal_search_query:
            legal_embedding = await self.gemini_client.generate_embedding(legal_search_query)
            legal_results = self.vector_store.search_by_vector(
                query_embedding=legal_embedding,
                top_k=settings.vector_search_top_k
            )
            for result in legal_results:
                if result["id"] not in seen_ids:
                    result["search_type"] = "legal_query"
                    all_results.append(result)
                    seen_ids.add(result["id"])
            
            logger.debug(f"Legal query search returned {len(legal_results)} results")
        
        # Step 3: Original query vector search
        original_embedding = await self.gemini_client.generate_embedding(query)
        original_results = self.vector_store.search_by_vector(
            query_embedding=original_embedding,
            top_k=settings.vector_search_top_k
        )
        for result in original_results:
            if result["id"] not in seen_ids:
                result["search_type"] = "original_query"
                all_results.append(result)
                seen_ids.add(result["id"])
        
        logger.debug(f"Original query search returned {len(original_results)} results")
        
        # Step 4: Additional queries search
        for add_query in additional_queries[:3]:
            try:
                add_embedding = await self.gemini_client.generate_embedding(add_query)
                add_results = self.vector_store.search_by_vector(
                    query_embedding=add_embedding,
                    top_k=5
                )
                for result in add_results:
                    if result["id"] not in seen_ids:
                        result["search_type"] = f"additional"
                        all_results.append(result)
                        seen_ids.add(result["id"])
                        
                logger.debug(f"Additional query '{add_query[:30]}...' returned {len(add_results)} results")
            except Exception as e:
                logger.warning(f"Error with additional query: {e}")
        
        # Step 5: Apply Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(all_results, k=60)
        
        # Step 6: Filter by similarity threshold and take top_k
        # Use lower threshold since we're using RRF scores
        filtered_results = [
            r for r in fused_results
            if r.get("similarity_score", 0) >= 0.1  # Lower threshold for RRF scores
        ][:top_k]
        
        # Step 7: Format results
        formatted_results = []
        for result in filtered_results:
            metadata = result.get("metadata", {})
            formatted_results.append({
                "source": metadata.get("source", ""),
                "article_number": metadata.get("article_number"),
                "paragraph_number": metadata.get("paragraph_number"),
                "chunk_type": metadata.get("chunk_type", "body"),
                "text": result.get("text", ""),
                "similarity_score": result.get("similarity_score", 0)
            })
        
        logger.info(f"Retrieved {len(formatted_results)} relevant chunks for query")
        return formatted_results
    
    def _reciprocal_rank_fusion(
        self,
        results: list[dict],
        k: int = 60
    ) -> list[dict]:
        """Apply Reciprocal Rank Fusion to a list of ranked results."""
        scores: dict[str, float] = {}
        documents: dict[str, dict] = {}
        
        # Group by search type first
        by_type: dict[str, list] = {}
        for result in results:
            search_type = result.get("search_type", "unknown")
            if search_type not in by_type:
                by_type[search_type] = []
            by_type[search_type].append(result)
        
        # Apply RRF for each search type with weighted scoring
        weights = {
            "legal_query": 2.0,  # Prioritize legal query results
            "original_query": 1.0,
            "additional": 0.8,
        }
        
        for search_type, type_results in by_type.items():
            weight = weights.get(search_type, 0.5)
            for rank, result in enumerate(type_results):
                doc_id = result["id"]
                rrf_score = weight / (k + rank + 1)
                scores[doc_id] = scores.get(doc_id, 0) + rrf_score
                if doc_id not in documents:
                    documents[doc_id] = result
        
        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Build result list with normalized scores
        max_score = max(scores.values()) if scores else 1
        final_results = []
        for doc_id in sorted_ids:
            doc = documents[doc_id].copy()
            doc["similarity_score"] = scores[doc_id] / max_score
            final_results.append(doc)
        
        return final_results
