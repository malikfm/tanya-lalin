from typing import List

import numpy as np
import ollama
from loguru import logger
from scipy.spatial.distance import cosine

from config import settings
from ingestions.models import LegalTextChunkDTO


def validate_embeddings_sanity(chunks: List[LegalTextChunkDTO]) -> bool:
    """Perform sanity checks on chunk embeddings to ensure numerical validity.
    
    This function validates that all embeddings:
    1. Exist and are not empty.
    2. Have the correct dimension (matching settings.embedding_dim).
    3. Contain only finite numerical values (no NaNs or infinities).
    4. Don't have extreme values that might indicate processing errors.
    
    Args:
        chunks (List[LegalTextChunkDTO]): List of LegalTextChunkDTO objects.
        
    Returns:
        bool: True if all embeddings pass validation, False otherwise.
    """
    logger.info("Starting sanity validation of embeddings...")
    
    # Get expected embedding dimension from config
    expected_dim = settings.embedding_dim
    logger.info(f"Expected embedding dimension: {expected_dim}")
    
    all_valid = True
    for i, chunk_obj in enumerate(chunks):
        embedding = chunk_obj.embedding
        
        # Check if embedding exists
        if not embedding:
            logger.error(f"Chunk {i}: No embedding found")
            all_valid = False
            continue
            
        # Check dimension
        if len(embedding) != expected_dim:
            logger.error(f"Chunk {i}: Dimension mismatch - got {len(embedding)}, expected {expected_dim}")
            all_valid = False
            continue
            
        # Convert to numpy array for validation
        try:
            embedding_array = np.array(embedding)
        except Exception as e:
            logger.error(f"Chunk {i}: Failed to convert embedding to numpy array: {e}")
            all_valid = False
            continue
            
        # Check for NaNs
        if np.isnan(embedding_array).any():
            logger.error(f"Chunk {i}: Contains NaN values")
            all_valid = False
            continue
            
        # Check for infinite values
        if np.isinf(embedding_array).any():
            logger.error(f"Chunk {i}: Contains infinite values")
            all_valid = False
            continue
            
        # Check for extreme values (values too large might indicate issues)
        if np.abs(embedding_array).max() > 1e6:
            logger.warning(
                f"Chunk {i}: Contains extremely large values (max: {np.abs(embedding_array).max()})."
                " Values too large might indicate issues, if it's expected, you can ignore this warning."
            )
            
        # Check consistency of shape
        if embedding_array.shape != (expected_dim,):
            logger.error(f"Chunk {i}: Shape mismatch - got {embedding_array.shape}, expected ({expected_dim},)")
            all_valid = False
            continue
    
    if all_valid:
        logger.info("Sanity validation passed: All embeddings are valid")
    else:
        logger.error("Sanity validation failed: Some embeddings have issues")
        
    return all_valid


def find_chunks_by_topic(chunks: List[LegalTextChunkDTO], keywords: List[str]) -> List[LegalTextChunkDTO]:
    """Find chunks containing specific keywords for validation purposes.
    
    This helper function searches through chunks to find those containing
    any of the specified keywords, which is useful for semantic validation
    by grouping semantically related chunks.
    
    Args:
        chunks (List[LegalTextChunkDTO]): List of LegalTextChunkDTO objects to search through.
        keywords (List[str]): List of keywords to search for (case-insensitive).
        
    Returns:
        List[dict]: List of LegalTextChunkDTO objects containing any of the specified keywords.
    """
    matching_chunks = []
    
    for chunk in chunks:
        if any(keyword.lower() in chunk.text.lower() for keyword in keywords):
            matching_chunks.append(chunk)
    
    return matching_chunks


def validate_semantic_consistency(chunks: List[LegalTextChunkDTO]) -> bool:
    """Validate semantic consistency by checking similarity of related chunks.
    
    This function tests whether semantically similar chunks have high cosine
    similarity compared to unrelated chunks. It works by:
    1. Finding chunks about similar topics (e.g., sanctions/punishments).
    2. Finding chunks about different topics (e.g., taxes vs traffic regulations).
    3. Comparing cosine similarities within and between topic groups.
    
    Args:
        chunks (List[LegalTextChunkDTO]): List of LegalTextChunkDTO objects.
        
    Returns:
        bool: True if semantic consistency validation passes, False otherwise.
    """
    logger.info("Starting semantic consistency validation...")
    
    # Try to find chunks about similar topics
    # Looking for Indonesian legal terms: "sanksi" (sanction), "pidana" (criminal), "peraturan" (regulation)
    # Find chunks related to sanctions/punishments
    sanksi_chunks = find_chunks_by_topic(chunks, ["sanksi", "pidana", "denda", "hukuman"])
    pajak_chunks = find_chunks_by_topic(chunks, ["pajak", "bea", "retribusi"])
    
    # Test similarity within same topic (should have high similarity)
    if len(sanksi_chunks) >= 2:
        logger.info(f"Found {len(sanksi_chunks)} chunks related to sanctions/punishments")
        
        # Compare first few chunks within the same topic
        for i in range(min(len(sanksi_chunks), 3)):
            for j in range(i + 1, min(len(sanksi_chunks), 3)):
                if i < len(sanksi_chunks) and j < len(sanksi_chunks):
                    emb1 = np.array(sanksi_chunks[i].embedding)
                    emb2 = np.array(sanksi_chunks[j].embedding)
                    
                    logger.info(f"Comparing between\nchunk {i}: {sanksi_chunks[i].embedding}\nchunk {j}: {sanksi_chunks[j].embedding}")
                    cosine_sim = 1 - cosine(emb1, emb2)
                    logger.info(f"Similarity: {cosine_sim:.4f}")
                    
                    # For similar topics, expect at least moderate similarity
                    if cosine_sim < 0.5:
                        logger.error(f"Low similarity between similar topic chunks: {cosine_sim:.4f}")
                        return False
    
    # Test similarity between different topics (should have lower similarity)
    if len(sanksi_chunks) > 0 and len(pajak_chunks) > 0:
        logger.info(f"Testing similarity between different topics: {len(sanksi_chunks)} sanksi chunks vs {len(pajak_chunks)} pajak chunks")
        
        # Compare chunks from different topics
        for i in range(min(len(sanksi_chunks), 2)):
            for j in range(min(len(pajak_chunks), 2)):
                emb1 = np.array(sanksi_chunks[i].embedding)
                emb2 = np.array(pajak_chunks[j].embedding)
                
                logger.info(f"Comparing between\nchunk: {sanksi_chunks[i].text}\nchunk: {pajak_chunks[j].text}")
                cosine_sim = 1 - cosine(emb1, emb2)
                logger.info(f"Similarity: {cosine_sim:.4f}")
                
        # For different topics, it should be generally lower than within-topic similarity
        logger.info(f"Note: Different topic similarity is expected to be lower than same-topic")
    
    logger.info("Semantic consistency validation completed")
    return True


def retrieve_top_k_chunks(query_embedding: List[float], chunks: List[LegalTextChunkDTO], k: int = 5) -> List[LegalTextChunkDTO]:
    """Retrieve top-k most similar chunks to a query embedding using cosine similarity.
    
    This helper function performs similarity search by computing cosine similarity
    between a query embedding and all chunk embeddings, then returning the top-k
    most similar chunks.
    
    Args:
        query_embedding (List[float]): Query embedding vector.
        chunks (List[LegalTextChunkDTO]): List of LegalTextChunkDTO objects.
        k (int): Number of top similar chunks to return. Defaults to 5.
        
    Returns:
        List[dict]: List of top-k most similar LegalTextChunkDTO objects sorted by similarity score.
    """
    query_array = np.array(query_embedding)
    
    # Calculate similarities with all chunks
    similarities = []
    for chunk in chunks:
        chunk_emb = np.array(chunk.embedding)
        sim = 1 - cosine(query_array, chunk_emb)  # Convert distance to similarity
        similarities.append((sim, chunk))
    
    # Sort by similarity (descending) and return top-k
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in similarities[:k]]


def validate_retrieval(chunks: List[LegalTextChunkDTO]) -> None:
    """Validate retrieval using sample Indonesian legal queries.
    
    This function tests the embedding-based retrieval system by simulating user queries
    and checking if relevant chunks are retrieved. It helps ensure (manual validation)
    that the embeddings capture semantic meaning appropriately for the intended use case.
    
    Args:
        chunks (List[LegalTextChunkDTO]): List of LegalTextChunkDTO objects.
        
    Returns:
        bool: True if retrieval validation passes, False otherwise.
    """
    logger.info("Starting retrieval validation...")
    logger.info("Please see the output and make sure that the retrieved chunks are relevant")
    
    # Sample Indonesian queries related to traffic law
    sample_queries = [
        "Apa tanggung jawab negara dalam lalu lintas jalan?",
        "Apa saja sanksi pidana dalam undang-undang lalu lintas?",
        "Bagaimana prosedur pelanggaran lalu lintas?",
        "Apa hak dan kewajiban pengguna jalan?"
    ]
    
    for query in sample_queries:
        logger.info(f"Testing query: {query}")
        
        try:
            # Generate embedding for query
            query_embedding_response = ollama.embed(model=settings.embedding_model, input=query)
            query_embedding = query_embedding_response["embeddings"][0]
            
            # Retrieve top-5 chunks
            top_chunks = retrieve_top_k_chunks(query_embedding, chunks, k=5)
            
            # Print retrieved chunks for manual validation
            for j, chunk in enumerate(top_chunks):
                logger.info(f"Retrieved chunk {j + 1} (similarity: {1 - cosine(np.array(query_embedding), np.array(chunk.embedding)):.4f}):")
                logger.info(
                    f"Article: {chunk.article_number}"
                    f", Paragraph: {chunk.paragraph_number or 'N/A'}"
                    f", Type: {chunk.chunk_type}"
                )
                logger.info(f"Text: {chunk.text}")
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            raise
    
    logger.info("Retrieval validation completed")


def run_all_validations(chunks: List[LegalTextChunkDTO]) -> bool:
    """Run all validation checks on chunk embeddings.
    
    This function orchestrates all validation checks to ensure the quality
    and consistency of generated embeddings. It performs:
    1. Numerical sanity checks.
    2. Semantic consistency validation.
    3. Retrieval tests.
    
    Args:
        chunks (List[LegalTextChunkDTO]): List of LegalTextChunkDTO objects to validate.
        
    Returns:
        bool: True if all validations pass, False otherwise.
    """
    logger.info("Running all validation checks on chunks...")
    logger.info(f"Total chunks to validate: {len(chunks)}")
    
    # Sanity check
    sanity_valid = validate_embeddings_sanity(chunks)
    
    # Semantic consistency
    semantic_valid = validate_semantic_consistency(chunks)
    
    # Retrieval validation
    validate_retrieval(chunks)
    
    all_valid = sanity_valid and semantic_valid
    
    logger.info(f"Validation Summary:")
    logger.info(f"Sanity validation: {'PASSED' if sanity_valid else 'FAILED'}")
    logger.info(f"Semantic consistency validation: {'PASSED' if semantic_valid else 'SKIPPED/ISSUES'}")
    logger.info(f"Overall validation: {'PASSED' if all_valid else 'FAILED'}")
    
    return all_valid
