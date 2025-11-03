"""Main module for chunking legal document items into smaller text chunks with embeddings.

Processes parsed legal document items, splits them into appropriately sized
chunks, generates embeddings for each chunk using Ollama, and validates the results.
The output is a JSONL file containing all chunks with their embeddings ready for
storage in a vector database.

The chunking process:
1. Reads parsed legal document items from JSONL files.
2. Splits text content into overlapping chunks based on token limits.
3. Generates embeddings for each chunk using Ollama.
4. Validates embeddings for quality and consistency.
5. Outputs validated chunks to a JSONL file.
"""
import argparse
import json
import re
from pathlib import Path
from typing import List

import ollama
from pydantic import ValidationError

from enums import LegalTextType
from ingestions.chunker.validation import run_all_validations
from ingestions.io import write_to_jsonl
from ingestions.models import LegalDocumentItem, LegalTextChunkDTO
from logging_setup import setup_logger

EMBEDDING_MODEL = "embeddinggemma"
OUTPUT_FILE_NAME = "chunks_with_embeddings.jsonl"
LEGAL_TEXT_SOURCE = "UU_22_2009_LLAJ"

logger = setup_logger()


def tokenize(text: str) -> List[str]:
    """Tokenize text using word boundaries and common punctuation.
    
    This function provides tokenization by considering both
    words and punctuation separately, which is important for Indonesian
    legal text that may contain various punctuation patterns.
    
    Args:
        text (str): Input text to tokenize.
        
    Returns:
        List[str]: List of tokens extracted from the text.
    """
    # Considers punctuation and words separately
    tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
    return tokens


def make_chunks(
    text: str,
    max_chunk_tokens: int = 300,
    overlap_tokens: int = 30
) -> List[str]:
    """Split text into chunks respecting token limits with overlap.
    
    This function implements a sliding window approach to text chunking,
    ensuring that chunks don't exceed the specified token limit while
    maintaining contextual continuity through token overlap.
    
    Args:
        text (str): Input text to chunk.
        max_chunk_tokens (int): Maximum number of tokens per chunk. Defaults to 300.
        overlap_tokens (int): Number of tokens to overlap between consecutive chunks.
            Defaults to 30.
        
    Returns:
        List[str]: List of text chunks, each within the token limit.
    """
    tokens = tokenize(text)
    total_tokens = len(tokens)
    
    # If text fits in one chunk, return it
    if total_tokens <= max_chunk_tokens:
        return [text]
    
    # Need to split into multiple chunks with overlap
    chunks = []
    start_idx = 0
    
    while start_idx < total_tokens:
        # Calculate end index for this chunk
        end_idx = min(start_idx + max_chunk_tokens, total_tokens)
        
        # Get chunk tokens
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = " ".join(chunk_tokens)
        chunks.append(chunk_text)
        
        if end_idx >= total_tokens:
            break
        start_idx = end_idx - overlap_tokens
    
    return chunks


def get_embedding(text: str) -> List[float]:
    """Generate embedding vector for text using Ollama.
    
    This function calls the Ollama embedding service to generate a numerical
    vector representation of the input text, which can be used for semantic
    similarity comparisons.
    
    Args:
        text (str): Input text to embed.
        
    Returns:
        List[float]: Embedding vector representing the semantic meaning of the text.
        
    Raises:
        Exception: If the Ollama service is unavailable or returns an error.
    """
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    return response["embeddings"][0]


def process_item(
    legal_document_item: LegalDocumentItem,
    chunk_type: str
) -> List[LegalTextChunkDTO]:
    """Process a single legal document item into chunks with embeddings.
    
    This function takes a parsed legal document item (representing a single
    article or paragraph), splits its text content into appropriate chunks,
    generates embeddings for each chunk, and packages everything with
    appropriate metadata.
    
    Args:
        legal_document_item (LegalDocumentItem): Legal document item containing 
            article_number, paragraph_number, and text.
        chunk_type (str): Type of chunk, either 'body' or 'elucidation'.
        
    Returns:
        List[LegalTextChunkDTO]: List of LegalTextChunkDTO objects with metadata and embeddings.
    """
    text = legal_document_item.text
    article_number = legal_document_item.article_number
    paragraph_number = legal_document_item.paragraph_number
    
    # Chunk the text
    text_chunks = make_chunks(text)
    
    # Create records for each chunk
    chunk_objects = []
    for i, text_chunk in enumerate(text_chunks):
        # Generate embedding
        embedding = get_embedding(text_chunk)
        chunk_objects.append(
            LegalTextChunkDTO(
                source=LEGAL_TEXT_SOURCE,
                article_number=article_number,
                paragraph_number=paragraph_number,
                chunk_index=i,
                chunk_type=chunk_type,
                text=text_chunk,
                embedding=embedding,
                token_count=len(tokenize(text_chunk))
            )
        )
    
    return chunk_objects


def process_file(
    input_path: Path,
    chunk_type: str
) -> List[LegalTextChunkDTO]:
    """Process a JSONL file and generate chunks for all legal document items.
    
    This function reads a JSONL file containing serialized LegalDocumentItem objects,
    processes each item to generate chunks with embeddings, and returns all
    processed chunks.
    
    Args:
        input_path (Path): Path to input JSONL file containing legal document items.
        chunk_type (str): Type of chunks to generate, either 'body' or 'elucidation'.
        
    Returns:
        List[LegalTextChunkDTO]: List of all chunk records (LegalTextChunkDTO objects) from the file.
        
    Raises:
        ValidationError: If any line in the file contains invalid JSON.
    """
    logger.info(f"Processing {input_path} as {chunk_type}...")
    all_chunks = []
    with open(input_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                legal_document_item = LegalDocumentItem.model_validate_json(line)
                
                # Process this legal document item
                chunk_objects = process_item(legal_document_item, chunk_type)
                all_chunks.extend(chunk_objects)
                
                logger.info(
                    f"Line {line_num}: Article {legal_document_item.article_number}"
                    f", Paragraph {legal_document_item.paragraph_number}"
                    f" -> {len(chunk_objects)} chunk(s)"
                )
            except ValidationError as e:
                logger.error(f"Error parsing line {line_num}: {e}")
                raise
    
    return all_chunks


def main():
    args_parser = argparse.ArgumentParser(
        description="Chunk parsed legal document items and generate embeddings"
    )
    args_parser.add_argument(
        "--body-file-path", 
        required=True, 
        help="Path to the JSONL file containing parsed body content"
    )
    args_parser.add_argument(
        "--elucidation-file-path", 
        required=True, 
        help="Path to the JSONL file containing parsed elucidation content"
    )
    args_parser.add_argument(
        "--output-dir", 
        required=True, 
        help="Directory where the output JSONL file will be saved"
    )
    args = args_parser.parse_args()

    body_file_path = Path(args.body_file_path)
    elucidation_file_path = Path(args.elucidation_file_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not body_file_path.exists():
        raise FileNotFoundError(f"Body file does not exist: {body_file_path}")
    
    if not elucidation_file_path.exists():
        raise FileNotFoundError(f"Elucidation file does not exist: {elucidation_file_path}")
    
    all_chunks = process_file(body_file_path, LegalTextType.BODY.value)
    all_chunks.extend(
        process_file(elucidation_file_path, LegalTextType.ELUCIDATION.value)
    )
    
    # Run validation checks before saving results
    logger.info("Running validation checks on chunks...")
    validation_passed = run_all_validations(all_chunks)
    
    if not validation_passed:
        logger.error("Validation failed. Please review the issues above.")
        return
    
    # Save results
    output_path = output_dir.joinpath(OUTPUT_FILE_NAME)
    write_to_jsonl(map(lambda chunk_obj: chunk_obj.model_dump_json(), all_chunks), output_path)
    
    # Print summary
    logger.info("Chunking complete!")
    logger.info("Summary:")
    logger.info(f"Total chunks created: {len(all_chunks)}")
    
    # Count chunks by type
    body_chunks = [chunk_obj for chunk_obj in all_chunks if chunk_obj.chunk_type == LegalTextType.BODY.value]
    elucidation_chunks = [chunk_obj for chunk_obj in all_chunks if chunk_obj.chunk_type == LegalTextType.ELUCIDATION.value]
    logger.info(f"Body chunks: {len(body_chunks)}")
    logger.info(f"Elucidation chunks: {len(elucidation_chunks)}")
    logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
