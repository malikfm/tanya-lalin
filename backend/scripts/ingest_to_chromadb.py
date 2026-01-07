"""Ingestion script to load legal document chunks into ChromaDB."""
import asyncio
import json
import sys
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from google import genai
from config import settings
from logging_setup import setup_logger


async def load_chunks_to_chromadb(
    body_file: Path,
    elucidation_file: Path,
    batch_size: int = 50
) -> None:
    """Load legal document chunks from JSONL files into ChromaDB.
    
    Args:
        body_file: Path to body content JSONL file
        elucidation_file: Path to elucidation content JSONL file
        batch_size: Number of documents to process in each batch
    """
    setup_logger()
    
    # Initialize Gemini client for embeddings
    client = genai.Client(api_key=settings.gemini_api_key)
    
    # Initialize ChromaDB
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    
    # Delete existing collection and create new one
    try:
        chroma_client.delete_collection(settings.chroma_collection_name)
        logger.info(f"Deleted existing collection: {settings.chroma_collection_name}")
    except Exception:
        pass
    
    collection = chroma_client.create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    logger.info(f"Created collection: {settings.chroma_collection_name}")
    
    # Load documents from files
    documents = []
    
    # Load body content
    logger.info(f"Loading body content from {body_file}...")
    with open(body_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                doc["chunk_type"] = "body"
                documents.append(doc)
    
    # Load elucidation content
    logger.info(f"Loading elucidation content from {elucidation_file}...")
    with open(elucidation_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                doc["chunk_type"] = "elucidation"
                documents.append(doc)
    
    logger.info(f"Total documents to process: {len(documents)}")
    
    # Process in batches
    for i in tqdm(range(0, len(documents), batch_size), desc="Processing batches"):
        batch = documents[i:i + batch_size]
        
        ids = []
        texts = []
        metadatas = []
        
        for doc in batch:
            article = doc.get("article_number", 0)
            paragraph = doc.get("paragraph_number")
            chunk_type = doc.get("chunk_type", "body")
            text = doc.get("text", "")
            
            # Create unique ID
            para_str = f"_p{paragraph}" if paragraph else ""
            doc_id = f"art{article}{para_str}_{chunk_type}"
            
            ids.append(doc_id)
            texts.append(text)
            
            # ChromaDB doesn't accept None values, so we need to handle paragraph_number
            metadata = {
                "source": "UU_22_2009_LLAJ",
                "article_number": article,
                "chunk_type": chunk_type
            }
            # Only add paragraph_number if it exists
            if paragraph is not None:
                metadata["paragraph_number"] = paragraph
            
            metadatas.append(metadata)
        
        # Generate embeddings using Gemini
        try:
            response = await client.aio.models.embed_content(
                model=settings.embedding_model,
                contents=texts
            )
            embeddings = [emb.values for emb in response.embeddings]
            
            # Add to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size}: {e}")
            raise
    
    final_count = collection.count()
    logger.info(f"Ingestion complete! Total documents in collection: {final_count}")


async def main():
    """Main entry point for ingestion script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load legal documents into ChromaDB")
    parser.add_argument(
        "--body-file",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "body.jsonl",
        help="Path to body content JSONL file"
    )
    parser.add_argument(
        "--elucidation-file",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "elucidation.jsonl",
        help="Path to elucidation content JSONL file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing"
    )
    
    args = parser.parse_args()
    
    if not args.body_file.exists():
        logger.error(f"Body file not found: {args.body_file}")
        sys.exit(1)
    
    if not args.elucidation_file.exists():
        logger.error(f"Elucidation file not found: {args.elucidation_file}")
        sys.exit(1)
    
    await load_chunks_to_chromadb(
        body_file=args.body_file,
        elucidation_file=args.elucidation_file,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    asyncio.run(main())
