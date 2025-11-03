"""Main module for upserting chunked legal text data into PostgreSQL database.

Handles the final step of the ingestion pipeline by reading
chunked legal text data from a JSONL file and upserting it into a
PostgreSQL database with pgvector extension for similarity search.
The upsert operation ensures that existing records are updated while
new records are inserted, preventing duplicates.

The database upsert process:
1. Reads chunked legal text data from JSONL file.
2. Connects to PostgreSQL database.
3. Transforms data into database records with proper typing.
4. Generates unique document keys for each record.
5. Performs upsert operations using PostgreSQL's ON CONFLICT feature.
6. Commits transactions and handles errors appropriately.
"""
import argparse
import json
from pathlib import Path
from hashlib import sha256
from typing import List

from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker

from app.db.models import LegalTextChunk
from config import settings
from enums import LegalTextType
from ingestions.models import LegalTextChunkDTO
from logging_setup import setup_logger

logger = setup_logger()


def upsert_chunks_to_db(input_file_path: str) -> None:
    """Upsert chunks from JSONL file to PostgreSQL database with pgvector support.
    
    This function reads chunked legal text data from a JSONL file and upserts
    the records into a PostgreSQL database. It uses PostgreSQL's ON CONFLICT
    feature to handle duplicates by updating existing records. Each record
    is assigned a unique document key based on its content for reliable
    identification.
    
    Args:
        input_file_path (str): Path to the JSONL file containing chunked legal text data.
            Each line should contain a serialized LegalTextChunkDTO object.
        
    Raises:
        ValidationError: If any line in the file contains invalid JSON.
        Exception: If any database operation fails (with automatic rollback).
        
    Example:
        >>> upsert_chunks_to_db("data/chunks_with_embeddings.jsonl")
        Reading chunks from data/chunks_with_embeddings.jsonl...
        Loaded 1250 chunks from file
        Starting upsert operation...
        Processed 100/1250 chunks...
        Processed 200/1250 chunks...
        Successfully upserted 1250 chunks to database
    """
    # Create database engine with connection pooling
    DATABASE_URL = (
        f"postgresql+psycopg://{settings.db_user}:{settings.db_password}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )
    
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    
    # Read chunks from JSONL file
    chunks: List[LegalTextChunkDTO] = []
    logger.info(f"Reading chunks from {input_file_path}...")
    
    with open(input_file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                legal_text_chunk = LegalTextChunkDTO.model_validate_json(line)
                chunks.append(legal_text_chunk)
            except ValidationError as e:
                logger.error(f"Error parsing line {line_num}: {e}")
                raise

    logger.info(f"Loaded {len(chunks)} chunks from file")
    
    if not chunks:
        logger.warning("No chunks found in the input file")
        return
    
    # Process chunks and upsert to database
    logger.info("Starting upsert operation...")
    with SessionLocal() as session:
        try:
            for i, chunk in enumerate(chunks):
                # Generate unique document key based on chunk identity
                key = f"{chunk.source}|{chunk.article_number}|{chunk.paragraph_number or ''}|{chunk.chunk_type}|{chunk.chunk_index}"
                doc_key = sha256(key.encode("utf-8")).hexdigest()
                
                # Prepare the data for upsert with proper enum conversion
                chunk_record = {
                    "source": chunk.source,
                    "article_number": chunk.article_number,
                    "paragraph_number": chunk.paragraph_number,
                    "chunk_type": LegalTextType(chunk.chunk_type),
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "embedding": chunk.embedding,
                    "doc_key": doc_key
                }
                
                # Use PostgreSQL ON CONFLICT to upsert
                stmt = insert(LegalTextChunk).values(**chunk_record)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[
                        "source", 
                        "article_number", 
                        "paragraph_number", 
                        "chunk_type", 
                        "chunk_index"
                    ],
                    set_=dict(
                        source=stmt.excluded.source,
                        article_number=stmt.excluded.article_number,
                        paragraph_number=stmt.excluded.paragraph_number,
                        chunk_type=stmt.excluded.chunk_type,
                        chunk_index=stmt.excluded.chunk_index,
                        text=stmt.excluded.text,
                        embedding=stmt.excluded.embedding,
                        doc_key=doc_key,
                        updated_at=text("now()")
                    )
                )
                
                session.execute(stmt)
                
                # Log progress every 100 records
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(chunks)} chunks...")
            
            session.commit()
            logger.info(f"Successfully upserted {len(chunks)} chunks to database")
        
        except Exception as e:
            session.rollback()
            logger.error(f"Error during upsert operation, rollback transaction.")
            raise
    
    logger.info("Upsert operation completed")


def main():
    args_parser = argparse.ArgumentParser(
        description="Upsert chunked legal text data to PostgreSQL database with pgvector support"
    )
    args_parser.add_argument(
        "--input-file", 
        required=True, 
        help="Path to the input JSONL file containing chunks with embeddings"
    )
    args = args_parser.parse_args()
    
    input_file_path = Path(args.input_file)
    if not input_file_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file_path}")
    
    upsert_chunks_to_db(str(input_file_path))


if __name__ == "__main__":
    raise SystemExit(main())
