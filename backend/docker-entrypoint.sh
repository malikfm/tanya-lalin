#!/bin/bash
set -e

# Check if ChromaDB has been initialized
if [ ! -f "/app/data/chroma_db/.initialized" ]; then
    echo "Initializing ChromaDB with document data..."
    python scripts/ingest_to_chromadb.py
    touch /app/data/chroma_db/.initialized
    echo "ChromaDB initialization complete."
else
    echo "ChromaDB already initialized, skipping ingestion."
fi

# Start the application
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
