#!/bin/bash
set -e

SEED_DIR="/app/data/seed"

# Default host (:: for IPv6 dual-stack, 0.0.0.0 for IPv4 only)
HOST=${HOST:-0.0.0.0}

# Check if ChromaDB has been initialized
if [ ! -f "/app/data/chroma_db/.initialized" ]; then
    if [ -d "$SEED_DIR" ] && [ "$(ls -A $SEED_DIR)" ]; then
        echo "Initializing ChromaDB with document data..."
        python -m scripts.ingest_to_chromadb --body-file $SEED_DIR/body.jsonl --elucidation-file $SEED_DIR/elucidation.jsonl
        touch /app/data/chroma_db/.initialized
        echo "ChromaDB initialization complete."
    else
        echo "No seed data found. Skipping ingestion."
    fi
else
    echo "ChromaDB already initialized, skipping ingestion."
fi

# Start the application
exec uvicorn app.main:app --host $HOST --port $PORT
