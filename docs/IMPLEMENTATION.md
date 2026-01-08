# Implementation Documentation

**Version**: 1.0.0  
**Last Updated**: 2026-01-08

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Decisions](#2-architecture-decisions)
3. [Implementation Details](#3-implementation-details)
4. [Challenges Faced](#4-challenges-faced)
5. [Current Limitations](#5-current-limitations)
6. [Future Improvements](#6-future-improvements)

---

## 1. Overview

Tanya Lalin is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about Indonesian traffic law (UU No. 22 Tahun 2009). The system transforms everyday Indonesian language questions into formal legal terminology, retrieves relevant legal text chunks, and generates natural language responses with proper article citations.

### Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM & Embeddings | Google Gemini API | Query rewriting, response generation, vector embeddings |
| Vector Store | ChromaDB | Semantic similarity search |
| Backend | FastAPI | REST API, RAG pipeline orchestration |
| Frontend | React + Vite | Chat interface |
| Deployment | Docker Compose | Containerized deployment |

---

## 2. Architecture Decisions

### 2.1 Why Google Gemini API?

**Decision**: Use Google Gemini API for both LLM (gemini-2.5-flash-lite) and embeddings (gemini-embedding-001).

**Rationale**:
- **Free Tier**: Gemini offers a generous free tier suitable for portfolio/demo projects
- **Unified Provider**: Using one provider simplifies API key management and reduces latency
- **Indonesian Language Support**: Gemini performs well with Indonesian text
- **Cost Efficiency**: gemini-2.5-flash-lite is optimized for speed and cost

**Trade-offs**:
- Dependency on external API (rate limits, availability)
- Less control compared to self-hosted models (Ollama)

### 2.2 Why ChromaDB Instead of PostgreSQL+pgvector?

**Decision**: Use ChromaDB as the vector store instead of PostgreSQL with pgvector.

**Rationale**:
- **Simplicity**: ChromaDB requires no database server setup
- **Zero Configuration**: Works out of the box with persistent storage
- **Sufficient Scale**: 1,214 documents is well within ChromaDB's optimal range
- **Docker Friendly**: Easy to persist data via Docker volumes

**Trade-offs**:
- No built-in keyword/BM25 search (specification mentioned hybrid search)
- Less suitable for very large datasets (millions of documents)
- No SQL query capabilities

### 2.3 Why Query Rewriting Over Hybrid Search?

**Specification vs Implementation**:
The original specification called for hybrid search (vector + keyword/BM25). However, the implementation uses a **multi-query vector search** approach instead.

**Decision**: Replace keyword search with LLM-based query rewriting and multi-query vector search.

**Rationale**:
- **Terminology Gap Solution**: The core problem is users using "lampu merah" while legal text says "Alat Pemberi Isyarat Lalu Lintas". Query rewriting directly addresses this.
- **No BM25 in ChromaDB**: ChromaDB doesn't have built-in keyword search, so implementing BM25 would require additional infrastructure.
- **Multi-Query Compensates**: Running vector search on multiple query variants (original, legal, additional phrases) provides similar coverage to hybrid search.
- **Weighted RRF**: Reciprocal Rank Fusion with weights prioritizes legal query results.

**Implementation**:
```
User Query
    │
    ▼
Query Rewriter (LLM) ──► Legal Search Query
    │                      │
    ├─ Original Query ─────┼─ Vector Search (weight: 1.0)
    │                      │
    └─ Additional Queries ─┼─ Vector Search (weight: 0.8)
                           │
                           ▼
                    Legal Query ─► Vector Search (weight: 2.0)
                           │
                           ▼
                    Reciprocal Rank Fusion
                           │
                           ▼
                    Top K Results
```

### 2.4 Why In-Memory Session Store?

**Decision**: Use in-memory dictionary for session storage instead of Redis or database.

**Rationale**:
- **Simplicity**: No additional infrastructure needed
- **Portfolio Project**: Sessions don't need to survive restarts for a demo
- **TTL Management**: Simple cleanup with expiration timestamps

**Trade-offs**:
- Sessions lost on container restart
- Not suitable for horizontal scaling (no shared state)

### 2.5 Why Article-Level Chunking?

**Decision**: Chunk documents at article/paragraph level rather than fixed token windows.

**Rationale**:
- **Semantic Coherence**: Each chunk is a complete legal article or paragraph
- **Clean Citations**: Easy to cite specific articles ("Pasal 287 ayat (2)")
- **No Context Loss**: No risk of splitting critical information across chunks

**Trade-offs**:
- Variable chunk sizes (some articles are very long)
- Potential for very long chunks exceeding context windows

---

## 3. Implementation Details

### 3.1 Document Parsing Pipeline

The PDF parser (`scripts/run_parser.py`) extracts legal text with structure awareness:

```
PDF Document
    │
    ▼
Page Extraction (PyMuPDF)
    │
    ▼
Header Skip (configurable lines per page)
    │
    ▼
Pattern Matching (regex-based)
    ├─ Article Pattern: "Pasal \d+"
    ├─ Paragraph Pattern: "\(\d+\)"
    ├─ Chapter/Section Patterns
    │
    ▼
Body vs Elucidation Separation
    │
    ▼
Validation (expected article count)
    │
    ▼
JSONL Output
    ├─ body.jsonl
    └─ elucidation.jsonl
```

**Key Design Choices**:
- Separate body and elucidation for better retrieval (users can get official explanations)
- Regex patterns specific to Indonesian legal document format
- Validation ensures all 326 articles are parsed correctly

### 3.2 Query Rewriting Strategy

The query rewriter uses a multi-layer approach:

1. **Static Term Mappings**: Dictionary of everyday-to-legal term mappings
   ```python
   "lampu merah" → ["Alat Pemberi Isyarat Lalu Lintas", "APILL"]
   "menerobos" → ["melanggar aturan perintah atau larangan"]
   ```

2. **Special Patterns**: Pre-defined legal search queries for common questions
   ```python
   "menerobos lampu merah" → "Setiap orang yang mengemudikan 
   Kendaraan Bermotor di Jalan yang melanggar aturan perintah 
   atau larangan yang dinyatakan dengan Alat Pemberi Isyarat 
   Lalu Lintas dipidana"
   ```

3. **LLM Rewriting**: Dynamic query expansion for novel questions
   - Input: User query
   - Output: JSON with legal_search_query and key_legal_phrases

**Why This Layered Approach?**
- Static mappings are fast and reliable for known terms
- Special patterns guarantee good retrieval for common questions
- LLM handles novel/complex queries not covered by static rules

### 3.3 Retrieval Service

The retrieval service performs multi-query vector search with weighted RRF:

```python
# Weight configuration
weights = {
    "legal_query": 2.0,      # Highest priority
    "original_query": 1.0,   # Standard weight
    "additional": 0.8,       # Lower weight
}
```

**Process**:
1. Generate embedding for legal search query
2. Generate embedding for original user query
3. Generate embeddings for additional key phrases (up to 3)
4. Run vector search for each query
5. Combine results using weighted Reciprocal Rank Fusion
6. Filter by minimum similarity threshold
7. Return top K results

### 3.4 Response Generation

The LLM service generates responses with strict guardrails:

- **System Instruction**: Defines response format, citation requirements, and behavior for missing information
- **Context Formatting**: Retrieved chunks are formatted with article references
- **Conversation History**: Last 3 turns included for context continuity
- **Temperature**: 0.3 for consistent, factual responses

**Error Handling**:
- No relevant chunks → `ResponseMessages.NO_RELEVANT_CHUNKS`
- API quota exceeded → `ResponseMessages.ERROR`
- Other exceptions → `ResponseMessages.ERROR`

### 3.5 Docker Deployment

The containerized deployment uses:

- **Backend**: Python 3.12-slim with uv package manager
- **Frontend**: nginx:alpine serving static React build
- **Data Flow**: Seed data mounted from host, ingested on first start
- **ChromaDB**: Persisted via Docker volume

**Entrypoint Logic**:
```bash
if [ ! -f "/app/data/chroma_db/.initialized" ]; then
    # First start: ingest seed data
    python -m scripts.ingest_to_chromadb ...
    touch /app/data/chroma_db/.initialized
fi
# Start uvicorn
exec uvicorn app.main:app --host $HOST --port $PORT
```

---

## 4. Challenges Faced

### 4.1 Terminology Gap (The Core Problem)

**Challenge**: Users ask "Apakah menerobos lampu merah melanggar aturan?" but the legal text says "melanggar aturan perintah atau larangan yang dinyatakan dengan Alat Pemberi Isyarat Lalu Lintas".

**Initial Approach**: Pure semantic search failed because embeddings of "lampu merah" and "Alat Pemberi Isyarat Lalu Lintas" are not similar enough.

**Solution**: Multi-layer query rewriting that explicitly translates everyday terms to legal terminology before embedding search.

**Lesson Learned**: Domain-specific vocabulary requires explicit handling; semantic similarity alone is insufficient for specialized terminology gaps.

### 4.2 API Rate Limiting

**Challenge**: Google Gemini free tier has rate limits (e.g., 20 requests/day for model). During testing, quota exceeded errors caused silent failures.

**Solution**: 
- Created `APIQuotaExceededError` custom exception
- Early detection of 429/RESOURCE_EXHAUSTED errors in `gemini_client.py`
- Propagate error to chat service instead of silent fallback
- Return user-friendly error message via `ResponseMessages.ERROR`

**Lesson Learned**: Always explicitly handle rate limit errors; silent failures are worse than visible errors.

### 4.3 Embedding Dimension Mismatch

**Challenge**: Changed embedding model from `text-embedding-004` (768 dim) to `gemini-embedding-001` (3072 dim), causing "Collection expecting dimension 768, got 3072" error.

**Solution**: 
- Update `EMBEDDING_DIM` config to 3072
- Delete and recreate ChromaDB collection
- Re-ingest all documents with new embeddings

**Lesson Learned**: Embedding model changes require full re-ingestion; maintain dimension configuration in sync with model choice.

### 4.4 Docker Data Ingestion

**Challenge**: Backend container started with empty ChromaDB because data files weren't available inside container.

**Solution**:
- Mount `./backend/data` to `/app/data/seed` in docker-compose
- Entrypoint script checks for seed files and ingests on first start
- Data persistence via Docker volume for subsequent restarts

**Lesson Learned**: Consider data bootstrap in containerized deployments; seed data needs explicit handling.

### 4.5 Nginx Template Substitution

**Challenge**: Nginx config needed dynamic `$BACKEND_URL` for different deployment environments. Initial approach of in-place substitution with envsubst failed (writing to file while reading).

**Solution**: Leverage nginx:alpine's built-in template processing from `/etc/nginx/templates/` directory.

**Lesson Learned**: Check existing container capabilities before implementing custom solutions.

---

## 5. Current Limitations

### 5.1 No True Hybrid Search

The implementation uses multi-query vector search instead of true hybrid search (vector + keyword/BM25). This means:

- Exact keyword matches may be missed if embeddings don't capture them
- No fallback for embedding model failures/limitations
- Relies heavily on query rewriting quality

### 5.2 Single Document Source

Currently only supports UU No. 22 Tahun 2009. Adding new documents requires:

- Custom regex patterns for different document formats
- Re-running parser and ingestion
- No dynamic document upload capability

### 5.3 In-Memory Sessions

Sessions are lost on container restart. Not suitable for production deployments requiring session persistence.

### 5.4 No User Authentication

Anyone can access the system. No user tracking, rate limiting per user, or personalization.

### 5.5 Rate Limit Handling

API quota exceeded results in error message to user, not graceful queuing or retry. Users must wait and retry manually.

### 5.6 No Conversation Memory Beyond Session

Each session is independent. No long-term user memory or learning from previous interactions.

### 5.7 Static Term Mappings

Term mappings are hardcoded. Adding new everyday-to-legal term mappings requires code changes and redeployment.

### 5.8 No Re-ranking

Retrieved chunks are not re-ranked with a cross-encoder model. RRF provides basic fusion but may miss optimal ordering.

---

## 6. Future Improvements

### 6.1 Short-term Improvements

#### 6.1.1 Add True Keyword Search
Implement BM25 search using SQLite FTS5 or Elasticsearch for hybrid retrieval that combines with vector search.

#### 6.1.2 Cross-encoder Re-ranking
Add a cross-encoder model (e.g., multilingual-MiniLM) to re-rank top candidates for better precision.

#### 6.1.3 Redis Session Store
Replace in-memory sessions with Redis for persistence and horizontal scaling support.

#### 6.1.4 Caching Layer
Cache frequent queries and their results to reduce API calls and improve response time.

#### 6.1.5 Rate Limiting
Implement per-IP or per-session rate limiting to prevent abuse.

### 6.2 Medium-term Improvements

#### 6.2.1 Multi-document Support
Extend parser to handle multiple legal document types:
- Peraturan Pemerintah (Government Regulations)
- Peraturan Menteri (Ministerial Regulations)
- Peraturan Kepolisian (Police Regulations)

#### 6.2.2 Dynamic Term Mapping
Store term mappings in database, allow admin to add new mappings without code changes.

#### 6.2.3 Feedback Loop
Allow users to rate responses (helpful/not helpful) to identify gaps and improve retrieval.

#### 6.2.4 Query Analytics
Track common queries to:
- Identify frequently asked topics
- Discover missing term mappings
- Prioritize content improvements

#### 6.2.5 Local Model Option
Support Ollama for fully offline deployment using local LLM models.

### 6.3 Long-term Improvements

#### 6.3.1 Fine-tuned Embedding Model
Fine-tune embedding model on Indonesian legal corpus for better domain-specific semantic understanding.

#### 6.3.2 Legal Knowledge Graph
Build knowledge graph of legal concepts, articles, and relationships for structured reasoning.

#### 6.3.3 Multi-language Support
Extend to support English questions about Indonesian traffic law for international users.

#### 6.3.4 Mobile Application
Native mobile apps for iOS and Android with offline capability.

#### 6.3.5 Voice Interface
Add speech-to-text and text-to-speech for accessibility.

---

## Appendix: Key Files Reference

| File | Purpose |
|------|---------|
| `backend/app/services/query_rewriter.py` | Query rewriting with term mappings and LLM |
| `backend/app/services/retrieval.py` | Multi-query vector search with RRF |
| `backend/app/services/llm.py` | Response generation with guardrails |
| `backend/app/services/chat.py` | RAG pipeline orchestration |
| `backend/scripts/run_parser.py` | PDF parsing CLI |
| `backend/scripts/ingest_to_chromadb.py` | Vector store ingestion |
| `backend/app/constants.py` | Centralized response messages |
| `compose.yml` | Docker deployment configuration |

---

*This documentation reflects the implementation state as of January 2026.*
