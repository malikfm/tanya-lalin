# Tanya Lalin - Project Specification

**Version**: 1.0.0  
**Date**: 2026-01-07  
**Status**: Final

---

## 1. Overview

### 1.1 Project Description

**Tanya Lalin** (short for "Tanya Lalu Lintas" - Ask Traffic) is a web-based chatbot powered by Retrieval-Augmented Generation (RAG) and Large Language Models (LLM) designed to answer questions about Indonesian traffic regulations. The system enables everyday citizens to ask traffic law questions in casual Indonesian language and receive accurate, legally-grounded answers.

### 1.2 Problem Statement

The previous implementation suffers from poor retrieval performance. When users ask questions using everyday language (e.g., "Apakah menerobos lampu merah melanggar aturan?" - "Is running a red light against the rules?"), the system fails to find relevant document chunks. This is because:

1. **Terminology Gap**: Legal documents use formal terms ("Alat Pemberi Isyarat Lalu Lintas" - Traffic Signal Device) while users use everyday language ("lampu merah" - red light).
2. **Pure Semantic Search Limitations**: The current approach relies solely on vector similarity which struggles with domain-specific terminology mismatches.
3. **Suboptimal Embedding Model**: The current embedding model may not be optimized for Indonesian legal text.

### 1.3 Project Goals

1. **Improve Retrieval Accuracy**: Ensure the system can find relevant legal text chunks even when users use everyday language.
2. **Maintain Factual Accuracy**: All responses must be grounded in the actual legal documents.
3. **User-Friendly Interface**: Simple, intuitive chat interface accessible to general public.
4. **Session-Based Conversation**: Support multi-turn conversations within a session.

---

## 2. Target Users

### 2.1 Primary Audience

- **General Public (Masyarakat Umum)**: Indonesian citizens who want to understand traffic regulations without legal expertise.

### 2.2 User Characteristics

- Use everyday Indonesian language, not formal legal terminology
- May not know specific article numbers or legal references
- Expect quick, clear, and accurate answers
- May ask follow-up questions in the same conversation

### 2.3 Example User Questions

| Everyday Language (User) | Formal Legal Term |
|--------------------------|-------------------|
| "lampu merah" (red light) | "Alat Pemberi Isyarat Lalu Lintas" (Traffic Signal Device) |
| "menerobos lampu merah" (running a red light) | "melanggar aturan perintah atau larangan yang dinyatakan dengan Alat Pemberi Isyarat Lalu Lintas" |
| "bahu jalan" (road shoulder) | "bahu Jalan" (road shoulder - same but context differs) |
| "trotoar" (sidewalk) | "trotoar" / "fasilitas Pejalan Kaki" |
| "parkir sembarangan" (illegal parking) | Various articles about parking regulations |
| "menyalip zigzag" (zigzag overtaking) | "aturan gerakan lalu lintas" (traffic movement rules) |
| "SIM" (driver's license) | "Surat Izin Mengemudi" |
| "STNK" (vehicle registration) | "Surat Tanda Nomor Kendaraan Bermotor" |

---

## 3. Data Sources

### 3.1 Current Data Source

- **UU No. 22 Tahun 2009 tentang Lalu Lintas dan Angkutan Jalan (LLAJ)**
  - Law on Road Traffic and Transportation
  - Contains 326 articles covering all aspects of Indonesian traffic regulations

### 3.2 Document Structure

The legal document contains:
- **Body Text**: Main content of each article and paragraph
- **Elucidation (Penjelasan)**: Official explanations/clarifications for articles

### 3.3 Future Considerations

The system architecture should allow for easy addition of:
- Government Regulations (Peraturan Pemerintah)
- Ministerial Regulations (Peraturan Menteri)
- Police Regulations (Peraturan Kepolisian)

---

## 4. Technical Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                    │
│                    (React/Vite Chat Interface)                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ HTTP/REST API
┌──────────────────────────────▼──────────────────────────────────────────┐
│                              BACKEND                                     │
│                         (FastAPI Server)                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Chat Service                                  │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │   │
│  │  │    Query      │  │   Hybrid      │  │    Response       │   │   │
│  │  │   Rewriter    │──▶  Retrieval   │──▶   Generator       │   │   │
│  │  │    (LLM)      │  │   Service     │  │     (LLM)         │   │   │
│  │  └───────────────┘  └───────┬───────┘  └───────────────────┘   │   │
│  └─────────────────────────────┼─────────────────────────────────────┘   │
│                                │                                          │
│  ┌─────────────────────────────▼─────────────────────────────────────┐   │
│  │                     Retrieval Layer                                │   │
│  │  ┌─────────────────────┐    ┌─────────────────────┐              │   │
│  │  │   Vector Search     │    │   Keyword Search    │              │   │
│  │  │   (Semantic)        │    │   (BM25/Full-text)  │              │   │
│  │  └─────────┬───────────┘    └─────────┬───────────┘              │   │
│  │            │      Reciprocal Rank     │                           │   │
│  │            └──────────Fusion──────────┘                           │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                          DATA LAYER                                      │
│  ┌─────────────────────┐    ┌─────────────────────────────────────┐    │
│  │   Vector Store      │    │      Document Store                  │    │
│  │   (Embeddings)      │    │      (Full Text + Metadata)          │    │
│  └─────────────────────┘    └─────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Key Components

#### 4.2.1 Query Rewriter (NEW)

**Purpose**: Transform everyday language queries into legal terminology-enriched queries.

**Implementation**:
- Use LLM to expand/rewrite user queries
- Map common terms to legal equivalents
- Generate multiple search variants

**Example**:
```
Input:  "Apakah menerobos lampu merah melanggar aturan?"
Output: "menerobos lampu merah, Alat Pemberi Isyarat Lalu Lintas, 
         melanggar aturan perintah larangan isyarat lalu lintas, 
         rambu lalu lintas warna merah"
```

#### 4.2.2 Hybrid Retrieval Service (NEW)

**Purpose**: Combine semantic search with keyword-based search for better recall.

**Components**:
1. **Vector Search**: Semantic similarity using embeddings
2. **Keyword Search**: BM25 or full-text search for exact term matching
3. **Fusion**: Reciprocal Rank Fusion (RRF) to combine results

#### 4.2.3 Response Generator

**Purpose**: Generate natural language responses based on retrieved legal chunks.

**Requirements**:
- Must cite specific articles and paragraphs
- Must use formal Indonesian language in response
- Must clearly state when information is not found

### 4.3 Technology Stack

#### 4.3.1 Backend

| Component | Technology | Notes |
|-----------|------------|-------|
| Web Framework | FastAPI | Async support, OpenAPI docs |
| LLM Provider | Google Gemini | gemini-2.5-flash-lite for responses |
| Embedding Model | Google Gemini | gemini-embedding-001 (3072 dimensions) |
| Vector Store | ChromaDB | Local persistent storage, no setup required |
| Session Store | In-memory dict | Simple TTL-based cleanup |

#### 4.3.2 Frontend

| Component | Technology | Notes |
|-----------|------------|-------|
| Framework | React + Vite | SPA with TypeScript |
| Styling | Tailwind CSS | With shadcn/ui components |
| State Management | React hooks | Simple state management |

### 4.4 Data Storage

#### 4.4.1 ChromaDB Collection

Instead of PostgreSQL, the implementation uses ChromaDB with the following structure:

```python
# Collection: legal_chunks
# Metadata per document:
{
    "source": "UU_22_2009_LLAJ",
    "article_number": 287,
    "paragraph_number": 2,  # Optional
    "chunk_type": "body"    # or "elucidation"
}

# Embedding: 3072-dimensional vector from gemini-embedding-001
# Document: Full text content of the article/paragraph
```

#### 4.4.2 Session Storage (In-Memory)

```python
# Session structure
{
    "id": "uuid",
    "messages": [
        {"role": "user", "content": "...", "timestamp": "..."},
        {"role": "assistant", "content": "...", "retrieved_chunks": [...]}
    ],
    "created_at": "...",
    "last_activity": "..."
}
# Sessions expire after SESSION_TTL_HOURS (default: 24)
```

---

## 5. API Specification

### 5.1 Chat Endpoints

#### POST /api/v1/chat

Start or continue a chat conversation.

**Request**:
```json
{
    "session_id": "uuid-optional",  // null for new session
    "message": "Apakah menerobos lampu merah melanggar aturan?"
}
```

**Response**:
```json
{
    "session_id": "abc123-...",
    "message_id": 1,
    "query": "Apakah menerobos lampu merah melanggar aturan?",
    "response": "Ya, menerobos lampu merah merupakan pelanggaran. Berdasarkan Pasal 287 ayat (2) UU 22/2009...",
    "retrieved_chunks": [
        {
            "source": "UU_22_2009_LLAJ",
            "article_number": 287,
            "paragraph_number": 2,
            "chunk_type": "body",
            "text": "Setiap orang yang mengemudikan Kendaraan Bermotor di Jalan yang melanggar aturan perintah atau larangan yang dinyatakan dengan Alat Pemberi Isyarat Lalu Lintas...",
            "similarity_score": 0.85
        }
    ],
    "timestamp": "2026-01-07T07:00:00Z"
}
```

#### GET /api/v1/chat/{session_id}/history

Get conversation history for a session.

**Response**:
```json
{
    "session_id": "abc123-...",
    "messages": [
        {
            "id": 1,
            "role": "user",
            "content": "Apakah menerobos lampu merah melanggar aturan?",
            "timestamp": "2026-01-07T07:00:00Z"
        },
        {
            "id": 2,
            "role": "assistant",
            "content": "Ya, menerobos lampu merah merupakan pelanggaran...",
            "timestamp": "2026-01-07T07:00:01Z"
        }
    ]
}
```

#### DELETE /api/v1/chat/{session_id}

Delete a chat session and its history.

### 5.2 Health Endpoints

#### GET /api/v1/health

Check system health status.

---

## 6. Frontend Specification

### 6.1 Design Requirements

- **Minimal, Clean Interface**: Similar to Claude, ChatGPT, or Gemini
- **Single Page Application**: No complex navigation
- **Responsive Design**: Works on mobile and desktop

### 6.2 UI Components

#### 6.2.1 Chat Container
- Full-height scrollable message area
- Messages displayed as bubbles (user right, assistant left)
- Auto-scroll to latest message

#### 6.2.2 Message Input
- Text input field at bottom
- Send button (disabled when empty or processing)
- Support Enter key to send (Shift+Enter for newline)
- Loading indicator during processing

#### 6.2.3 Message Bubbles
- **User Message**: Right-aligned, colored background
- **Assistant Message**: Left-aligned, neutral background
- Support for formatted text (markdown-like)
- Collapsible "Sources" section for retrieved chunks

### 6.3 Session Management

- Generate session ID on first message (store in localStorage)
- Load conversation history on page refresh
- Optional: "New Chat" button to start fresh session

---

## 7. RAG Pipeline Specification

### 7.1 Ingestion Pipeline

```
PDF Document
     │
     ▼
┌─────────────┐
│   Parser    │ ──▶ Extract articles, paragraphs, structure
└─────────────┘
     │
     ▼
┌─────────────┐
│  Chunker    │ ──▶ Split into semantic chunks (300 tokens, 30 overlap)
└─────────────┘
     │
     ▼
┌─────────────┐
│  Embedder   │ ──▶ Generate embeddings for each chunk
└─────────────┘
     │
     ▼
┌─────────────┐
│   Loader    │ ──▶ Store in vector database
└─────────────┘
```

### 7.2 Query Pipeline

```
User Query
     │
     ▼
┌─────────────────────┐
│   Query Rewriter    │ ──▶ 1. Static term mappings
│   (Multi-layer)     │     2. Special pattern matching
│                     │     3. LLM-based expansion
└─────────────────────┘
     │
     ├─ Legal Search Query ─────────► Vector Search (weight: 2.0)
     │
     ├─ Original Query ─────────────► Vector Search (weight: 1.0)
     │
     └─ Additional Key Phrases ─────► Vector Search (weight: 0.8)
                     │
                     ▼
          ┌─────────────────────┐
          │  Reciprocal Rank    │ ──▶ Weighted combination
          │      Fusion         │
          └─────────────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Response Generator  │ ──▶ Generate answer with citations
          │       (LLM)         │
          └─────────────────────┘
                     │
                     ▼
               Final Response
```

**Note**: The implementation uses multi-query vector search instead of hybrid (vector + keyword) search. This approach compensates for the lack of BM25 in ChromaDB by running multiple vector searches with different query variants.

### 7.3 Query Rewriting Strategy

The Query Rewriter uses a multi-layer approach:

1. **Static Term Mappings**: Dictionary of 25+ everyday-to-legal term mappings
   ```python
   "lampu merah" → ["Alat Pemberi Isyarat Lalu Lintas", "APILL"]
   "menerobos" → ["melanggar aturan perintah atau larangan"]
   ```

2. **Special Patterns**: Pre-defined legal search queries for common questions
   ```python
   "menerobos lampu merah" → "Setiap orang yang mengemudikan Kendaraan 
   Bermotor di Jalan yang melanggar aturan perintah atau larangan yang 
   dinyatakan dengan Alat Pemberi Isyarat Lalu Lintas dipidana"
   ```

3. **LLM Rewriting**: Dynamic expansion for novel questions
   - Output: `legal_search_query` and `key_legal_phrases`

### 7.4 Multi-Query Search Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Vector Search Top-K | 10 | Per-query retrieval count |
| Legal Query Weight | 2.0 | Highest priority in RRF |
| Original Query Weight | 1.0 | Standard weight |
| Additional Query Weight | 0.8 | Lower priority |
| RRF K Parameter | 60 | Standard RRF constant |
| Final Top-K | 5 | Chunks passed to LLM |
| Minimum Similarity | 0.1 | Lower threshold for RRF scores |

### 7.5 Response Generation Prompt

```
Anda adalah asisten ahli hukum lalu lintas Indonesia. Jawablah pertanyaan pengguna 
berdasarkan kutipan dokumen hukum yang diberikan.

ATURAN:
1. Jawab langsung dan ringkas tanpa frasa pembuka seperti "Berdasarkan dokumen...".
2. Gunakan bahasa Indonesia formal.
3. Kutip nomor pasal dan ayat yang relevan menggunakan tanda kutip.
4. Jika informasi tidak ditemukan, nyatakan dengan jelas.
5. Jangan mengarang informasi yang tidak ada dalam konteks.

KONTEKS PERCAKAPAN:
{conversation_history}

KUTIPAN DOKUMEN HUKUM:
{retrieved_chunks}

PERTANYAAN PENGGUNA:
{user_query}

JAWABAN:
```

---

## 8. Evaluation Criteria

### 8.1 Retrieval Quality

| Metric | Target | Description |
|--------|--------|-------------|
| Recall@5 | > 80% | Relevant chunk in top 5 results |
| Precision@5 | > 60% | Percentage of relevant chunks in top 5 |

### 8.2 Response Quality

| Criteria | Description |
|----------|-------------|
| Factual Accuracy | Response matches actual legal text |
| Citation Accuracy | Correct article/paragraph references |
| Completeness | All relevant aspects addressed |
| Clarity | Easy to understand by general public |

### 8.3 Test Cases

| # | User Query | Expected Relevant Articles |
|---|------------|---------------------------|
| 1 | Apakah menerobos lampu merah melanggar aturan? | Art. 106(4)c, Art. 287(2) |
| 2 | Apakah berhenti sebentar pada tempat yang dilarang parkir melanggar aturan? | Art. 106(4)e, Art. 287(3) |
| 3 | Pada situasi macet, apakah boleh menggunakan bahu jalan? | Art. 108, Art. 287 |
| 4 | Bagaimana aturan tentang motor yang menaiki trotoar? | Art. 45, Art. 106, Art. 284 |
| 5 | Apakah trotoar boleh dijadikan tempat parkir? | Art. 43, Art. 45 |
| 6 | Apakah menyalip secara zigzag melanggar aturan? | Art. 106(4)d, Art. 109-112 |
| 7 | Apakah pinggir jalan boleh dijadikan tempat parkir? | Art. 43, Art. 287 |

---

## 9. Implementation Phases

### Phase 1: Core RAG Improvements (Priority: HIGH)

1. **Implement Query Rewriter**
   - LLM-based query expansion
   - Term mapping dictionary
   
2. **Implement Hybrid Search**
   - Add keyword/BM25 search capability
   - Implement Reciprocal Rank Fusion
   
3. **Upgrade Embedding Model**
   - Evaluate and select Indonesian-optimized model
   - Re-embed all documents

### Phase 2: Session Management (Priority: MEDIUM)

1. **Backend Session Support**
   - Add session and message tables
   - Implement session CRUD API
   
2. **Conversation Context**
   - Include conversation history in LLM prompt
   - Implement context window management

### Phase 3: Frontend Enhancement (Priority: MEDIUM)

1. **Chat UI Update**
   - Implement clean chat interface
   - Add session management
   
2. **UX Improvements**
   - Loading states
   - Error handling
   - Source citation display

### Phase 4: Evaluation & Tuning (Priority: HIGH)

1. **Create Evaluation Dataset**
   - Build question-answer pairs
   - Include edge cases
   
2. **Iterative Improvement**
   - Test with evaluation dataset
   - Tune parameters

---

## 10. Configuration

### 10.1 Environment Variables

```bash
# Application
LOG_LEVEL=INFO

# Google Gemini API
GEMINI_API_KEY=your-gemini-api-key-here

# LLM Configuration
LLM_MODEL=gemini-2.5-flash-lite
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIM=3072

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=legal_chunks

# RAG Configuration
VECTOR_SEARCH_TOP_K=10
FINAL_TOP_K=5
MIN_SIMILARITY=0.3

# Session Configuration
SESSION_TTL_HOURS=24
MAX_CONTEXT_MESSAGES=10

# CORS Configuration
CORS_ORIGINS=*  # or comma-separated list
```

---

## 11. Constraints and Considerations

### 11.1 Budget Constraints

- This is a portfolio/demo project with limited budget
- Prefer open-source or free-tier solutions where possible
- Consider using local models (Ollama) for development

### 11.2 Technical Constraints

- Must handle Indonesian language effectively
- Must work with legal document structure
- Response time should be reasonable for interactive use

### 11.3 Out of Scope

- User authentication/login
- Persistent user profiles
- Multiple language support
- Real-time collaboration
- Mobile native apps

---

## 12. Appendix

### A. Legal Term Mapping Dictionary

| Everyday Term | Legal Term | Related Articles |
|---------------|------------|------------------|
| lampu merah | Alat Pemberi Isyarat Lalu Lintas | 25, 95, 103, 106, 287 |
| SIM | Surat Izin Mengemudi | 77-88, 281, 288 |
| STNK | Surat Tanda Nomor Kendaraan Bermotor | 64-70, 288 |
| helm | helm standar nasional Indonesia | 57, 106, 291 |
| sabuk pengaman | sabuk keselamatan | 106, 289 |
| parkir | Parkir | 43-44, 106, 287 |
| bahu jalan | bahu Jalan | 43, 108 |
| trotoar | trotoar, fasilitas Pejalan Kaki | 45, 131 |
| menyalip | melewati Kendaraan | 109-112, 287 |
| ngebut | batas kecepatan | 106, 115, 287 |
| kaca spion | kaca spion | 48, 106, 285 |
| plat nomor | Tanda Nomor Kendaraan Bermotor | 68, 280 |

### B. Article Categories

| Category | Article Range | Description |
|----------|---------------|-------------|
| Road Infrastructure | 1-46 | Roads, terminals, parking facilities |
| Vehicles | 47-76 | Vehicle requirements, registration |
| Drivers | 77-94 | Driver licensing, requirements |
| Traffic Rules | 95-123 | Traffic management, regulations |
| Traffic Safety | 124-147 | Safety requirements |
| Transportation | 148-186 | Public transportation |
| Impact & Environment | 187-219 | Environmental impact |
| Development | 220-243 | Vehicle development |
| Enforcement | 244-273 | Enforcement, authority |
| Penalties | 274-317 | Criminal and civil penalties |
| Transitional | 318-326 | Transitional provisions |

---

*Document prepared for Tanya Lalin project refactoring initiative.*
