# Tanya Lalin

**Tanya Lalin** (short for "Tanya Lalu Lintas", meaning "Ask Traffic") is a RAG-based chatbot that answers questions about Indonesian traffic regulations based on Law No. 22 of 2009 on Road Traffic and Transportation.

## Features

- **Hybrid Search**: Combines vector search (semantic) and keyword search for better retrieval accuracy
- **Query Rewriting**: Automatically transforms everyday language into formal legal terminology
- **Multi-turn Chat**: Supports continuous conversations within a session
- **Source Citations**: Displays the relevant articles and paragraphs used as the basis for answers
- **Fast and Free**: Uses Google Gemini API (free tier) and ChromaDB (local vector store)

## Tech Stack

### Backend
- **FastAPI** for the web framework
- **Google Gemini** for LLM and embeddings
- **ChromaDB** for the vector database
- **Python 3.10+**

### Frontend
- **React + Vite** for the UI framework
- **Tailwind CSS** for styling
- **shadcn/ui** for UI components

## Quick Start

### 1. Get Gemini API Key

1. Visit https://aistudio.google.com/apikey
2. Create an API key
3. Copy the API key

### 2. Setup Backend

```bash
# Copy environment file (from project root)
cp .env.example .env

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your-api-key-here

cd backend

# Install dependencies (using uv)
uv sync
```

### 3. Parse PDF and Ingest Data

If you need to parse a new PDF document:

```bash
# Parse PDF to JSONL (from backend directory)
uv run python -m scripts.run_parser \
  --input-pdf-path "../UU 22 Tahun 2009.pdf" \
  --output-dir ./data \
  --body-start 2 \
  --body-end 143 \
  --elucidation-start 150 \
  --header-lines-to-skip 3 \
  --expected-total-articles 326
```

It will create JSONL files in `data/`. If JSONL files already exist in `data/`, you can skip parsing and directly ingest:

```bash
# Ingest JSONL data to ChromaDB
uv run python scripts/ingest_to_chromadb.py
```

### 4. Run Backend Server

```bash
uv run uvicorn app.main:app --reload
```

### 5. Setup Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run development server (with backend URL for local dev)
VITE_API_URL=http://localhost:8000 npm run dev
```

### 6. Open App

Open http://localhost:5173 in your browser.

## Example Questions

- "Apakah menerobos lampu merah melanggar aturan?" (Is running a red light against the rules?)
- "Apa sanksi jika tidak memakai helm?" (What is the penalty for not wearing a helmet?)
- "Apakah bahu jalan boleh dilalui saat macet?" (Can you use the road shoulder during traffic jams?)
- "Bagaimana aturan tentang motor yang menaiki trotoar?" (What are the rules about motorcycles on sidewalks?)
- "Apakah menyalip secara zigzag melanggar aturan?" (Is zigzag overtaking against the rules?)

## Project Structure

```
tanya-lalin/
├── backend/
│   ├── app/
│   │   ├── api/           # API endpoints
│   │   ├── core/          # Core services (Gemini, ChromaDB, Session)
│   │   ├── services/      # Business logic (RAG pipeline)
│   │   └── main.py        # FastAPI app
│   ├── scripts/
│   │   ├── run_parser.py          # CLI: Parse PDF to JSONL
│   │   ├── ingest_to_chromadb.py  # CLI: Ingest JSONL to ChromaDB
│   │   ├── pdf_parser.py          # PDF parsing logic
│   │   ├── pdf_patterns.py        # Regex patterns for parsing
│   │   ├── parser_validation.py   # Parsing validation
│   │   ├── models.py              # Data models
│   │   └── enums.py               # Enum definitions
│   ├── data/              # Parsed JSONL files
│   ├── config.py          # Configuration
│   └── logging_setup.py   # Logging configuration
├── frontend/
│   └── src/
│       ├── components/    # React components
│       └── pages/         # Page components
└── docs/
    └── SPECIFICATION.md   # Project specification
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/chat` | Send chat message |
| GET | `/api/v1/chat/{session_id}/history` | Get session history |
| DELETE | `/api/v1/chat/{session_id}` | Delete session |
| GET | `/api/v1/health` | Health check |

## Configuration

Environment variables in `.env`:

```bash
# Google Gemini API
GEMINI_API_KEY=your-api-key-here

# Models
LLM_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=text-embedding-004

# RAG Configuration
VECTOR_SEARCH_TOP_K=10
FINAL_TOP_K=5
MIN_SIMILARITY=0.3
```


## How It Works

1. **Query Rewriting**: When a user asks a question in everyday Indonesian, the system uses an LLM to expand the query with formal legal terminology. For example, "lampu merah" (red light) becomes "Alat Pemberi Isyarat Lalu Lintas" (Traffic Signal Device).

2. **Hybrid Retrieval**: The system performs multiple vector searches using the original query, the expanded legal query, and individual legal terms. Results are combined using Reciprocal Rank Fusion (RRF).

3. **Response Generation**: The LLM generates a response based on the retrieved legal text chunks, citing specific articles and paragraphs.

## Docker Deployment

### Prerequisites

- Docker and Docker Compose installed
- Gemini API key

### Deploy with Docker Compose

1. Clone the repository and navigate to the project root:

```bash
git clone <repository-url>
cd tanya-lalin
```

2. Create environment file:

```bash
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

3. Build and start the services:

```bash
docker-compose up -d --build
```

4. Open http://localhost in your browser.

### Useful Commands

```bash
# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# View service status
docker-compose ps
```

### Production Notes

- ChromaDB data is persisted in a Docker volume (`tanya-lalin-chroma-data`)
- The frontend is served via nginx on port 80
- API requests are proxied from frontend to backend internally
- Health checks are configured for both services

## License

GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
