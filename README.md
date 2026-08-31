# Tanya Lalin

Tanya Lalin is a portfolio-grade RAG application for questions about Indonesian road traffic and transportation rules. It turns everyday Indonesian questions into evidence-grounded answers with traceable citations to official legal text.

The application is intentionally small enough to understand end to end, while demonstrating production-minded architecture: a typed RAG pipeline, exact vector and BM25 retrieval, bounded validation and repair, session persistence, structured errors, rate limiting, observability, automated evaluation, and one deployable web image.

> **Corpus scope:** the bundled corpus contains the original text and official elucidation of Law No. 22 of 2009. It does not consolidate later amendments or Constitutional Court decisions. Tanya Lalin is an information tool, not a substitute for professional legal advice.

## Highlights

- One FastAPI service serves `/api/v1/*` and the built React application.
- Hybrid retrieval combines exact cosine search, dual-query BM25, explicit article lookup,
  family-normalized RRF, and a bounded `gpt-4o-mini` reranker.
- Answers are emitted atomically only after citation and groundedness validation.
- Unicode-normalized injection checks and corpus-scope policy stop unsafe or unverifiable
  requests before retrieval or generation.
- SSE communicates pipeline progress without exposing unvalidated model tokens.
- In-memory sessions work with zero infrastructure; Redis is optional.
- The 823-chunk, 1,536-dimensional corpus index is versioned and bundled.
- Python and TypeScript are strict, tested, linted, and enforced in CI.

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), and Node.js 20.

```bash
cp .env.example .env
# Set OPENAI_API_KEY in .env
make install
```

Run the backend and frontend in separate terminals:

```bash
cd backend && uv run uvicorn app.main:create_app --factory --reload
```

```bash
cd frontend && npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` to FastAPI.

Run all local quality gates with:

```bash
make check
```

## Container deployment

```bash
docker compose up --build
```

The application is available on `http://127.0.0.1:8000`. To use Redis, set `REDIS_URL=redis://redis:6379/0` and start the profile:

```bash
docker compose --profile redis up --build
```

Use one application worker with in-memory sessions. Multiple workers require Redis so all workers share session state.

## Repository map

```text
backend/app/
  api/             HTTP contracts, errors, and routes
  application/     session-aware chat use case
  rag/             typed query-to-answer pipeline
  infrastructure/  corpus, providers, sessions, logging, rate limits
corpus/             committed chunks, vectors, and manifest
frontend/src/
  api/              same-origin API and SSE client
  features/chat/    chat state and interface
docs/               product, architecture, data, evaluation, and operations
```

## API summary

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/chat` | Return one validated answer |
| `POST` | `/api/v1/chat/stream` | Stream progress and one complete answer |
| `GET` | `/api/v1/chat/{session_id}/history` | Restore a session with citations |
| `DELETE` | `/api/v1/chat/{session_id}` | Delete a session |
| `GET` | `/api/v1/health/live` | Process liveness |
| `GET` | `/api/v1/health/ready` | Corpus, provider, and session readiness |

See [Product specification](docs/PRODUCT_SPEC.md), [Architecture](docs/ARCHITECTURE.md), [Data](docs/DATA.md), [Evaluation](docs/EVALUATION.md), and [Operations](docs/OPERATIONS.md).

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
