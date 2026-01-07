# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Set PATH to use virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY config.py logging_setup.py ./

# Copy and set entrypoint script
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Create directory for ChromaDB
RUN mkdir -p /app/data/chroma_db /app/data/seed

# Env (can be overriden)
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD sh -c "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/api/v1/health')\""

# Run the entrypoint script
ENTRYPOINT ["./docker-entrypoint.sh"]
