FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install PyTorch CPU-only FIRST (before uv sync) to keep image small on Render.
# sentence-transformers needs torch but the GPU variant is 2 GB+ — CPU is ~300 MB.
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
COPY pyproject.toml .
RUN uv sync --no-dev

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Pre-download the embedding model into the image at build time.
# This means zero model-download latency on cold start.
RUN python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('Model cached.')
"

# Expose port
EXPOSE 8000

# Health check — allow 2 min for cold start (model load + Pinecone init)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
