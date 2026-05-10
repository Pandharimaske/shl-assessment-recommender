FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install project dependencies
COPY pyproject.toml .
RUN uv sync --no-dev

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Pre-download the fastembed ONNX model into the image at build time.
# fastembed uses ONNX runtime (no PyTorch) — ~80 MB RAM at runtime vs ~400 MB for torch.
# Baking the model here means zero download latency on cold start.
RUN uv run python -c "from fastembed import TextEmbedding; m = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2'); list(m.embed(['warmup'])); print('fastembed model cached.')"

# Expose port
EXPOSE 8000

# Health check — allow 2 min for cold start
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
