# SHL Assessment Recommender

Conversational AI agent that recommends SHL assessments based on role descriptions.

## Setup

```bash
# Install uv
pip install uv

# Create venv and install deps
uv venv
uv sync

# Copy env file
cp .env.example .env
# Fill in your API keys

# Run scraper first (one-time)
uv run python scripts/scrape_catalog.py

# Start server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or with Docker
docker compose up --build
```

## Endpoints

- `GET /health` → `{"status": "ok"}`
- `POST /chat` → conversational recommendation
