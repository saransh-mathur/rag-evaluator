"""Main FastAPI application."""

from __future__ import annotations

import logging
import os

import requests as http_requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from db.connection import init_db, engine
from api import documents, queries

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database…")
    try:
        init_db()
        logger.info("Database initialised successfully")
    except Exception as e:
        logger.error(f"Database initialisation failed: {e}")
    yield
    logger.info("Shutting down…")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG AI Chat API",
    description="Local RAG evaluation with hybrid retrieval, streaming, and evaluation",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(queries.router)


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "RAG AI Chat API v2", "docs": "/docs"}


@app.get("/health")
async def health():
    """
    Deep health check: verifies PostgreSQL and Ollama are reachable.
    Returns overall status + per-dependency status.
    """
    results: dict[str, str] = {}

    # PostgreSQL
    try:
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        results["postgres"] = "ok"
    except Exception as e:
        results["postgres"] = f"error: {e}"

    # Ollama
    ollama_url = os.getenv("EMBED_BASE_URL", "http://localhost:11434")
    try:
        r = http_requests.get(f"{ollama_url}/api/tags", timeout=3)
        results["ollama"] = "ok" if r.status_code == 200 else f"http {r.status_code}"
    except Exception as e:
        results["ollama"] = f"error: {e}"

    overall = "healthy" if all(v == "ok" for v in results.values()) else "degraded"
    return {"status": overall, "dependencies": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
