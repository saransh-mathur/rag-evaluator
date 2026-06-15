"""Embedding service using local Nomic model, with LRU cache."""

from __future__ import annotations

import requests
import numpy as np
from typing import List
import os
from dotenv import load_dotenv

from services.cache import get_cached_embedding, set_cached_embedding

load_dotenv()

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11434")
EMBED_MODEL    = os.getenv("EMBED_MODEL",    "nomic-embed-text")


def embed_text(text: str) -> List[float]:
    """
    Embed a single text using the local Nomic model.
    Results are cached by text hash to avoid redundant Ollama calls.

    Returns:
        768-dimensional float list
    """
    cached = get_cached_embedding(text)
    if cached is not None:
        return cached

    try:
        response = requests.post(
            f"{EMBED_BASE_URL.rstrip('/')}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=120,
        )
        response.raise_for_status()
        embedding = response.json()["embedding"]
        set_cached_embedding(text, embedding)
        return embedding
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not connect to Ollama at {EMBED_BASE_URL}. "
            "Make sure Ollama is running: ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed multiple texts, using cache where available."""
    return [embed_text(t) for t in texts]


def embed_array(texts: List[str]) -> np.ndarray:
    """Embed multiple texts and return as numpy array of shape (N, 768)."""
    return np.array(embed_batch(texts), dtype=np.float32)
