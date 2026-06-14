"""Embedding service using local Nomic model."""

import requests
import numpy as np
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def embed_text(text: str) -> List[float]:
    """
    Embed a single text using local Nomic model.
    
    Args:
        text: Text to embed
        
    Returns:
        Vector embedding (768 dimensions)
    """
    try:
        response = requests.post(
            f"{EMBED_BASE_URL.rstrip('/')}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not connect to Ollama at {EMBED_BASE_URL}. "
            "Make sure Ollama is running: ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Embed multiple texts.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embeddings
    """
    embeddings = []
    for text in texts:
        embeddings.append(embed_text(text))
    return embeddings


def embed_array(texts: List[str]) -> np.ndarray:
    """
    Embed multiple texts and return as numpy array.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        Numpy array of shape (len(texts), 768)
    """
    embeddings = embed_batch(texts)
    return np.array(embeddings, dtype=np.float32)
