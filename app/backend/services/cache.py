"""
LRU in-memory cache for embedding vectors.

Prevents redundant Ollama calls when the same text is embedded multiple
times (e.g. repeated queries, re-uploads of the same doc).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import List


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# Module-level cache: key = sha256(text), value = embedding list
_cache: dict[str, List[float]] = {}
_MAX_SIZE = 2048  # maximum number of cached embeddings


def get_cached_embedding(text: str) -> List[float] | None:
    return _cache.get(_hash(text))


def set_cached_embedding(text: str, embedding: List[float]) -> None:
    if len(_cache) >= _MAX_SIZE:
        # Evict oldest entry (dict preserves insertion order in Python 3.7+)
        oldest_key = next(iter(_cache))
        del _cache[oldest_key]
    _cache[_hash(text)] = embedding


def cache_size() -> int:
    return len(_cache)


def clear_cache() -> None:
    _cache.clear()
