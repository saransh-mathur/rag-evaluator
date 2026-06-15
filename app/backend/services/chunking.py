"""Text chunking utilities for the FastAPI backend."""

from __future__ import annotations

from typing import List


def clean_text(text: str) -> str:
    """Remove null bytes and surrounding whitespace."""
    return text.replace("\x00", "").strip()


def chunk_text(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> List[str]:
    """
    Split *text* into overlapping fixed-size chunks.

    Args:
        text: Source text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        Non-empty list of chunk strings.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)

    return [c for c in chunks if c]
