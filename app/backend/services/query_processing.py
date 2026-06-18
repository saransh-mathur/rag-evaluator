"""Query normalization helpers used before retrieval."""

from __future__ import annotations

import re


ABBREVIATIONS = {
    "auth": "authentication",
    "db": "database",
    "k8s": "kubernetes",
    "llm": "large language model",
    "pg": "postgres",
    "postgresql": "postgres",
    "rag": "retrieval augmented generation",
    "s3": "simple storage service",
}


def normalize_query(query: str) -> str:
    """
    Normalize a user query for lexical and vector retrieval.

    This intentionally stays conservative: it removes noisy punctuation and
    expands common technical abbreviations without changing query meaning.
    """
    q = query.lower()
    q = re.sub(r"[^\w\s]", " ", q)
    words = [ABBREVIATIONS.get(word, word) for word in q.split()]
    return " ".join(words)
