"""Reusable helper functions for RAG evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def read_json(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def contains_all(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return all(phrase.lower() in lowered for phrase in phrases)


def contains_any(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in phrases)


def top_k_similar(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    k: int,
) -> list[tuple[int, float]]:
    if doc_vecs.size == 0:
        return []
    sims = cosine_similarity(query_vec.reshape(1, -1), doc_vecs)[0]
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def retrieval_hit(retrieved_text: str, expected_phrases: list[str]) -> bool:
    if not expected_phrases:
        return True
    return contains_any(retrieved_text, expected_phrases)


def answer_hit(generated: str, expected_phrases: list[str]) -> bool:
    if not expected_phrases:
        return True
    return contains_all(generated, expected_phrases)


def hallucination_flag(
    generated: str,
    context: str,
    expected_phrases: list[str],
) -> bool:
    """True when the answer misses expected facts but still looks confident."""
    if answer_hit(generated, expected_phrases):
        return False
    if not generated.strip():
        return False
    return not contains_any(generated, expected_phrases)
