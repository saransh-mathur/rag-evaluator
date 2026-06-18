"""Maximal Marginal Relevance diversification for retrieved chunks."""

from __future__ import annotations

from typing import Any

import numpy as np


def maximal_marginal_relevance(
    query_embedding: list[float],
    candidates: list[tuple[Any, float]],
    top_k: int,
    lambda_mult: float = 0.7,
) -> list[tuple[Any, float]]:
    """
    Select relevant but non-redundant chunks from ranked candidates.

    lambda_mult closer to 1 favors relevance; closer to 0 favors diversity.
    """
    if top_k <= 0 or not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates[:top_k]

    try:
        query_vec = _normalize_vector(query_embedding)
        candidate_vecs = [_normalize_vector(chunk.embedding) for chunk, _ in candidates]
        matrix = np.vstack(candidate_vecs)
    except Exception:
        return candidates[:top_k]

    relevance = matrix @ query_vec
    selected: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected) < top_k:
        if not selected:
            chosen = max(remaining, key=lambda idx: relevance[idx])
        else:
            selected_matrix = matrix[selected]
            similarity_to_selected = matrix[list(remaining)] @ selected_matrix.T
            max_similarity = similarity_to_selected.max(axis=1)
            remaining_indices = list(remaining)
            mmr_scores = (
                lambda_mult * relevance[remaining_indices]
                - (1.0 - lambda_mult) * max_similarity
            )
            chosen = remaining_indices[int(np.argmax(mmr_scores))]

        selected.append(chosen)
        remaining.remove(chosen)

    return [candidates[idx] for idx in selected]


def _normalize_vector(vector: Any) -> np.ndarray:
    arr = np.asarray(list(vector), dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("zero vector")
    return arr / norm
