"""
Lightweight re-ranker using TF-IDF cosine similarity.

Falls back gracefully if sklearn is unavailable.
For production use, swap with a cross-encoder model
(e.g. cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers).
"""

from __future__ import annotations

from typing import Any


def rerank(
    query: str,
    candidates: list[tuple[Any, float]],
    top_n: int | None = None,
) -> list[tuple[Any, float]]:
    """
    Re-rank retrieved (chunk, score) pairs by TF-IDF similarity to the query.

    Args:
        query:      User question
        candidates: List of (chunk_object, vector_score) from pgvector
        top_n:      Return only top_n results (None = return all)

    Returns:
        Re-ranked list of (chunk_object, combined_score)
    """
    if not candidates:
        return candidates

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        texts = [c.text for c, _ in candidates]
        corpus = [query] + texts

        vec = TfidfVectorizer(stop_words="english", max_features=10000)
        tfidf = vec.fit_transform(corpus)

        query_vec = tfidf[0]
        doc_vecs  = tfidf[1:]
        tfidf_scores = cosine_similarity(query_vec, doc_vecs)[0]

        # Combine: 60% vector similarity + 40% TF-IDF
        combined = []
        for i, (chunk, vec_score) in enumerate(candidates):
            score = 0.6 * vec_score + 0.4 * float(tfidf_scores[i])
            combined.append((chunk, round(score, 4)))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_n] if top_n else combined

    except Exception:
        # If sklearn fails for any reason, return original order
        return candidates[:top_n] if top_n else candidates
