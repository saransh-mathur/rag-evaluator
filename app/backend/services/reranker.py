"""Pluggable rerankers for retrieved chunks."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "tfidf").lower()
DEFAULT_CROSS_ENCODER_MODEL = os.getenv(
    "RERANK_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RERANK_ALLOW_DOWNLOAD = os.getenv("RERANK_ALLOW_DOWNLOAD", "false").lower() == "true"


def rerank(
    query: str,
    candidates: list[tuple[Any, float]],
    top_n: int | None = None,
    provider: str | None = None,
) -> list[tuple[Any, float]]:
    """
    Re-rank retrieved (chunk, score) pairs using the configured provider.

    Args:
        query:      User question
        candidates: List of (chunk_object, score) pairs
        top_n:      Return only top_n results (None = return all)
        provider:   "tfidf", "cross_encoder", or "none"

    Returns:
        Re-ranked list of (chunk_object, score)
    """
    if not candidates:
        return candidates

    selected_provider = (provider or DEFAULT_RERANK_PROVIDER).lower()
    if selected_provider in {"none", "off", "disabled"}:
        return candidates[:top_n] if top_n else candidates
    if selected_provider in {"cross-encoder", "cross_encoder", "crossencoder"}:
        return _cross_encoder_rerank(query, candidates, top_n)
    if selected_provider != "tfidf":
        logger.warning("Unknown rerank provider '%s'; falling back to tfidf", provider)

    return _tfidf_rerank(query, candidates, top_n)


def _tfidf_rerank(
    query: str,
    candidates: list[tuple[Any, float]],
    top_n: int | None = None,
) -> list[tuple[Any, float]]:
    """Re-rank candidates with local TF-IDF cosine similarity."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

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
        logger.warning("TF-IDF rerank failed; returning original order", exc_info=True)
        return candidates[:top_n] if top_n else candidates


def _cross_encoder_rerank(
    query: str,
    candidates: list[tuple[Any, float]],
    top_n: int | None = None,
) -> list[tuple[Any, float]]:
    """Re-rank candidates with a sentence-transformers CrossEncoder."""
    try:
        model = _cross_encoder_model()
        pairs = [(query, chunk.text) for chunk, _ in candidates]
        raw_scores = model.predict(pairs)

        reranked = [
            (chunk, round(float(score), 4))
            for (chunk, _), score in zip(candidates, raw_scores)
        ]
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked[:top_n] if top_n else reranked
    except Exception:
        logger.warning(
            "Cross-encoder rerank failed; falling back to tfidf",
            exc_info=True,
        )
        return _tfidf_rerank(query, candidates, top_n)


@lru_cache(maxsize=1)
def _cross_encoder_model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        DEFAULT_CROSS_ENCODER_MODEL,
        automodel_args={"local_files_only": not RERANK_ALLOW_DOWNLOAD},
        tokenizer_args={"local_files_only": not RERANK_ALLOW_DOWNLOAD},
        config_args={"local_files_only": not RERANK_ALLOW_DOWNLOAD},
    )
