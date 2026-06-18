"""
Vector + BM25 hybrid retrieval with optional HyDE query expansion and re-ranking.

Search pipeline:
  1. (Optional) HyDE — expand query to hypothetical answer for better embedding
  2. pgvector cosine search — top_k * 2 candidates
  3. BM25 keyword search  — over the same candidate pool
  4. Reciprocal Rank Fusion — merge vector + BM25 scores
  5. Re-ranker             — TF-IDF re-scoring of top candidates
  6. Return top_k results
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import cast as sa_cast, Float, text

from services.embeddings import embed_text
from services.faiss_store import faiss_store
from services.reranker import rerank
from db.models import Document, DocumentChunk

logger = logging.getLogger(__name__)


VECTOR_STORE = os.getenv("VECTOR_STORE", "faiss").lower()

# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------

def _bm25_score(query: str, chunks: list[DocumentChunk]) -> list[float]:
    """Score chunks against query using BM25."""
    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [c.text.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())
        # Normalize to [0, 1]
        max_s = max(scores) if max(scores) > 0 else 1.0
        return [float(s / max_s) for s in scores]
    except Exception:
        return [0.0] * len(chunks)


def _reciprocal_rank_fusion(
    vector_ranked: list[tuple[DocumentChunk, float]],
    bm25_ranked:   list[tuple[DocumentChunk, float]],
    k: int = 60,
) -> list[tuple[DocumentChunk, float]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    k=60 is the standard constant from the RRF paper.
    """
    scores: dict[int, float] = {}
    chunk_map: dict[int, DocumentChunk] = {}

    for rank, (chunk, _) in enumerate(vector_ranked, start=1):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
        chunk_map[chunk.id] = chunk

    for rank, (chunk, _) in enumerate(bm25_ranked, start=1):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
        chunk_map[chunk.id] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [(chunk_map[cid], round(scores[cid], 6)) for cid in sorted_ids]


# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------

def search_similar_chunks(
    db: Session,
    query: str,
    top_k: int = 8,
    user_id: str | None = None,
    use_hyde: bool = False,
    use_rerank: bool = True,
    doc_mode: bool = True,
) -> List[Tuple[DocumentChunk, float]]:
    """
    Hybrid vector + BM25 search with optional HyDE and re-ranking.

    Args:
        db:          SQLAlchemy session
        query:       User question
        top_k:       Number of final results to return
        user_id:     Scope results to this user's documents
        use_hyde:    Expand query via HyDE before embedding
        use_rerank:  Re-rank results with TF-IDF cross-scoring
        doc_mode:    If False, skip retrieval (general chat mode)

    Returns:
        List of (DocumentChunk, similarity_score) sorted descending
    """
    if not doc_mode:
        return []

    # --- Step 1: HyDE query expansion ---
    embed_query = query
    if use_hyde:
        try:
            from services.generation import generate_hypothetical_document
            hypothetical = generate_hypothetical_document(query)
            if hypothetical and hypothetical != query:
                embed_query = hypothetical
                logger.info(f"HyDE expanded query: {hypothetical[:80]}")
        except Exception:
            pass  # fall back to original query

    # --- Step 2: Vector search (fetch 2× top_k for re-ranking pool) ---
    fetch_k = top_k * 2
    query_embedding = [float(x) for x in embed_text(embed_query)]

    if VECTOR_STORE == "faiss":
        vector_ranked = faiss_store.search_chunks(db, query_embedding, fetch_k, user_id)
    else:
        distance_expr = sa_cast(
            DocumentChunk.embedding.op("<=>")(query_embedding),
            Float,
        )
        base_q = db.query(DocumentChunk, distance_expr.label("distance"))
        if user_id:
            base_q = (
                base_q
                .join(Document, DocumentChunk.document_id == Document.id)
                .filter(Document.user_id == user_id)
            )
        vector_rows = (
            base_q
            .order_by(distance_expr)
            .limit(fetch_k)
            .all()
        )
        vector_ranked = [
            (chunk, round(1.0 - float(dist), 4))
            for chunk, dist in vector_rows
        ]
        
    if not vector_ranked:
        return []

    chunks_pool = [c for c, _ in vector_ranked]

    # --- Step 3: BM25 over the candidate pool ---
    bm25_scores = _bm25_score(query, chunks_pool)
    bm25_ranked = sorted(
        zip(chunks_pool, bm25_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    # --- Step 4: RRF merge ---
    fused = _reciprocal_rank_fusion(vector_ranked, list(bm25_ranked))

    # --- Step 5: Re-rank ---
    if use_rerank and len(fused) > 1:
        fused = rerank(query, fused, top_n=top_k)
    else:
        fused = fused[:top_k]

    logger.info(
        f"search_similar_chunks: query='{query[:50]}' "
        f"pool={len(chunks_pool)} final={len(fused)}"
    )
    return fused


def get_chunk_by_id(db: Session, chunk_id: int) -> DocumentChunk | None:
    return db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()


def search_document_text(
    db: Session,
    document_id: int,
    query: str,
    limit: int = 10,
) -> List[DocumentChunk]:
    """
    Simple keyword search within a single document's chunks.
    Used by the document search endpoint.
    """
    q = query.lower()
    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .all()
    )
    matched = [c for c in chunks if q in (c.text or "").lower()]
    return matched[:limit]
