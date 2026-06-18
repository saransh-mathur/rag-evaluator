"""Advanced retrieval pipeline with normalized query and full-corpus BM25."""

from __future__ import annotations

import logging
import os

from sqlalchemy import Float, cast as sa_cast
from sqlalchemy.orm import Session

from db.models import Document, DocumentChunk
from services.embeddings import embed_text
from services.faiss_store import faiss_store
from services.generation import generate_query_expansions
from services.mmr import maximal_marginal_relevance
from services.query_processing import normalize_query
from services.reranker import DEFAULT_RERANK_PROVIDER, rerank
from services.retrieval import _bm25_score, _reciprocal_rank_fusion
from services.retrieval_strategies.base import RetrievalResult

logger = logging.getLogger(__name__)
VECTOR_STORE = os.getenv("VECTOR_STORE", "faiss").lower()


class AdvancedPipeline:
    """
    Normalized hybrid retrieval over independent vector and BM25 candidate sets.

    Unlike the legacy hybrid pipeline, BM25 searches the full user corpus rather
    than only the chunks already found by vector search.
    """

    name = "advanced"
    expansion_count = 4
    rerank_provider = DEFAULT_RERANK_PROVIDER

    def retrieve(
        self,
        db: Session,
        query: str,
        top_k: int = 8,
        user_id: str | None = None,
        chat_history: list[dict] | None = None,
        doc_mode: bool = True,
    ) -> RetrievalResult:
        if not doc_mode:
            return RetrievalResult(chunks=[], metadata={"strategy": self.name})

        normalized_query = normalize_query(query)
        expanded_queries = self._expand_queries(query, normalized_query)
        retrieval_queries = [normalized_query] + expanded_queries
        fetch_k = max(top_k * 4, 20)

        vector_ranked = self._search_all_variants(
            search_fn=self._vector_search,
            db=db,
            queries=retrieval_queries,
            top_k=fetch_k,
            user_id=user_id,
        )
        bm25_ranked = self._search_all_variants(
            search_fn=self._bm25_search,
            db=db,
            queries=retrieval_queries,
            top_k=fetch_k,
            user_id=user_id,
        )

        fused = _reciprocal_rank_fusion(vector_ranked, bm25_ranked)
        rerank_k = max(top_k * 3, top_k)
        if len(fused) > 1:
            reranked = rerank(
                normalized_query,
                fused,
                top_n=rerank_k,
                provider=self.rerank_provider,
            )
        else:
            reranked = fused[:rerank_k]

        query_embedding = embed_text(normalized_query)
        diversified = maximal_marginal_relevance(
            query_embedding=query_embedding,
            candidates=reranked,
            top_k=top_k,
        )

        logger.info(
            "AdvancedPipeline: query='%s' normalized='%s' expansions=%s vector=%s bm25=%s final=%s",
            query[:50],
            normalized_query[:50],
            len(expanded_queries),
            len(vector_ranked),
            len(bm25_ranked),
            len(diversified),
        )
        return RetrievalResult(
            chunks=diversified,
            metadata={
                "strategy": self.name,
                "rerank_provider": self.rerank_provider,
                "normalized_query": normalized_query,
                "expanded_queries": expanded_queries,
                "stages": [
                    "normalize",
                    "query_expansion",
                    "vector",
                    "full_corpus_bm25",
                    "rrf",
                    "rerank",
                    "mmr",
                ],
            },
        )

    def _expand_queries(self, query: str, normalized_query: str) -> list[str]:
        expansions = generate_query_expansions(query, n=self.expansion_count)
        normalized_expansions: list[str] = []
        seen = {normalized_query}

        for expansion in expansions:
            normalized = normalize_query(expansion)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_expansions.append(normalized)

        return normalized_expansions

    def _search_all_variants(
        self,
        search_fn,
        db: Session,
        queries: list[str],
        top_k: int,
        user_id: str | None,
    ) -> list[tuple[DocumentChunk, float]]:
        best_by_chunk_id: dict[int, tuple[DocumentChunk, float]] = {}

        for variant in queries:
            for chunk, score in search_fn(
                db=db,
                query=variant,
                top_k=top_k,
                user_id=user_id,
            ):
                current = best_by_chunk_id.get(chunk.id)
                if current is None or score > current[1]:
                    best_by_chunk_id[chunk.id] = (chunk, score)

        ranked = sorted(
            best_by_chunk_id.values(),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]

    def _vector_search(
        self,
        db: Session,
        query: str,
        top_k: int,
        user_id: str | None,
    ) -> list[tuple[DocumentChunk, float]]:
        query_embedding = [float(x) for x in embed_text(query)]
        if VECTOR_STORE == "faiss":
            return faiss_store.search_chunks(db, query_embedding, top_k, user_id)
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

        rows = base_q.order_by(distance_expr).limit(top_k).all()
        return [
            (chunk, round(1.0 - float(distance), 4))
            for chunk, distance in rows
        ]

    def _bm25_search(
        self,
        db: Session,
        query: str,
        top_k: int,
        user_id: str | None,
    ) -> list[tuple[DocumentChunk, float]]:
        base_q = db.query(DocumentChunk)
        if user_id:
            base_q = (
                base_q
                .join(Document, DocumentChunk.document_id == Document.id)
                .filter(Document.user_id == user_id)
            )

        chunks = base_q.all()
        scores = _bm25_score(query, chunks)
        ranked = sorted(
            zip(chunks, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            (chunk, round(float(score), 4))
            for chunk, score in ranked[:top_k]
            if float(score) > 0.0
        ]
