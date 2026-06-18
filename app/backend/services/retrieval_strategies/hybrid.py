"""Hybrid retrieval pipeline backed by the current production search."""

from __future__ import annotations

from sqlalchemy.orm import Session

from services.retrieval import search_similar_chunks
from services.reranker import DEFAULT_RERANK_PROVIDER
from services.retrieval_strategies.base import RetrievalResult


class HybridPipeline:
    """Current pgvector + BM25 + RRF + optional reranker pipeline."""

    name = "hybrid"

    def __init__(self, use_hyde: bool = False, use_rerank: bool = True) -> None:
        self.use_hyde = use_hyde
        self.use_rerank = use_rerank

    def retrieve(
        self,
        db: Session,
        query: str,
        top_k: int = 8,
        user_id: str | None = None,
        chat_history: list[dict] | None = None,
        doc_mode: bool = True,
    ) -> RetrievalResult:
        chunks = search_similar_chunks(
            db=db,
            query=query,
            top_k=top_k,
            user_id=user_id,
            use_hyde=self.use_hyde,
            use_rerank=self.use_rerank,
            doc_mode=doc_mode,
        )
        return RetrievalResult(
            chunks=chunks,
            metadata={
                "strategy": self.name,
                "stages": ["vector", "bm25", "rrf", "rerank"],
                "use_hyde": self.use_hyde,
                "use_rerank": self.use_rerank,
                "rerank_provider": DEFAULT_RERANK_PROVIDER if self.use_rerank else "none",
            },
        )
