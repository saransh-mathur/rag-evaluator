"""Basic vector-only retrieval pipeline."""

from __future__ import annotations

import logging
import os

from sqlalchemy import Float, cast as sa_cast
from sqlalchemy.orm import Session

from db.models import Document, DocumentChunk
from services.embeddings import embed_text
from services.faiss_store import faiss_store
from services.retrieval_strategies.base import RetrievalResult

logger = logging.getLogger(__name__)
VECTOR_STORE = os.getenv("VECTOR_STORE", "faiss").lower()


class BasicPipeline:
    """pgvector cosine search without BM25, RRF, HyDE, or reranking."""

    name = "basic"

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

        query_embedding = [float(x) for x in embed_text(query)]
        if VECTOR_STORE == "faiss":
            chunks = faiss_store.search_chunks(db, query_embedding, top_k, user_id)
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
            rows = base_q.order_by(distance_expr).limit(top_k).all()
            chunks = [
                (chunk, round(1.0 - float(distance), 4))
                for chunk, distance in rows
            ]

        logger.info(
            "BasicPipeline: query='%s' final=%s",
            query[:50],
            len(chunks),
        )
        return RetrievalResult(
            chunks=chunks,
            metadata={"strategy": self.name, "stages": ["vector"]},
        )
