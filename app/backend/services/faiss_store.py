"""FAISS in-memory vector store manager."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from sqlalchemy.orm import Session

from db.models import Document, DocumentChunk

logger = logging.getLogger(__name__)


class FaissStore:
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.index = None
        self.chunk_to_user: dict[int, str] = {}
        self._init_index()

    def _init_index(self) -> None:
        try:
            import faiss
            # IndexIDMap allows us to use PostgreSQL's chunk_id instead of FAISS's auto-incrementing ID
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        except ImportError:
            logger.warning("faiss-cpu not installed. FAISS store will be disabled.")
            self.index = None

    def load_from_db(self, db: Session) -> None:
        if self.index is None:
            return

        self.index.reset()
        self.chunk_to_user.clear()

        logger.info("Loading vectors from PostgreSQL into FAISS...")
        chunks = (
            db.query(DocumentChunk, Document.user_id)
            .join(Document, DocumentChunk.document_id == Document.id)
            .all()
        )

        ids, vecs = [], []
        for chunk, user_id in chunks:
            if chunk.embedding is not None:
                ids.append(chunk.id)
                vecs.append(chunk.embedding)
                self.chunk_to_user[chunk.id] = user_id

        if ids:
            self.add(ids, vecs, {i: u for i, u in zip(ids, [u for _, u in chunks])})
            logger.info(f"Successfully loaded {len(ids)} vectors into FAISS.")

    def add(self, chunk_ids: List[int], embeddings: List[List[float]], user_ids_map: dict[int, str]) -> None:
        if self.index is None or not chunk_ids:
            return
        import faiss
        vecs_np = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vecs_np)  # Normalize for Inner Product (makes it Cosine Similarity)
        ids_np = np.array(chunk_ids, dtype=np.int64)
        self.index.add_with_ids(vecs_np, ids_np)
        self.chunk_to_user.update(user_ids_map)

    def remove_document_chunks(self, chunk_ids: List[int]) -> None:
        if self.index is None or not chunk_ids:
            return
        import faiss
        self.index.remove_ids(np.array(chunk_ids, dtype=np.int64))
        for cid in chunk_ids:
            self.chunk_to_user.pop(cid, None)

    def search_chunks(
        self, db: Session, query_embedding: List[float], top_k: int, user_id: str | None = None
    ) -> List[Tuple[DocumentChunk, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        import faiss
        vec_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vec_np)

        fetch_k = self.index.ntotal if user_id else top_k
        distances, indices = self.index.search(vec_np, fetch_k)

        results = [(int(cid), float(d)) for cid, d in zip(indices[0], distances[0]) 
                   if cid != -1 and (not user_id or self.chunk_to_user.get(cid) == user_id)]
        results = results[:top_k]
        
        chunk_map = {c.id: c for c in db.query(DocumentChunk).filter(DocumentChunk.id.in_([cid for cid, _ in results])).all()}
        return [(chunk_map[cid], dist) for cid, dist in results if cid in chunk_map]

faiss_store = FaissStore()