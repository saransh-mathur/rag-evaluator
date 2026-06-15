"""Vector retrieval service using PostgreSQL pgvector."""

from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import Session
from sqlalchemy import func, cast as sa_cast, Float
from typing import List, Tuple
from services.embeddings import embed_text
from db.models import Document, DocumentChunk


def search_similar_chunks(
    db: Session,
    query: str,
    top_k: int = 5,
    user_id: str = None,
) -> List[Tuple[DocumentChunk, float]]:

    query_embedding = embed_text(query)

    query_embedding = [float(x) for x in query_embedding]

    # cast the distance expression to Float so pgvector's type processor
    # doesn't attempt to deserialize it as a Vector
    distance_expr = sa_cast(
        DocumentChunk.embedding.op("<=>")(query_embedding),
        Float
    )

    base_query = db.query(
        DocumentChunk,
        distance_expr.label("distance")
    )

    if user_id:
        print("applying user filter via join on Document")
        base_query = (
            base_query
            .join(Document, DocumentChunk.document_id == Document.id)
            .filter(Document.user_id == user_id)
        )

    print("executing query...")
    results = (
        base_query
        .order_by(distance_expr)
        .limit(top_k)
        .all()
    )

    print("results count:", len(results))
    final_results = []
    for idx, item in enumerate(results):
        chunk, distance = item
        similarity = 1 - float(distance)
        print(f"result[{idx}] chunk_id={chunk.id}, distance={distance}, similarity={similarity}")
        final_results.append((chunk, similarity))

    return final_results

def get_chunk_by_id(db: Session, chunk_id: int) -> DocumentChunk:
    """Get a specific chunk by ID."""
    return db.query(DocumentChunk).filter(
        DocumentChunk.id == chunk_id
    ).first()


def get_user_chunks(db: Session, user_id: str) -> List[DocumentChunk]:
    """Get all chunks for a user."""
    return db.query(DocumentChunk).join(
        DocumentChunk.document
    ).filter(
        DocumentChunk.document.user_id == user_id
    ).all()
