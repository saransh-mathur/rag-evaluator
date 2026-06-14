"""Vector retrieval service using PostgreSQL pgvector."""

from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Tuple
from db.models import DocumentChunk
from services.embeddings import embed_text


def search_similar_chunks(
    db: Session,
    query: str,
    top_k: int = 5,
    user_id: str = None,
) -> List[Tuple[DocumentChunk, float]]:
    """
    Search for similar chunks using vector similarity.
    
    Args:
        db: Database session
        query: Query text to search
        top_k: Number of top results to return
        user_id: Filter by user (optional)
        
    Returns:
        List of (DocumentChunk, similarity_score) tuples
    """
    # Embed the query
    query_embedding = embed_text(query)
    
    # Build base query
    base_query = db.query(
        DocumentChunk,
        (1 - func.cosine_distance(
            DocumentChunk.embedding,
            query_embedding
        )).label("similarity")
    )
    
    # Filter by user if provided
    if user_id:
        base_query = base_query.join(
            DocumentChunk.document
        ).filter(
            DocumentChunk.document.user_id == user_id
        )
    
    # Order by similarity and limit
    results = base_query.order_by(
        func.cosine_distance(
            DocumentChunk.embedding,
            query_embedding
        )
    ).limit(top_k).all()
    
    return results


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
