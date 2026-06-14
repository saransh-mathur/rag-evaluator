"""Query and chat API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import json
from db.connection import get_db
from db.models import QueryHistory, User, DocumentChunk
from services.retrieval import search_similar_chunks
from services.generation import generate_answer, generate_with_history

router = APIRouter(prefix="/api/queries", tags=["queries"])


class QueryRequest(BaseModel):
    """User query request."""
    question: str
    user_id: str
    top_k: int = 5
    temperature: float = 0.1


class QueryResponse(BaseModel):
    """Query response with answer and sources."""
    question: str
    answer: str
    retrieved_chunks: List[dict]
    top_similarity: float
    query_id: int


class HistoryResponse(BaseModel):
    """Query history item."""
    id: int
    question: str
    answer: str
    created_at: str
    chunk_count: int


@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    req: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Process a user question and return answer with sources.
    
    1. Embed the question
    2. Search for similar chunks
    3. Generate answer using context
    4. Save to history
    """
    try:
        # Ensure user exists
        user = db.query(User).filter(User.id == req.user_id).first()
        if not user:
            user = User(id=req.user_id)
            db.add(user)
            db.flush()
        
        # Search similar chunks
        search_results = search_similar_chunks(
            db,
            query=req.question,
            top_k=req.top_k,
            user_id=req.user_id
        )
        
        if not search_results:
            return QueryResponse(
                question=req.question,
                answer="No documents found. Please upload documents first.",
                retrieved_chunks=[],
                top_similarity=0.0,
                query_id=-1
            )
        
        # Prepare context
        retrieved_chunks = []
        context_parts = []
        top_similarity = 0.0
        
        for chunk, similarity in search_results:
            retrieved_chunks.append({
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "filename": chunk.document.filename,
                "text": chunk.text[:200] + "...",  # Preview
                "similarity": round(float(similarity), 4)
            })
            context_parts.append(chunk.text)
            if similarity > top_similarity:
                top_similarity = similarity
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer
        answer = generate_answer(
            question=req.question,
            context=context,
            temperature=req.temperature
        )
        
        # Save to history
        retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]
        history = QueryHistory(
            user_id=req.user_id,
            question=req.question,
            answer=answer,
            retrieved_chunks_ids=json.dumps(retrieved_ids),
            top_similarity=top_similarity
        )
        db.add(history)
        db.commit()
        
        return QueryResponse(
            question=req.question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            top_similarity=top_similarity,
            query_id=history.id
        )
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_query_history(
    user_id: str,
    limit: int = 50,
    db: Session = Depends(get_db)
) -> List[HistoryResponse]:
    """Get query history for a user."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    
    queries = db.query(QueryHistory).filter(
        QueryHistory.user_id == user_id
    ).order_by(
        QueryHistory.created_at.desc()
    ).limit(limit).all()
    
    return [
        HistoryResponse(
            id=q.id,
            question=q.question,
            answer=q.answer,
            created_at=q.created_at.isoformat(),
            chunk_count=len(json.loads(q.retrieved_chunks_ids or "[]"))
        )
        for q in queries
    ]


@router.get("/history/{query_id}")
async def get_query_detail(
    query_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed view of a specific query with all chunk info."""
    query = db.query(QueryHistory).filter(
        QueryHistory.id == query_id
    ).first()
    
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")
    
    chunk_ids = json.loads(query.retrieved_chunks_ids or "[]")
    chunks = db.query(DocumentChunk).filter(
        DocumentChunk.id.in_(chunk_ids)
    ).all() if chunk_ids else []
    
    return {
        "id": query.id,
        "question": query.question,
        "answer": query.answer,
        "created_at": query.created_at.isoformat(),
        "retrieved_chunks": [
            {
                "chunk_id": c.id,
                "filename": c.document.filename,
                "text": c.text,
                "chunk_index": c.chunk_index
            }
            for c in chunks
        ]
    }


@router.delete("/history/{query_id}")
async def delete_query(
    query_id: int,
    user_id: str = None,
    db: Session = Depends(get_db)
):
    """Delete a query from history."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    
    query = db.query(QueryHistory).filter(
        QueryHistory.id == query_id,
        QueryHistory.user_id == user_id
    ).first()
    
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")
    
    db.delete(query)
    db.commit()
    
    return {"message": "Query deleted successfully"}
