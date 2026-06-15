"""Query and chat API endpoints."""

import json
import traceback
from typing import Iterator, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.connection import get_db
from db.models import DocumentChunk, QueryHistory, User
from services.generation import generate_answer, generate_answer_stream, generate_suggestions
from services.retrieval import search_similar_chunks

router = APIRouter(prefix="/api/queries", tags=["queries"])


class QueryRequest(BaseModel):
    """User query request."""
    question: str
    user_id: str
    top_k: int = 8
    temperature: float = 0.1
    max_tokens: int = 2048


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
async def ask_question(req: QueryRequest, db: Session = Depends(get_db)):
    """
    Process a user question and return answer with sources.

    1. Ensure user exists
    2. Search for similar chunks
    3. Generate answer using context
    4. Save to history
    """
    try:
        user = db.query(User).filter(User.id == req.user_id).first()
        if not user:
            user = User(id=req.user_id)
            db.add(user)
            db.flush()

        search_results = search_similar_chunks(
            db,
            query=req.question,
            top_k=req.top_k,
            user_id=req.user_id
        )

        if not search_results:
            answer = generate_answer(
                question=req.question,
                context="",
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )

            history = QueryHistory(
                user_id=req.user_id,
                question=req.question,
                answer=answer,
                retrieved_chunks_ids=json.dumps([]),
                top_similarity=0.0
            )
            db.add(history)
            db.commit()
            db.refresh(history)

            return QueryResponse(
                question=req.question,
                answer=answer,
                retrieved_chunks=[],
                top_similarity=0.0,
                query_id=history.id
            )

        retrieved_chunks = []
        context_parts = []
        top_similarity = 0.0

        for item in search_results:
            if isinstance(item, tuple) and len(item) == 2:
                chunk, similarity = item
            else:
                chunk, similarity = item, 0.0

            filename = None
            if getattr(chunk, "document", None) is not None:
                filename = getattr(chunk.document, "filename", None)

            retrieved_chunks.append({
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "filename": filename or "unknown",
                "text": (chunk.text[:200] + "...") if chunk.text and len(chunk.text) > 200 else (chunk.text or ""),
                "similarity": round(float(similarity), 4)
            })

            if chunk.text:
                context_parts.append(chunk.text)

            if similarity is not None and float(similarity) > top_similarity:
                top_similarity = float(similarity)

        context = "\n\n---\n\n".join(context_parts)

        answer = generate_answer(
            question=req.question,
            context=context,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

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
        db.refresh(history)

        return QueryResponse(
            question=req.question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            top_similarity=top_similarity,
            query_id=history.id
        )

    except Exception as e:
        print(traceback.format_exc())
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------

def _sse(payload: dict) -> str:
    """Format a dict as a Server-Sent Events data line."""
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/ask-stream")
async def ask_question_stream(req: QueryRequest, db: Session = Depends(get_db)):
    """
    Streaming version of /ask.

    SSE event sequence:
      1. {"type": "sources", "retrieved_chunks": [...], "top_similarity": float}
      2. {"type": "token",   "content": "<token>"}   — one per LLM token
      3. {"type": "done",    "query_id": int}
    """

    def event_stream() -> Iterator[str]:
        try:
            # Ensure user exists
            user = db.query(User).filter(User.id == req.user_id).first()
            if not user:
                db.add(User(id=req.user_id))
                db.flush()

            # --- Retrieval -------------------------------------------------
            search_results = search_similar_chunks(
                db,
                query=req.question,
                top_k=req.top_k,
                user_id=req.user_id,
            )

            retrieved_chunks: list[dict] = []
            context_parts: list[str] = []
            top_similarity = 0.0

            for item in search_results:
                chunk, similarity = item if isinstance(item, tuple) else (item, 0.0)
                filename = (
                    getattr(chunk.document, "filename", None)
                    if getattr(chunk, "document", None)
                    else None
                )
                retrieved_chunks.append(
                    {
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "filename": filename or "unknown",
                        "text": (
                            (chunk.text[:200] + "...")
                            if chunk.text and len(chunk.text) > 200
                            else (chunk.text or "")
                        ),
                        "similarity": round(float(similarity), 4),
                    }
                )
                if chunk.text:
                    context_parts.append(chunk.text)
                if similarity is not None and float(similarity) > top_similarity:
                    top_similarity = float(similarity)

            # Fire sources event immediately so the UI can show them
            yield _sse(
                {
                    "type": "sources",
                    "retrieved_chunks": retrieved_chunks,
                    "top_similarity": top_similarity,
                }
            )

            context = "\n\n---\n\n".join(context_parts)

            # --- Streaming generation --------------------------------------
            full_answer_parts: list[str] = []
            for token in generate_answer_stream(
                question=req.question,
                context=context,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            ):
                full_answer_parts.append(token)
                yield _sse({"type": "token", "content": token})

            full_answer = "".join(full_answer_parts)

            # --- Persist history -------------------------------------------
            retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]
            history = QueryHistory(
                user_id=req.user_id,
                question=req.question,
                answer=full_answer,
                retrieved_chunks_ids=json.dumps(retrieved_ids),
                top_similarity=top_similarity,
            )
            db.add(history)
            db.commit()
            db.refresh(history)

            yield _sse({"type": "done", "query_id": history.id})

        except Exception:
            print(traceback.format_exc())
            db.rollback()
            yield _sse({"type": "error", "detail": "Streaming generation failed"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if proxied
        },
    )


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

    chunks = (
        db.query(DocumentChunk).filter(
            DocumentChunk.id.in_(chunk_ids)
        ).all()
        if chunk_ids else []
    )

    return {
        "id": query.id,
        "question": query.question,
        "answer": query.answer,
        "created_at": query.created_at.isoformat(),
        "retrieved_chunks": [
            {
                "chunk_id": c.id,
                "filename": c.document.filename if getattr(c, "document", None) else "unknown",
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


# ---------------------------------------------------------------------------
# Follow-up suggestions
# ---------------------------------------------------------------------------

class SuggestRequest(BaseModel):
    question: str
    answer: str


@router.post("/suggest")
async def suggest_followups(req: SuggestRequest):
    """Return up to 3 follow-up question suggestions based on a Q&A exchange."""
    try:
        suggestions = generate_suggestions(req.question, req.answer)
        return {"suggestions": suggestions}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Answer feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    query_id: int
    user_id: str
    feedback: int  # 1 = thumbs up, -1 = thumbs down


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    """Record thumbs up / thumbs down feedback for an answer."""
    if req.feedback not in (1, -1):
        raise HTTPException(status_code=400, detail="feedback must be 1 or -1")

    query = db.query(QueryHistory).filter(
        QueryHistory.id == req.query_id,
        QueryHistory.user_id == req.user_id,
    ).first()

    if not query:
        raise HTTPException(status_code=404, detail="Query not found")

    query.feedback = req.feedback
    db.commit()
    return {"message": "Feedback recorded", "query_id": req.query_id, "feedback": req.feedback}