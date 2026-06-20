"""Query and chat API endpoints."""

from __future__ import annotations

import json
import logging
import traceback
from typing import Iterator, List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.connection import get_db
from db.models import DocumentChunk, QueryHistory, User
from services.generation import (
    generate_answer,
    generate_answer_stream,
    generate_suggestions,
    generate_search_autocomplete,
)
from services.retrieval_strategies import (
    DEFAULT_RETRIEVAL_STRATEGY,
    get_retrieval_strategy,
    list_retrieval_strategies,
)
from services.query_processing import determine_query_route

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/queries", tags=["queries"])

# ---------------------------------------------------------------------------
# Simple in-memory rate limiter (requests per minute per user)
# ---------------------------------------------------------------------------
import time
from collections import defaultdict

_rate_store: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT = 30   # max requests per minute per user_id


def _check_rate_limit(user_id: str) -> None:
    now = time.time()
    window = 60.0
    _rate_store[user_id] = [t for t in _rate_store[user_id] if now - t < window]
    if len(_rate_store[user_id]) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({_RATE_LIMIT} requests/minute). Please wait.",
        )
    _rate_store[user_id].append(now)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question:     str
    user_id:      str
    top_k:        int   = 8
    strategy:     str   = DEFAULT_RETRIEVAL_STRATEGY
    temperature:  float = 0.1
    max_tokens:   int   = 2048
    doc_mode:     bool  = True    # True = use retrieved context; False = general chat
    use_hyde:     bool  = False   # HyDE query expansion
    use_rerank:   bool  = True    # TF-IDF re-ranking
    chat_history: list[dict] | None = None  # multi-turn conversation context


class QueryResponse(BaseModel):
    question:         str
    answer:           str
    retrieved_chunks: List[dict]
    top_similarity:   float
    query_id:         int
    doc_mode:         bool
    strategy:         str
    token_usage:      dict | None = None


class HistoryResponse(BaseModel):
    id:          int
    question:    str
    answer:      str
    created_at:  str
    chunk_count: int
    feedback:    int | None
    starred:     bool
    doc_mode:    bool


class SuggestRequest(BaseModel):
    question: str
    answer:   str


class FeedbackRequest(BaseModel):
    query_id: int
    user_id:  str
    feedback: int   # 1 or -1


class StarRequest(BaseModel):
    query_id: int
    user_id:  str
    starred:  bool


# ---------------------------------------------------------------------------
# Shared retrieval + context builder
# ---------------------------------------------------------------------------

def compress_chunk_text(text: str, query: str) -> str:
    """
    Split the chunk text into sentences and keep only the sentences
    that contain any of the alphanumeric keywords from the query.
    """
    import re
    # Extract clean alphanumeric keywords from query
    query_words = set(re.findall(r"\w+", query.lower()))
    if not query_words:
        return text

    # Simple sentence splitter
    sentences = re.split(r"(?<=[.!?])\s+", text)
    matched_sentences = []
    for sent in sentences:
        sent_words = set(re.findall(r"\w+", sent.lower()))
        # If there's overlap in words, keep the sentence
        if query_words.intersection(sent_words):
            matched_sentences.append(sent)

    if not matched_sentences:
        # Fallback to first 2 sentences if no keywords match
        fallback = " ".join(sentences[:2])
        print(f"[DEBUG] No keywords matched. Fallback to start: {len(fallback)} chars.")
        return fallback
        
    compressed = " ".join(matched_sentences)
    print(f"[DEBUG] Compressed chunk text size from {len(text)} to {len(compressed)} chars.")
    return compressed


def _build_context_and_chunks(
    db: Session,
    req: QueryRequest,
) -> tuple[str, list[dict], float]:
    """
    Run retrieval and build:
      - context string (with source headers)
      - retrieved_chunks list for the response
      - top_similarity float
    """
    strategy_name = (req.strategy or DEFAULT_RETRIEVAL_STRATEGY).lower()
    try:
        strategy = get_retrieval_strategy(strategy_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if strategy_name == DEFAULT_RETRIEVAL_STRATEGY:
        # Preserve the legacy toggles for existing callers during Phase 1.
        strategy.use_hyde = req.use_hyde if hasattr(strategy, "use_hyde") else False
        strategy.use_rerank = req.use_rerank if hasattr(strategy, "use_rerank") else True

    retrieval_result = strategy.retrieve(
        db=db,
        query=req.question,
        top_k=req.top_k,
        user_id=req.user_id,
        chat_history=req.chat_history,
        doc_mode=req.doc_mode,
    )
    search_results = retrieval_result.chunks

    retrieved_chunks: list[dict] = []
    context_parts:    list[str]  = []
    top_similarity = 0.0
    total = len(search_results)

    for idx, item in enumerate(search_results):
        chunk, similarity = item if isinstance(item, tuple) else (item, 0.0)
        filename = (
            getattr(chunk.document, "filename", None)
            if getattr(chunk, "document", None) else None
        ) or "unknown"

        # Apply Sentence-level Keyword Compression (Step 2)
        compressed_text = compress_chunk_text(chunk.text, req.question)

        # Contextual header prepended to each chunk
        header = f"[Source: {filename}, chunk {chunk.chunk_index + 1}]"
        context_parts.append(f"{header}\n{compressed_text}")

        retrieved_chunks.append({
            "chunk_id":    chunk.id,
            "document_id": chunk.document_id,
            "filename":    filename,
            "text":        (compressed_text[:200] + "…") if len(compressed_text or "") > 200 else (compressed_text or ""),
            "similarity":  round(float(similarity), 4),
        })

        if float(similarity) > top_similarity:
            top_similarity = float(similarity)

    context = "\n\n---\n\n".join(context_parts)
    return context, retrieved_chunks, top_similarity


def _ensure_user(db: Session, user_id: str) -> None:
    if not db.query(User).filter(User.id == user_id).first():
        db.add(User(id=user_id))
        db.flush()


# ---------------------------------------------------------------------------
# /ask  (blocking)
# ---------------------------------------------------------------------------

@router.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest, db: Session = Depends(get_db)):
    """Blocking question endpoint."""
    try:
        _check_rate_limit(req.user_id)
        _ensure_user(db, req.user_id)

        # Check Query Router (Step 1)
        route = determine_query_route(req.question)
        if route == "GENERAL":
            print(f"[DEBUG] Routing bypassed RAG context for question: '{req.question[:50]}'")
            req.doc_mode = False

        context, retrieved_chunks, top_similarity = _build_context_and_chunks(db, req)

        print(f"[DEBUG] Context built. Size: {len(context)} chars. Chunks retrieved: {len(retrieved_chunks)}")

        answer, usage = generate_answer(
            question=req.question,
            context=context,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            chat_history=req.chat_history,
            doc_mode=req.doc_mode,
        )

        retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]
        history = QueryHistory(
            user_id=req.user_id,
            question=req.question,
            answer=answer,
            retrieved_chunks_ids=json.dumps(retrieved_ids),
            top_similarity=top_similarity,
            doc_mode=req.doc_mode,
        )
        db.add(history)
        db.commit()
        db.refresh(history)

        print(f"[DEBUG] Query stored successfully. Usage metrics: {usage}")

        return QueryResponse(
            question=req.question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            top_similarity=top_similarity,
            query_id=history.id,
            doc_mode=req.doc_mode,
            strategy=(req.strategy or DEFAULT_RETRIEVAL_STRATEGY).lower(),
            token_usage=usage,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# /ask-stream  (SSE)
# ---------------------------------------------------------------------------

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/ask-stream")
async def ask_question_stream(req: QueryRequest, db: Session = Depends(get_db)):
    """
    Streaming SSE endpoint.

    Event sequence:
      {"type":"sources",  "retrieved_chunks":[…], "top_similarity":float}
      {"type":"token",    "content":"…"}           — one per LLM token
      {"type":"done",     "query_id":int}
      {"type":"error",    "detail":"…"}             — on failure
    """
    try:
        _check_rate_limit(req.user_id)
    except HTTPException as e:
        async def rate_error():
            yield _sse({"type": "error", "detail": e.detail})
        return StreamingResponse(rate_error(), media_type="text/event-stream")

    def event_stream() -> Iterator[str]:
        try:
            _ensure_user(db, req.user_id)

            # Check Query Router (Step 1)
            route = determine_query_route(req.question)
            if route == "GENERAL":
                print(f"[DEBUG] Streaming routing bypassed RAG context for question: '{req.question[:50]}'")
                req.doc_mode = False

            context, retrieved_chunks, top_similarity = _build_context_and_chunks(db, req)

            print(f"[DEBUG] Streaming context built. Size: {len(context)} chars. Chunks: {len(retrieved_chunks)}")

            yield _sse({
                "type":             "sources",
                "retrieved_chunks": retrieved_chunks,
                "top_similarity":   top_similarity,
            })

            full_parts: list[str] = []
            for token in generate_answer_stream(
                question=req.question,
                context=context,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                chat_history=req.chat_history,
                doc_mode=req.doc_mode,
            ):
                full_parts.append(token)
                yield _sse({"type": "token", "content": token})

            full_answer = "".join(full_parts)

            retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]
            history = QueryHistory(
                user_id=req.user_id,
                question=req.question,
                answer=full_answer,
                retrieved_chunks_ids=json.dumps(retrieved_ids),
                top_similarity=top_similarity,
                doc_mode=req.doc_mode,
            )
            db.add(history)
            db.commit()
            db.refresh(history)

            # Estimate streaming response token usage (Step 3/4)
            from services.generation import _build_prompt
            messages = _build_prompt(req.question, context, req.chat_history, req.doc_mode)
            prompt_text = "".join(m["content"] for m in messages)
            prompt_tokens = len(prompt_text) // 4
            completion_tokens = len(full_answer) // 4
            total_tokens = prompt_tokens + completion_tokens
            
            usage_dict = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            print(f"[DEBUG] Streaming complete. Usage: {usage_dict}")

            yield _sse({
                "type": "done",
                "query_id": history.id,
                "token_usage": usage_dict
            })

        except Exception:
            logger.error(traceback.format_exc())
            db.rollback()
            yield _sse({"type": "error", "detail": "Streaming generation failed"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@router.get("/strategies")
async def get_strategies():
    return {
        "default": DEFAULT_RETRIEVAL_STRATEGY,
        "strategies": list_retrieval_strategies(),
    }


@router.get("/history")
async def get_query_history(
    user_id: str,
    limit:   int  = 50,
    starred: bool = False,
    db: Session = Depends(get_db),
) -> List[HistoryResponse]:
    q = db.query(QueryHistory).filter(QueryHistory.user_id == user_id)
    if starred:
        q = q.filter(QueryHistory.starred == True)
    queries = q.order_by(QueryHistory.created_at.desc()).limit(limit).all()

    return [
        HistoryResponse(
            id=q.id,
            question=q.question,
            answer=q.answer,
            created_at=q.created_at.isoformat(),
            chunk_count=len(json.loads(q.retrieved_chunks_ids or "[]")),
            feedback=q.feedback,
            starred=bool(q.starred),
            doc_mode=bool(q.doc_mode),
        )
        for q in queries
    ]


@router.get("/history/{query_id}")
async def get_query_detail(query_id: int, db: Session = Depends(get_db)):
    query = db.query(QueryHistory).filter(QueryHistory.id == query_id).first()
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")

    chunk_ids = json.loads(query.retrieved_chunks_ids or "[]")
    chunks = (
        db.query(DocumentChunk).filter(DocumentChunk.id.in_(chunk_ids)).all()
        if chunk_ids else []
    )

    return {
        "id":       query.id,
        "question": query.question,
        "answer":   query.answer,
        "created_at": query.created_at.isoformat(),
        "feedback": query.feedback,
        "starred":  query.starred,
        "doc_mode": query.doc_mode,
        "retrieved_chunks": [
            {
                "chunk_id":    c.id,
                "filename":    c.document.filename if getattr(c, "document", None) else "unknown",
                "text":        c.text,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ],
    }


@router.delete("/history/{query_id}")
async def delete_query(query_id: int, user_id: str, db: Session = Depends(get_db)):
    q = db.query(QueryHistory).filter(
        QueryHistory.id == query_id, QueryHistory.user_id == user_id
    ).first()
    if not q:
        raise HTTPException(status_code=404, detail="Query not found")
    db.delete(q)
    db.commit()
    return {"message": "Query deleted"}


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------

@router.post("/suggest")
async def suggest_followups(req: SuggestRequest):
    try:
        return {"suggestions": generate_suggestions(req.question, req.answer)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    if req.feedback not in (1, -1):
        raise HTTPException(status_code=400, detail="feedback must be 1 or -1")
    q = db.query(QueryHistory).filter(
        QueryHistory.id == req.query_id, QueryHistory.user_id == req.user_id
    ).first()
    if not q:
        raise HTTPException(status_code=404, detail="Query not found")
    q.feedback = req.feedback
    db.commit()
    return {"message": "Feedback recorded"}


# ---------------------------------------------------------------------------
# Star / pin
# ---------------------------------------------------------------------------

@router.post("/star")
async def star_query(req: StarRequest, db: Session = Depends(get_db)):
    q = db.query(QueryHistory).filter(
        QueryHistory.id == req.query_id, QueryHistory.user_id == req.user_id
    ).first()
    if not q:
        raise HTTPException(status_code=404, detail="Query not found")
    q.starred = req.starred
    db.commit()
    return {"message": "Star updated", "starred": req.starred}


# ---------------------------------------------------------------------------
# Autocomplete
# ---------------------------------------------------------------------------

@router.get("/autocomplete")
async def autocomplete(
    user_id: str,
    q:       str,
    db:      Session = Depends(get_db),
):
    """Return up to 5 past question completions matching partial input."""
    if len(q) < 2:
        return {"suggestions": []}
    past = [
        row.question
        for row in db.query(QueryHistory.question)
        .filter(QueryHistory.user_id == user_id)
        .order_by(QueryHistory.created_at.desc())
        .limit(200)
        .all()
    ]
    return {"suggestions": generate_search_autocomplete(q, past)}
