"""Query and chat API endpoints."""

from __future__ import annotations

import json
import logging
import traceback
from typing import Iterator, List

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.connection import get_db
from db.models import DocumentChunk, QueryHistory, User, UserAccount
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
    retrieved_chunks: List[dict] | None = None
    top_similarity:   float | None = None
    chunk_evaluations: dict | None = None


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
def run_background_chunk_eval(history_id: int, question: str, chunks: list[dict]):
    from db.connection import SessionLocal
    from db.models import QueryHistory
    from services.generation import evaluate_chunks_relevance, evaluate_rag_triad
    import json
    
    # Rebuild context text for RAG Triad
    context_parts = []
    for c in chunks:
        context_parts.append(f"[Source: {c.get('filename', 'unknown')}]\n{c.get('text', '')}")
    context_text = "\n\n---\n\n".join(context_parts)
    
    db = SessionLocal()
    try:
        q = db.query(QueryHistory).filter(QueryHistory.id == history_id).first()
        if q:
            chunk_evals = evaluate_chunks_relevance(question, chunks)
            triad_evals = evaluate_rag_triad(question, q.answer, context_text)
            combined_evals = {
                "chunks": chunk_evals,
                "triad": triad_evals
            }
            q.chunk_evaluations = json.dumps(combined_evals)
            db.commit()
            print(f"[DEBUG] Combined evaluations stored for query ID {history_id}")
    except Exception as e:
        db.rollback()
        print(f"[DEBUG] Failed to save background evaluations: {e}")
    finally:
        db.close()


# /ask  (blocking)
# ---------------------------------------------------------------------------

@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Blocking question endpoint."""
    start_time = time.time()
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

        custom_ins = None
        username = req.user_id.split("_")[0] if "_" in req.user_id else req.user_id
        if username:
            user_acc = db.query(UserAccount).filter(UserAccount.username == username).first()
            if user_acc:
                custom_ins = getattr(user_acc, "custom_instructions", None)

        answer, usage = generate_answer(
            question=req.question,
            context=context,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            chat_history=req.chat_history,
            doc_mode=req.doc_mode,
            custom_instructions=custom_ins,
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

        # Enqueue chunk relevance and RAG Triad evaluation via Celery
        try:
            from tasks import evaluate_query_triad
            evaluate_query_triad.delay(history.id)
        except Exception as e:
            print(f"[DEBUG] Failed to queue Celery evaluation: {e}, falling back to background task")
            background_tasks.add_task(run_background_chunk_eval, history.id, req.question, retrieved_chunks)

        print(f"[DEBUG] Query stored successfully. Usage metrics: {usage}")

        # Record query metrics in telemetry
        from db.models import SystemMetric
        latency = time.time() - start_time
        total_tokens = usage.get("total_tokens", 0) if usage else 0
        base_username = req.user_id.split("_")[0] if "_" in req.user_id else req.user_id
        db.add(SystemMetric(
            event_type="query",
            username=base_username,
            tokens=total_tokens,
            latency=latency,
            details=req.question[:200]
        ))
        db.commit()

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
async def ask_question_stream(
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
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
        start_time = time.time()
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

            custom_ins = None
            username = req.user_id.split("_")[0] if "_" in req.user_id else req.user_id
            if username:
                user_acc = db.query(UserAccount).filter(UserAccount.username == username).first()
                if user_acc:
                    custom_ins = getattr(user_acc, "custom_instructions", None)

            full_parts: list[str] = []
            for token in generate_answer_stream(
                question=req.question,
                context=context,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                chat_history=req.chat_history,
                doc_mode=req.doc_mode,
                custom_instructions=custom_ins,
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

            # Enqueue chunk relevance and RAG Triad evaluation via Celery
            try:
                from tasks import evaluate_query_triad
                evaluate_query_triad.delay(history.id)
            except Exception as e:
                print(f"[DEBUG] Failed to queue Celery evaluation: {e}, falling back to background task")
                background_tasks.add_task(run_background_chunk_eval, history.id, req.question, retrieved_chunks)

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

            # Record query metrics in telemetry
            from db.models import SystemMetric
            latency = time.time() - start_time
            base_username = req.user_id.split("_")[0] if "_" in req.user_id else req.user_id
            db.add(SystemMetric(
                event_type="query",
                username=base_username,
                tokens=total_tokens,
                latency=latency,
                details=req.question[:200]
            ))
            db.commit()

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

    # Gather all unique chunk IDs
    all_chunk_ids = set()
    query_chunk_ids_map = {}
    for q_hist in queries:
        try:
            cids = json.loads(q_hist.retrieved_chunks_ids or "[]")
        except Exception:
            cids = []
        query_chunk_ids_map[q_hist.id] = cids
        all_chunk_ids.update(cids)
        
    # Fetch all chunks in one query
    chunks_map = {}
    if all_chunk_ids:
        chunks = db.query(DocumentChunk).filter(DocumentChunk.id.in_(list(all_chunk_ids))).all()
        for c in chunks:
            chunks_map[c.id] = {
                "chunk_id":    c.id,
                "filename":    c.document.filename if getattr(c, "document", None) else "unknown",
                "text":        c.text,
                "chunk_index": c.chunk_index,
            }

    results = []
    for q_hist in queries:
        cids = query_chunk_ids_map.get(q_hist.id, [])
        ret_chunks = []
        for cid in cids:
            if cid in chunks_map:
                chunk_info = dict(chunks_map[cid])
                chunk_info["similarity"] = q_hist.top_similarity or 0.0
                ret_chunks.append(chunk_info)
        results.append(
            HistoryResponse(
                id=q_hist.id,
                question=q_hist.question,
                answer=q_hist.answer,
                created_at=q_hist.created_at.isoformat(),
                chunk_count=len(cids),
                feedback=q_hist.feedback,
                starred=bool(q_hist.starred),
                doc_mode=bool(q_hist.doc_mode),
                retrieved_chunks=ret_chunks,
                top_similarity=q_hist.top_similarity,
                chunk_evaluations=json.loads(q_hist.chunk_evaluations or "{}"),
            )
        )
    return results


@router.get("/users")
async def get_users(db: Session = Depends(get_db)):
    """Get list of all user sessions."""
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [{"id": u.id, "created_at": u.created_at.isoformat()} for u in users]


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
        "chunk_evaluations": json.loads(query.chunk_evaluations or "{}"),
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


# ---------------------------------------------------------------------------
# Auth & Telemetry / Admin Metrics
# ---------------------------------------------------------------------------
from datetime import datetime, timedelta

class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/auth/login")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate UserAccount and return user info."""
    import hashlib
    user = db.query(UserAccount).filter(UserAccount.username == req.username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
        
    pwd_hash = hashlib.sha256(req.password.encode()).hexdigest()
    if pwd_hash != user.password_hash:
        raise HTTPException(status_code=401, detail="Invalid username or password")
        
    return {
        "status": "ok",
        "username": user.username,
        "role": user.role
    }


@router.post("/auth/register")
async def register(req: LoginRequest, db: Session = Depends(get_db)):
    """Register a new user account (defaults to user role)."""
    import hashlib
    existing = db.query(UserAccount).filter(UserAccount.username == req.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
        
    pwd_hash = hashlib.sha256(req.password.encode()).hexdigest()
    new_user = UserAccount(username=req.username, password_hash=pwd_hash, role="user")
    db.add(new_user)
    db.commit()
    return {"status": "ok", "message": "User registered successfully"}


@router.get("/auth/user-role/{username}")
async def get_user_role(username: str, db: Session = Depends(get_db)):
    """Fetch user account role based on username (helper for UI autologin)."""
    user = db.query(UserAccount).filter(UserAccount.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User account not found")
    return {
        "username": user.username,
        "role":     user.role
    }


@router.get("/admin/metrics")
async def get_admin_metrics(db: Session = Depends(get_db)):
    """Fetch administrative analytics (TPM, RPM, OCR metrics)."""
    from db.models import SystemMetric
    
    one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
    
    # 1. RPM (Queries in last minute)
    rpm = db.query(SystemMetric).filter(
        SystemMetric.event_type == "query",
        SystemMetric.created_at >= one_minute_ago
    ).count()
    
    # 2. TPM (Sum of tokens in last minute)
    tpm_query = db.query(SystemMetric.tokens).filter(
        SystemMetric.event_type == "query",
        SystemMetric.created_at >= one_minute_ago
    ).all()
    tpm = sum(t[0] for t in tpm_query if t[0] is not None)
    
    # 3. OCR success / failure count
    ocr_success = db.query(SystemMetric).filter(SystemMetric.event_type == "ocr_success").count()
    ocr_failure = db.query(SystemMetric).filter(SystemMetric.event_type == "ocr_failure").count()
    
    # 4. Embedding success / failure count
    embed_success = db.query(SystemMetric).filter(SystemMetric.event_type == "embed_success").count()
    embed_failure = db.query(SystemMetric).filter(SystemMetric.event_type == "embed_failure").count()
    
    # 5. Retrieve last 100 log items
    logs = db.query(SystemMetric).order_by(SystemMetric.created_at.desc()).limit(100).all()
    log_list = [
        {
            "id": l.id,
            "event_type": l.event_type,
            "username": l.username,
            "tokens": l.tokens,
            "latency": round(l.latency, 3) if l.latency else 0.0,
            "details": l.details,
            "created_at": l.created_at.isoformat()
        }
        for l in logs
    ]
    
    return {
        "rpm": rpm,
        "tpm": tpm,
        "ocr_success": ocr_success,
        "ocr_failure": ocr_failure,
        "embed_success": embed_success,
        "embed_failure": embed_failure,
        "logs": log_list
    }


@router.post("/admin/clear-history")
async def clear_history(db: Session = Depends(get_db)):
    """Clear all chat history and telemetry logs from the database."""
    try:
        db.query(QueryHistory).delete()
        from db.models import SystemMetric
        db.query(SystemMetric).delete()
        db.commit()
        return {"status": "ok", "message": "All chat history and system metrics have been cleared successfully."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database purge failed: {str(e)}")


@router.delete("/history/clear-session/{session_id}")
async def clear_session_history(session_id: str, db: Session = Depends(get_db)):
    """Clear all chat history for a specific session."""
    try:
        db.query(QueryHistory).filter(QueryHistory.user_id == session_id).delete()
        db.commit()
        return {"status": "ok", "message": f"History for session '{session_id}' has been cleared successfully."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/users-report")
async def get_users_report(db: Session = Depends(get_db)):
    """Fetch detailed statistics for all user accounts."""
    try:
        from db.models import UserAccount, User, QueryHistory, Document, SystemMetric
        accounts = db.query(UserAccount).order_by(UserAccount.created_at.desc()).all()
        
        report = []
        for acc in accounts:
            username = acc.username
            sessions = db.query(User).filter(User.id.like(f"{username}_%")).all()
            session_ids = [s.id for s in sessions]
            session_ids.append(username)
            
            queries_count = db.query(QueryHistory).filter(QueryHistory.user_id.in_(session_ids)).count()
            starred_count = db.query(QueryHistory).filter(
                QueryHistory.user_id.in_(session_ids),
                QueryHistory.starred == True
            ).count()
            docs_count = db.query(Document).filter(Document.user_id.in_(session_ids)).count()
            
            pos_feedback = db.query(QueryHistory).filter(
                QueryHistory.user_id.in_(session_ids),
                QueryHistory.feedback == 1
            ).count()
            neg_feedback = db.query(QueryHistory).filter(
                QueryHistory.user_id.in_(session_ids),
                QueryHistory.feedback == -1
            ).count()
            
            metrics = db.query(SystemMetric).filter(SystemMetric.username == username).all()
            total_tokens = sum(m.tokens for m in metrics if m.tokens is not None)
            avg_latency = 0.0
            latencies = [m.latency for m in metrics if m.latency is not None and m.latency > 0]
            if latencies:
                avg_latency = round(sum(latencies) / len(latencies), 2)
                
            # Calculate user-specific RPM & TPM
            one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
            rpm = sum(1 for m in metrics if m.event_type == "query" and m.created_at >= one_minute_ago)
            tpm = sum(m.tokens for m in metrics if m.event_type == "query" and m.tokens is not None and m.created_at >= one_minute_ago)
            
            # Retrieve token history for the last 50 queries
            user_queries = [m for m in metrics if m.event_type == "query"]
            user_queries.sort(key=lambda x: x.created_at)
            token_history = [
                {
                    "created_at": q.created_at.isoformat(),
                    "tokens": q.tokens or 0
                }
                for q in user_queries[-50:]
            ]
                
            report.append({
                "username": acc.username,
                "role": acc.role,
                "created_at": acc.created_at.isoformat(),
                "sessions_count": len(sessions),
                "queries_count": queries_count,
                "starred_count": starred_count,
                "documents_count": docs_count,
                "feedback_positive": pos_feedback,
                "feedback_negative": neg_feedback,
                "total_tokens": total_tokens,
                "avg_latency": avg_latency,
                "rpm": rpm,
                "tpm": tpm,
                "token_history": token_history
            })
            
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.delete("/admin/users/{username}")
async def delete_user_account(username: str, db: Session = Depends(get_db)):
    """Delete a user account and purge all their sessions, documents, and query history."""
    try:
        account = db.query(UserAccount).filter(UserAccount.username == username).first()
        if not account:
            raise HTTPException(status_code=404, detail="User account not found")
        if username == "admin":
            raise HTTPException(status_code=400, detail="Cannot delete the root admin account")

        from db.models import SystemMetric, User, Document, DocumentChunk
        db.query(SystemMetric).filter(SystemMetric.username == username).delete()

        user_sessions = db.query(User).filter(User.id.like(f"{username}_%")).all()
        session_ids = [s.id for s in user_sessions]
        session_ids.append(username)

        db.query(QueryHistory).filter(QueryHistory.user_id.in_(session_ids)).delete(synchronize_session=False)

        user_docs = db.query(Document).filter(Document.user_id.in_(session_ids)).all()
        doc_ids = [d.id for d in user_docs]
        if doc_ids:
            db.query(DocumentChunk).filter(DocumentChunk.document_id.in_(doc_ids)).delete(synchronize_session=False)
            db.query(Document).filter(Document.id.in_(doc_ids)).delete(synchronize_session=False)

        db.query(User).filter(User.id.in_(session_ids)).delete(synchronize_session=False)
        db.delete(account)
        db.commit()
        return {"status": "ok", "message": f"User '{username}' and all associated history/files have been purged successfully."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(e)}")


class ChangeRoleRequest(BaseModel):
    username: str
    role: str


@router.post("/admin/users/change-role")
async def change_user_role(req: ChangeRoleRequest, db: Session = Depends(get_db)):
    """Promote or demote a user account."""
    if req.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="Role must be 'admin' or 'user'")
    if req.username == "admin":
        raise HTTPException(status_code=400, detail="Cannot change the role of the root admin account")
        
    user = db.query(UserAccount).filter(UserAccount.username == req.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User account not found")
        
    user.role = req.role
    db.commit()
    return {"status": "ok", "message": f"User '{req.username}' role updated to '{req.role}'."}


class CustomInstructionsRequest(BaseModel):
    username: str
    custom_instructions: str


@router.get("/auth/custom-instructions/{username}")
async def get_custom_instructions(username: str, db: Session = Depends(get_db)):
    """Fetch custom instructions for a specific user account."""
    user = db.query(UserAccount).filter(UserAccount.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User account not found")
    return {"status": "ok", "custom_instructions": user.custom_instructions or ""}


@router.post("/auth/custom-instructions")
async def save_custom_instructions(req: CustomInstructionsRequest, db: Session = Depends(get_db)):
    """Save or update custom instructions for a user account."""
    user = db.query(UserAccount).filter(UserAccount.username == req.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User account not found")
    user.custom_instructions = req.custom_instructions
    db.commit()
    return {"status": "ok", "message": "Custom instructions updated successfully."}
