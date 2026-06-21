import os
import sys
import json
import logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from celery_app import celery_app
from db.connection import SessionLocal
from db.models import QueryHistory, Document, DocumentChunk

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.evaluate_query_triad")
def evaluate_query_triad(history_id: int):
    """
    Celery task to run RAG Triad (Faithfulness, Answer Relevance, Context Recall)
    evaluations on a completed user query.
    """
    logger.info(f"[Celery] Starting RAG Triad evaluation for query history ID {history_id}")
    db = SessionLocal()
    try:
        q_hist = db.query(QueryHistory).filter(QueryHistory.id == history_id).first()
        if not q_hist:
            logger.error(f"[Celery] QueryHistory ID {history_id} not found.")
            return f"Error: QueryHistory ID {history_id} not found"

        # Load retrieved chunks text
        try:
            chunk_ids = json.loads(q_hist.retrieved_chunks_ids or "[]")
        except Exception:
            chunk_ids = []

        context_text = ""
        if chunk_ids:
            chunks = db.query(DocumentChunk).filter(DocumentChunk.id.in_(chunk_ids)).all()
            context_parts = []
            for c in chunks:
                filename = c.document.filename if getattr(c, "document", None) else "unknown"
                context_parts.append(f"[Source: {filename}, chunk {c.chunk_index + 1}]\n{c.text}")
            context_text = "\n\n---\n\n".join(context_parts)

        # Run combined evaluations: chunk-level and RAG Triad
        from services.generation import evaluate_chunks_relevance, evaluate_rag_triad
        
        # Prepare chunk dicts for evaluate_chunks_relevance
        chunks_input = []
        if chunk_ids:
            chunks = db.query(DocumentChunk).filter(DocumentChunk.id.in_(chunk_ids)).all()
            for c in chunks:
                chunks_input.append({
                    "chunk_id": c.id,
                    "text": c.text,
                    "filename": c.document.filename if getattr(c, "document", None) else "unknown"
                })
        
        chunk_evals = evaluate_chunks_relevance(q_hist.question, chunks_input)
        triad_evals = evaluate_rag_triad(q_hist.question, q_hist.answer, context_text)
        
        combined_evals = {
            "chunks": chunk_evals,
            "triad": triad_evals
        }

        # Store JSON evaluations back in the database
        q_hist.chunk_evaluations = json.dumps(combined_evals)
        db.commit()
        logger.info(f"[Celery] Successfully saved combined RAG evaluations for query ID {history_id}")
        return f"Evaluated query ID {history_id} successfully"

    except Exception as e:
        db.rollback()
        logger.error(f"[Celery] Task evaluate_query_triad failed: {e}", exc_info=True)
        raise e
    finally:
        db.close()


@celery_app.task(name="tasks.generate_document_summary")
def generate_document_summary_task(document_id: int, text: str, filename: str):
    """
    Celery task to generate and store document summaries asynchronously.
    """
    logger.info(f"[Celery] Starting summarization for document ID {document_id} ({filename})")
    db = SessionLocal()
    try:
        from services.generation import generate_document_summary
        summary = generate_document_summary(text, filename)
        if summary:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.summary = summary
                db.commit()
                logger.info(f"[Celery] Summary saved successfully for document ID {document_id}")
                return f"Summarized document {document_id}"
        return "No summary generated"
    except Exception as e:
        db.rollback()
        logger.error(f"[Celery] Task generate_document_summary failed: {e}", exc_info=True)
        raise e
    finally:
        db.close()
