"""Document upload and management API endpoints."""

from __future__ import annotations

import asyncio
import io
import logging
import traceback

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Depends, HTTPException, Query
from pypdf import PdfReader
from sqlalchemy.orm import Session

from db.connection import get_db
from db.models import Document, DocumentChunk, User
from services.chunking import chunk_text, clean_text
from services.embeddings import embed_text
from services.faiss_store import faiss_store
from services.retrieval import search_document_text

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes, filename: str = "", username: str = "") -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = [page.extract_text() or "" for page in reader.pages]
    text = clean_text("\n\n".join(p for p in parts if p))

    # If very little text is found, it might be a scanned PDF. Fallback to OCR.
    if len(text.strip()) < 50:
        import shutil
        has_tesseract = shutil.which("tesseract") is not None
        has_poppler = shutil.which("pdftoppm") is not None

        try:
            import pytesseract
            from pdf2image import convert_from_bytes
        except ImportError:
            raise RuntimeError(
                "Scanned PDF detected, but OCR Python dependencies are missing. "
                "Please run: pip install pytesseract pdf2image"
            )

        if not has_tesseract:
            # Log failure
            from db.connection import get_db_context
            from db.models import SystemMetric
            with get_db_context() as db:
                db.add(SystemMetric(event_type="ocr_failure", username=username, details=f"Scanned PDF: {filename}, Error: Tesseract binary missing"))
            raise RuntimeError(
                "Scanned PDF detected, but Tesseract OCR engine is not installed on the system. "
                "Please run: sudo apt install tesseract-ocr"
            )
        if not has_poppler:
            # Log failure
            from db.connection import get_db_context
            from db.models import SystemMetric
            with get_db_context() as db:
                db.add(SystemMetric(event_type="ocr_failure", username=username, details=f"Scanned PDF: {filename}, Error: Poppler pdftoppm missing"))
            raise RuntimeError(
                "Scanned PDF detected, but Poppler utilities (pdftoppm) are not installed on the system. "
                "Please run: sudo apt install poppler-utils"
            )

        try:
            logger.info("Running OCR on scanned PDF...")
            images = convert_from_bytes(pdf_bytes)
            ocr_parts = [pytesseract.image_to_string(img) for img in images]
            text = clean_text("\n\n".join(p for p in ocr_parts if p))
            
            # Log success
            from db.connection import get_db_context
            from db.models import SystemMetric
            with get_db_context() as db:
                db.add(SystemMetric(event_type="ocr_success", username=username, details=f"Scanned PDF: {filename}"))
        except Exception as e:
            logger.warning(f"OCR failed for PDF: {e}")
            # Log failure
            from db.connection import get_db_context
            from db.models import SystemMetric
            with get_db_context() as db:
                db.add(SystemMetric(event_type="ocr_failure", username=username, details=f"Scanned PDF: {filename}, Error: {str(e)}"))
            raise RuntimeError(f"OCR processing failed: {e}")
            
    return text


def extract_text_from_image(img_bytes: bytes, filename: str = "", username: str = "") -> str:
    import shutil
    has_tesseract = shutil.which("tesseract") is not None

    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise RuntimeError("OCR Python dependencies are missing. Run: pip install pytesseract Pillow")

    if not has_tesseract:
        # Log failure
        from db.connection import get_db_context
        from db.models import SystemMetric
        with get_db_context() as db:
            db.add(SystemMetric(event_type="ocr_failure", username=username, details=f"Image: {filename}, Error: Tesseract binary missing"))
        raise RuntimeError(
            "Image uploaded, but Tesseract OCR engine is not installed on the system. "
            "Please run: sudo apt install tesseract-ocr"
        )

    try:
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img)
        cleaned = clean_text(text)
        
        # Log success
        from db.connection import get_db_context
        from db.models import SystemMetric
        with get_db_context() as db:
            db.add(SystemMetric(event_type="ocr_success", username=username, details=f"Image: {filename}"))
            
        return cleaned
    except Exception as e:
        # Log failure
        from db.connection import get_db_context
        from db.models import SystemMetric
        with get_db_context() as db:
            db.add(SystemMetric(event_type="ocr_failure", username=username, details=f"Image: {filename}, Error: {str(e)}"))
        raise RuntimeError(f"Image OCR failed: {e}")


def _generate_summary_bg(document_id: int, text: str, filename: str) -> None:
    """Background task: generate and store document summary."""
    from db.connection import get_db_context
    from services.generation import generate_document_summary
    try:
        summary = generate_document_summary(text, filename)
        if summary:
            with get_db_context() as db:
                doc = db.query(Document).filter(Document.id == document_id).first()
                if doc:
                    doc.summary = summary
                    logger.info(f"Summary stored for document {document_id}")
    except Exception:
        logger.warning(f"Summary generation failed for doc {document_id}", exc_info=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Query(..., description="User ID"),
    collection: str = Query(None, description="Optional collection / tag name"),
    db: Session = Depends(get_db),
):
    """
    Upload and embed a document.

    Steps:
      1. Extract text (PDF / txt / md)
      2. Chunk text
      3. Embed each chunk via Ollama
      4. Store document + chunks in DB
      5. Enqueue background summary generation
    """
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            user = User(id=user_id)
            db.add(user)
            db.flush()

        content = await file.read()
        filename_lower = (file.filename or "").lower()

        if filename_lower.endswith(".pdf") or file.content_type == "application/pdf":
            text = extract_text_from_pdf(content, filename=file.filename, username=user_id)
        elif filename_lower.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")) or (file.content_type and file.content_type.startswith("image/")):
            text = extract_text_from_image(content, filename=file.filename, username=user_id)
        else:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = content.decode("latin-1", errors="ignore")
            text = clean_text(text)

        if not text:
            raise HTTPException(status_code=400, detail="No extractable text found")

        doc = Document(
            user_id=user_id,
            filename=file.filename,
            original_content=text,
            collection_tag=collection,
        )
        db.add(doc)
        db.flush()

        chunks = chunk_text(text)
        logger.info(f"Uploading {file.filename}: {len(chunks)} chunks")

        db_chunks = []
        for idx, chunk_str in enumerate(chunks):
            try:
                embedding = embed_text(chunk_str)
                db_chunk = DocumentChunk(
                    document_id=doc.id,
                    text=chunk_str,
                    embedding=embedding,
                    chunk_index=idx,
                )
                db.add(db_chunk)
                db_chunks.append(db_chunk)
            except Exception as e:
                # Log embedding failure
                from db.models import SystemMetric
                db.add(SystemMetric(
                    event_type="embed_failure",
                    username=user_id,
                    details=f"File: {file.filename}, Error: {str(e)}"
                ))
                db.commit()
                raise RuntimeError(f"Embedding generation failed: {e}")

        # Log embedding success
        from db.models import SystemMetric
        db.add(SystemMetric(
            event_type="embed_success",
            username=user_id,
            details=f"File: {file.filename}, Chunks: {len(db_chunks)}"
        ))

        db.commit()
        db.refresh(doc)

        # Sync to FAISS
        faiss_store.add(
            chunk_ids=[c.id for c in db_chunks],
            embeddings=[c.embedding for c in db_chunks],
            user_ids_map={c.id: user_id for c in db_chunks}
        )

        # Generate summary asynchronously via Celery — doesn't block the upload response
        try:
            from tasks import generate_document_summary_task
            generate_document_summary_task.delay(doc.id, text, file.filename)
        except Exception as e:
            logger.warning(f"Failed to queue Celery summary task: {e}, falling back to background task")
            background_tasks.add_task(
                _generate_summary_bg, doc.id, text, file.filename
            )

        return {
            "document_id":    doc.id,
            "filename":       file.filename,
            "chunks_created": len(chunks),
            "collection":     collection,
            "message":        "Document uploaded successfully. Summary generating in background.",
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(traceback.format_exc())
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_documents(
    user_id: str = Query(...),
    collection: str = Query(None, description="Filter by collection tag"),
    db: Session = Depends(get_db),
):
    """List all documents for a user, optionally filtered by collection."""
    q = db.query(Document).filter(Document.user_id == user_id)
    if collection:
        q = q.filter(Document.collection_tag == collection)
    docs = q.all()

    return [
        {
            "id":             doc.id,
            "filename":       doc.filename,
            "uploaded_at":    doc.uploaded_at.isoformat(),
            "chunks":         len(doc.chunks),
            "summary":        doc.summary or "",
            "collection_tag": doc.collection_tag or "",
        }
        for doc in docs
    ]


@router.get("/collections")
async def list_collections(
    user_id: str = Query(...),
    db: Session = Depends(get_db),
):
    """Return all distinct collection tags for a user."""
    docs = db.query(Document).filter(Document.user_id == user_id).all()
    tags = sorted({d.collection_tag for d in docs if d.collection_tag})
    return {"collections": tags}


@router.patch("/{document_id}/collection")
async def update_collection(
    document_id: int,
    collection: str = Query(...),
    user_id: str = Query(...),
    db: Session = Depends(get_db),
):
    """Update the collection tag on a document."""
    doc = db.query(Document).filter(
        Document.id == document_id, Document.user_id == user_id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    doc.collection_tag = collection
    db.commit()
    return {"message": "Collection updated", "collection": collection}


@router.get("/{document_id}/search")
async def search_document(
    document_id: int,
    q: str = Query(..., description="Keyword to search"),
    user_id: str = Query(...),
    db: Session = Depends(get_db),
):
    """Keyword search within a single document's chunks."""
    doc = db.query(Document).filter(
        Document.id == document_id, Document.user_id == user_id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    hits = search_document_text(db, document_id, q)
    return {
        "query":   q,
        "results": [
            {"chunk_id": c.id, "chunk_index": c.chunk_index, "text": c.text}
            for c in hits
        ],
    }


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    user_id: str = Query(...),
    db: Session = Depends(get_db),
):
    doc = db.query(Document).filter(
        Document.id == document_id, Document.user_id == user_id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    chunk_ids = [c.id for c in db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).all()]
    faiss_store.remove_document_chunks(chunk_ids)
    
    db.delete(doc)
    db.commit()
    return {"message": "Document deleted successfully"}
