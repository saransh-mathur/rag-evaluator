"""Document upload and management API endpoints."""

import io
import os
import traceback

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from pypdf import PdfReader
from sqlalchemy.orm import Session

from db.connection import get_db
from db.models import Document, DocumentChunk, User
from services.chunking import chunk_text, clean_text
from services.embeddings import embed_text

router = APIRouter(prefix="/api/documents", tags=["documents"])


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text:
            parts.append(page_text)
    return clean_text("\n\n".join(parts))


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Query(..., description="User ID"),
    db: Session = Depends(get_db)
):
    try:
        print("upload_document called")
        print("user_id:", user_id)
        print("filename:", file.filename)
        print("content_type:", file.content_type)

        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")

        user = db.query(User).filter(User.id == user_id).first()
        print("user exists:", bool(user))
        if not user:
            user = User(id=user_id)
            db.add(user)
            db.flush()
            print("created user")

        content = await file.read()
        print("file bytes:", len(content))

        filename_lower = (file.filename or "").lower()
        if filename_lower.endswith(".pdf") or file.content_type == "application/pdf":
            print("detected pdf, extracting text")
            text = extract_text_from_pdf(content)
            print("pdf text extracted")
        else:
            try:
                text = content.decode("utf-8")
                print("decoded utf-8")
            except UnicodeDecodeError:
                text = content.decode("latin-1", errors="ignore")
                print("decoded latin-1 with ignore")

            text = clean_text(text)

        print("text length:", len(text))

        if not text:
            raise HTTPException(status_code=400, detail="No extractable text found in uploaded file")

        doc = Document(
            user_id=user_id,
            filename=file.filename,
            original_content=text
        )
        db.add(doc)
        db.flush()
        print("document id:", doc.id)

        chunks = chunk_text(text)
        print("chunk count:", len(chunks))

        for idx, chunk_text_str in enumerate(chunks):
            print("processing chunk:", idx, "len:", len(chunk_text_str))
            embedding = embed_text(chunk_text_str)
            print("embedding type:", type(embedding), "len:", len(embedding))

            chunk = DocumentChunk(
                document_id=doc.id,
                text=chunk_text_str,
                embedding=embedding,
                chunk_index=idx
            )
            db.add(chunk)

        db.commit()
        db.refresh(doc)
        print("commit successful")

        return {
            "document_id": doc.id,
            "filename": file.filename,
            "chunks_created": len(chunks),
            "message": "Document uploaded successfully"
        }

    except Exception as e:
        print(traceback.format_exc())
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_documents(
    user_id: str = None,
    db: Session = Depends(get_db)
):
    """List all documents for a user."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    docs = db.query(Document).filter(
        Document.user_id == user_id
    ).all()

    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "uploaded_at": doc.uploaded_at.isoformat(),
            "chunks": len(doc.chunks)
        }
        for doc in docs
    ]


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    user_id: str = None,
    db: Session = Depends(get_db)
):
    """Delete a document and its chunks."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == user_id
    ).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    db.delete(doc)
    db.commit()

    return {"message": "Document deleted successfully"}