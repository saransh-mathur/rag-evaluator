"""Document upload and management API endpoints."""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import io
from db.connection import get_db
from db.models import Document, DocumentChunk, User
from services.embeddings import embed_text

router = APIRouter(prefix="/api/documents", tags=["documents"])


def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 120) -> List[str]:
    """Split text into overlapping chunks."""
    text = text.strip()
    if not text:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    
    return [c for c in chunks if c]


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = None,
    db: Session = Depends(get_db)
):
    """
    Upload and process a document.
    
    - Reads file content
    - Chunks the text
    - Embeds each chunk
    - Stores in PostgreSQL
    """
    try:
        # Ensure user exists
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            user = User(id=user_id)
            db.add(user)
            db.flush()
        
        # Read file
        content = await file.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
        
        # Create document record
        doc = Document(
            user_id=user_id,
            filename=file.filename,
            original_content=text
        )
        db.add(doc)
        db.flush()
        
        # Chunk and embed
        chunks = chunk_text(text)
        for idx, chunk_text_str in enumerate(chunks):
            embedding = embed_text(chunk_text_str)
            chunk = DocumentChunk(
                document_id=doc.id,
                text=chunk_text_str,
                embedding=embedding,
                chunk_index=idx
            )
            db.add(chunk)
        
        db.commit()
        
        return {
            "document_id": doc.id,
            "filename": file.filename,
            "chunks_created": len(chunks),
            "message": "Document uploaded successfully"
        }
    
    except Exception as e:
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
