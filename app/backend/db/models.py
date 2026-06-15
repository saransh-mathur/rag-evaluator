"""SQLAlchemy models for RAG AI application."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class User(Base):
    """User sessions/authentication."""
    __tablename__ = "users"

    id         = Column(String(255), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="user")
    queries   = relationship("QueryHistory", back_populates="user")


class Document(Base):
    """Uploaded documents."""
    __tablename__ = "documents"

    id               = Column(Integer, primary_key=True)
    user_id          = Column(String(255), ForeignKey("users.id"))
    filename         = Column(String(255), nullable=False)
    original_content = Column(Text, nullable=False)
    summary          = Column(Text, nullable=True)          # AI-generated summary
    collection_tag   = Column(String(100), nullable=True)   # user-defined collection
    uploaded_at      = Column(DateTime, default=datetime.utcnow)

    user   = relationship("User", back_populates="documents")
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    """Text chunks from documents with embeddings."""
    __tablename__ = "document_chunks"

    id          = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    text        = Column(Text, nullable=False)
    embedding   = Column(Vector(768), nullable=False)
    chunk_index = Column(Integer)

    document = relationship("Document", back_populates="chunks")


class QueryHistory(Base):
    """Chat history and QA results."""
    __tablename__ = "query_history"

    id                   = Column(Integer, primary_key=True)
    user_id              = Column(String(255), ForeignKey("users.id"))
    question             = Column(Text, nullable=False)
    answer               = Column(Text, nullable=False)
    retrieved_chunks_ids = Column(Text)    # JSON list of chunk IDs
    top_similarity       = Column(Float)
    feedback             = Column(Integer, nullable=True)   # 1=👍  -1=👎
    starred              = Column(Boolean, default=False)   # pinned/starred
    doc_mode             = Column(Boolean, default=True)    # document vs general mode
    created_at           = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="queries")
