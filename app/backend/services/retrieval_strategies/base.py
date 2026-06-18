"""Shared retrieval strategy contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy.orm import Session

from db.models import DocumentChunk


@dataclass
class RetrievalResult:
    """Result returned by every retrieval strategy."""

    chunks: list[tuple[DocumentChunk, float]]
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalStrategy(Protocol):
    """Protocol implemented by named retrieval pipelines."""

    name: str

    def retrieve(
        self,
        db: Session,
        query: str,
        top_k: int = 8,
        user_id: str | None = None,
        chat_history: list[dict] | None = None,
        doc_mode: bool = True,
    ) -> RetrievalResult:
        ...
