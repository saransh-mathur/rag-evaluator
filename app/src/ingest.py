"""Read documents from disk and prepare chunked objects for embedding."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Make app/shared importable when running from app/src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.chunking import chunk_text  # noqa: F401 — re-exported for callers


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    text: str
    chunk_index: int


def read_markdown_files(docs_dir: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for path in sorted(docs_dir.glob("*.md")):
        docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def build_chunks(
    docs_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for filename, content in read_markdown_files(docs_dir):
        for idx, piece in enumerate(chunk_text(content, chunk_size, chunk_overlap)):
            chunk_id = f"{Path(filename).stem}__{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_file=filename,
                    text=piece,
                    chunk_index=idx,
                )
            )
    return chunks
