# RAG Chunking Strategies

Retrieval-augmented generation quality depends heavily on how source documents are split into chunks before embedding.

## Fixed-size chunking

Split text every N characters or tokens with optional **chunk overlap** between consecutive segments. Overlap (for example 10–20% of chunk size) preserves sentences or bullet lists that would otherwise be cut at boundaries.

Typical starting points:

- Chunk size: 500–1000 characters for markdown notes
- Chunk overlap: 100–150 characters

## Semantic chunking

**Semantic chunking** splits on document structure—headings, paragraphs, code blocks—rather than arbitrary byte limits. For technical docs this keeps related concepts together and often improves retrieval precision compared to naive fixed-size splitting.

## Metadata enrichment

Store source file, heading path, and chunk index with each vector. At query time, showing provenance helps debugging and user trust.

## Evaluation loop

Measure retrieval hit rate and answer correctness when tuning chunk size and overlap. Weak performance on specific question types often signals chunks that are too large (noise) or too small (missing context).
