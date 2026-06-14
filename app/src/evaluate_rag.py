"""Embed chunks, retrieve context, generate answers, and score RAG quality."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import requests
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from ingest import Chunk, build_chunks
from utils import (
    answer_hit,
    hallucination_flag,
    read_json,
    retrieval_hit,
    top_k_similar,
    write_csv,
    write_json,
)


def embed_texts(texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    url = f"{config.EMBED_BASE_URL.rstrip('/')}/api/embeddings"
    for text in texts:
        resp = requests.post(
            url,
            json={"model": config.EMBED_MODEL, "prompt": text},
            timeout=120,
        )
        resp.raise_for_status()
        vectors.append(resp.json()["embedding"])
    return np.array(vectors, dtype=np.float32)


def generate_answer(question: str, context: str) -> str:
    client = OpenAI(base_url=config.GEN_BASE_URL, api_key=config.GEN_API_KEY)
    prompt = (
        "You are a helpful technical assistant. Answer using ONLY the provided context. "
        "If the context does not contain enough information, say you do not know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    response = client.chat.completions.create(
        model=config.GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def run_evaluation(
    docs_dir: Path,
    tests_file: Path,
    output_dir: Path,
    top_k: int,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    test_cases = read_json(tests_file)
    chunks: list[Chunk] = build_chunks(docs_dir, chunk_size, chunk_overlap)

    if not chunks:
        raise RuntimeError(f"No chunks found in {docs_dir}")

    chunk_texts = [c.text for c in chunks]
    chunk_embeddings = embed_texts(chunk_texts)

    results: list[dict] = []
    retrieval_rows: list[dict] = []
    hallucination_examples: list[dict] = []

    for case in test_cases:
        question = case["question"]
        q_vec = embed_texts([question])[0]
        ranked = top_k_similar(q_vec, chunk_embeddings, top_k)

        retrieved_chunks = [chunks[i] for i, _ in ranked]
        retrieved_text = "\n\n".join(c.text for c in retrieved_chunks)
        similarities = [score for _, score in ranked]

        generated = generate_answer(question, retrieved_text)

        ctx_hit = retrieval_hit(
            retrieved_text,
            case.get("expected_context_contains", []),
        )
        ans_hit = answer_hit(
            generated,
            case.get("expected_answer_contains", []),
        )
        is_hallucination = hallucination_flag(
            generated,
            retrieved_text,
            case.get("expected_answer_contains", []),
        )

        if is_hallucination:
            hallucination_examples.append(
                {
                    "id": case["id"],
                    "question": question,
                    "generated_answer": generated,
                    "expected_answer_contains": case.get("expected_answer_contains", []),
                }
            )

        results.append(
            {
                "id": case["id"],
                "question": question,
                "question_type": case.get("question_type", "unknown"),
                "retrieval_hit": int(ctx_hit),
                "answer_hit": int(ans_hit),
                "hallucination": int(is_hallucination),
                "top_similarity": round(max(similarities) if similarities else 0.0, 4),
                "avg_similarity": round(
                    float(np.mean(similarities)) if similarities else 0.0,
                    4,
                ),
                "generated_answer": generated,
            }
        )

        for rank, (chunk, score) in enumerate(
            zip(retrieved_chunks, similarities),
            start=1,
        ):
            retrieval_rows.append(
                {
                    "id": case["id"],
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "similarity": round(float(score), 4),
                }
            )

    total = len(results)
    retrieval_rate = sum(r["retrieval_hit"] for r in results) / total if total else 0.0
    answer_rate = sum(r["answer_hit"] for r in results) / total if total else 0.0
    hallucination_rate = sum(r["hallucination"] for r in results) / total if total else 0.0

    type_rows: list[dict] = []
    types = sorted({r["question_type"] for r in results})
    for qtype in types:
        subset = [r for r in results if r["question_type"] == qtype]
        n = len(subset)
        type_rows.append(
            {
                "question_type": qtype,
                "count": n,
                "retrieval_hit_rate": round(
                    sum(r["retrieval_hit"] for r in subset) / n if n else 0.0,
                    4,
                ),
                "answer_hit_rate": round(
                    sum(r["answer_hit"] for r in subset) / n if n else 0.0,
                    4,
                ),
                "hallucination_rate": round(
                    sum(r["hallucination"] for r in subset) / n if n else 0.0,
                    4,
                ),
            }
        )

    summary = {
        "total_questions": total,
        "retrieval_hit_rate": round(retrieval_rate, 4),
        "answer_hit_rate": round(answer_rate, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "top_k": top_k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embed_model": config.EMBED_MODEL,
        "gen_model": config.GEN_MODEL,
    }

    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "results.csv", results)
    write_csv(output_dir / "retrieval_hits.csv", retrieval_rows)
    write_csv(output_dir / "question_type_summary.csv", type_rows)
    write_json(output_dir / "hallucination_examples.json", hallucination_examples)

    print(f"Evaluated {total} questions")
    print(f"Retrieval hit rate: {retrieval_rate:.1%}")
    print(f"Answer hit rate: {answer_rate:.1%}")
    print(f"Hallucination rate: {hallucination_rate:.1%}")
    print(f"Results saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local RAG evaluation")
    parser.add_argument("--docs-dir", type=Path, default=APP_DOCS_DEFAULT)
    parser.add_argument("--tests-file", type=Path, default=APP_TESTS_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=APP_OUTPUT_DEFAULT)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    args = parser.parse_args()

    run_evaluation(
        docs_dir=args.docs_dir,
        tests_file=args.tests_file,
        output_dir=args.output_dir,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


APP_ROOT = Path(__file__).resolve().parent.parent
APP_DOCS_DEFAULT = APP_ROOT / "data" / "sample_docs"
APP_TESTS_DEFAULT = APP_ROOT / "data" / "test_cases.json"
APP_OUTPUT_DEFAULT = APP_ROOT / "runs" / "latest"


if __name__ == "__main__":
    main()
