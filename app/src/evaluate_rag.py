"""Evaluate RAG quality by querying the live backend API.

The evaluation pipeline now goes through the same retrieval + generation
path that real users experience:

  POST /api/queries/ask  →  returns answer + retrieved_chunks + top_similarity

This means evaluation results reflect actual pgvector search quality and
the production LLM prompt, not a separate in-memory simulation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from utils import (
    answer_hit,
    hallucination_flag,
    read_json,
    retrieval_hit,
    write_csv,
    write_json,
)

# Default backend URL — override with --api-url
DEFAULT_API_URL = "http://localhost:8000"


def ask_backend(
    api_url: str,
    user_id: str,
    question: str,
    top_k: int,
) -> dict:
    """
    Call POST /api/queries/ask and return the parsed JSON response.

    Returns a dict with keys: question, answer, retrieved_chunks, top_similarity, query_id
    """
    url = f"{api_url.rstrip('/')}/api/queries/ask"
    payload = {
        "question": question,
        "user_id": user_id,
        "top_k": top_k,
        "temperature": 0.1,
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def run_evaluation(
    tests_file: Path,
    output_dir: Path,
    top_k: int,
    api_url: str,
    user_id: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    test_cases = read_json(tests_file)

    results: list[dict] = []
    retrieval_rows: list[dict] = []
    hallucination_examples: list[dict] = []

    for case in test_cases:
        question = case["question"]
        print(f"  evaluating: {case['id']} — {question[:60]}")

        response = ask_backend(api_url, user_id, question, top_k)

        generated = response["answer"]
        top_similarity = float(response.get("top_similarity", 0.0))
        chunks = response.get("retrieved_chunks", [])

        # Build retrieved_text from chunk snippets for phrase-match checks.
        # The API returns truncated text (200 chars); good enough for hit checks.
        retrieved_text = "\n\n".join(c.get("text", "") for c in chunks)
        similarities = [float(c.get("similarity", 0.0)) for c in chunks]

        ctx_hit = retrieval_hit(retrieved_text, case.get("expected_context_contains", []))
        ans_hit = answer_hit(generated, case.get("expected_answer_contains", []))
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
                "top_similarity": round(top_similarity, 4),
                "avg_similarity": round(
                    float(np.mean(similarities)) if similarities else 0.0, 4
                ),
                "generated_answer": generated,
            }
        )

        for rank, chunk in enumerate(chunks, start=1):
            retrieval_rows.append(
                {
                    "id": case["id"],
                    "rank": rank,
                    "chunk_id": chunk.get("chunk_id"),
                    "source_file": chunk.get("filename", "unknown"),
                    "similarity": round(float(chunk.get("similarity", 0.0)), 4),
                }
            )

    total = len(results)
    retrieval_rate = sum(r["retrieval_hit"] for r in results) / total if total else 0.0
    answer_rate = sum(r["answer_hit"] for r in results) / total if total else 0.0
    hallucination_rate = (
        sum(r["hallucination"] for r in results) / total if total else 0.0
    )

    type_rows: list[dict] = []
    for qtype in sorted({r["question_type"] for r in results}):
        subset = [r for r in results if r["question_type"] == qtype]
        n = len(subset)
        type_rows.append(
            {
                "question_type": qtype,
                "count": n,
                "retrieval_hit_rate": round(
                    sum(r["retrieval_hit"] for r in subset) / n if n else 0.0, 4
                ),
                "answer_hit_rate": round(
                    sum(r["answer_hit"] for r in subset) / n if n else 0.0, 4
                ),
                "hallucination_rate": round(
                    sum(r["hallucination"] for r in subset) / n if n else 0.0, 4
                ),
            }
        )

    summary = {
        "total_questions": total,
        "retrieval_hit_rate": round(retrieval_rate, 4),
        "answer_hit_rate": round(answer_rate, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "top_k": top_k,
        "embed_model": config.EMBED_MODEL,
        "gen_model": config.GEN_MODEL,
        "api_url": api_url,
        "eval_user_id": user_id,
    }

    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "results.csv", results)
    write_csv(output_dir / "retrieval_hits.csv", retrieval_rows)
    write_csv(output_dir / "question_type_summary.csv", type_rows)
    write_json(output_dir / "hallucination_examples.json", hallucination_examples)

    print(f"\nEvaluated {total} questions")
    print(f"Retrieval hit rate : {retrieval_rate:.1%}")
    print(f"Answer hit rate    : {answer_rate:.1%}")
    print(f"Hallucination rate : {hallucination_rate:.1%}")
    print(f"Results saved to   : {output_dir}")


def main() -> None:
    APP_ROOT = Path(__file__).resolve().parent.parent
    default_tests = APP_ROOT / "data" / "test_cases.json"
    default_output = APP_ROOT / "runs" / "latest"

    parser = argparse.ArgumentParser(
        description="Evaluate RAG quality via the live backend API"
    )
    parser.add_argument("--tests-file", type=Path, default=default_tests)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Base URL of the running backend (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--user-id",
        default="eval-user",
        help=(
            "user_id whose uploaded documents are searched during evaluation. "
            "Upload your test documents via the UI or API before running eval."
        ),
    )
    args = parser.parse_args()

    run_evaluation(
        tests_file=args.tests_file,
        output_dir=args.output_dir,
        top_k=args.top_k,
        api_url=args.api_url,
        user_id=args.user_id,
    )


if __name__ == "__main__":
    main()
