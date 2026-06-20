"""Reusable helper functions for RAG evaluation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent.parent
load_dotenv(APP_DIR / ".env")


def read_json(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def contains_all(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return all(phrase.lower() in lowered for phrase in phrases)


def contains_any(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in phrases)


def top_k_similar(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    k: int,
) -> list[tuple[int, float]]:
    if doc_vecs.size == 0:
        return []
    sims = cosine_similarity(query_vec.reshape(1, -1), doc_vecs)[0]
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def retrieval_hit(retrieved_text: str, expected_phrases: list[str]) -> bool:
    if not expected_phrases:
        return True
    return contains_any(retrieved_text, expected_phrases)


def answer_hit(generated: str, expected_phrases: list[str]) -> bool:
    if not expected_phrases:
        return True

    # Try LLM-as-a-Judge first
    try:
        from openai import OpenAI
        api_key = os.getenv("GEN_API_KEY")
        if api_key:
            client = OpenAI(
                base_url=os.getenv("GEN_BASE_URL", "http://localhost:11434/v1"),
                api_key=api_key,
            )
            model = os.getenv("GEN_MODEL", "gemma4:12b")
            phrases_str = "\n".join(f"- {p}" for p in expected_phrases)
            prompt = (
                "You are an objective AI grader. You will judge if a Student Answer semantically "
                "contains all of the expected facts listed below.\n\n"
                "Expected Facts:\n"
                f"{phrases_str}\n\n"
                f"Student Answer:\n{generated}\n\n"
                "Respond with a single word: 'YES' if all expected facts are semantically covered "
                "in the student's answer, or 'NO' if any expected fact is missing or contradicted. "
                "Do not write any other explanation or words."
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
                timeout=15,
            )
            res = response.choices[0].message.content or ""
            res_clean = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip().upper()
            if "YES" in res_clean:
                return True
            elif "NO" in res_clean:
                return False
    except Exception:
        pass

    # Fallback to substring matching
    return contains_all(generated, expected_phrases)


def hallucination_flag(
    generated: str,
    context: str,
    expected_phrases: list[str],
) -> bool:
    """True when the answer misses expected facts or fails groundness checks."""
    if not generated.strip():
        return False

    # Try LLM-as-a-Judge first
    try:
        from openai import OpenAI
        api_key = os.getenv("GEN_API_KEY")
        if api_key:
            client = OpenAI(
                base_url=os.getenv("GEN_BASE_URL", "http://localhost:11434/v1"),
                api_key=api_key,
            )
            model = os.getenv("GEN_MODEL", "gemma4:12b")
            prompt = (
                "You are an objective AI grader. You will compare a Generated Answer with the retrieved Source Context.\n\n"
                f"Source Context:\n{context}\n\n"
                f"Generated Answer:\n{generated}\n\n"
                "Check if the generated answer contains assertions or claims that are unsupported by, contradicted by, "
                "or completely missing from the Source Context.\n"
                "Respond with a single word: 'YES' if the generated answer contains hallucinations (claims unsupported "
                "by the context), or 'NO' if the generated answer is completely grounded in the context. "
                "Do not write any other explanation or words."
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
                timeout=15,
            )
            res = response.choices[0].message.content or ""
            res_clean = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip().upper()
            if "YES" in res_clean:
                return True
            elif "NO" in res_clean:
                return False
    except Exception:
        pass

    # Fallback to the original substring check
    if answer_hit(generated, expected_phrases):
        return False
    return not contains_any(generated, expected_phrases)
