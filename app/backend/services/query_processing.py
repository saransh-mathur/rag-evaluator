"""Query normalization and routing helpers used before retrieval."""

from __future__ import annotations

import re


ABBREVIATIONS = {
    "auth": "authentication",
    "db": "database",
    "k8s": "kubernetes",
    "llm": "large language model",
    "pg": "postgres",
    "postgresql": "postgres",
    "rag": "retrieval augmented generation",
    "s3": "simple storage service",
}


def normalize_query(query: str) -> str:
    """
    Normalize a user query for lexical and vector retrieval.

    This intentionally stays conservative: it removes noisy punctuation and
    expands common technical abbreviations without changing query meaning.
    """
    q = query.lower()
    q = re.sub(r"[^\w\s]", " ", q)
    words = [ABBREVIATIONS.get(word, word) for word in q.split()]
    return " ".join(words)


def determine_query_route(query: str) -> str:
    """
    Classify the query into either 'RAG' or 'GENERAL'.
    If the question is general or a greeting and does not need document context, return 'GENERAL'.
    """
    import os
    from openai import OpenAI

    try:
        base_url = os.getenv("GEN_BASE_URL", "http://localhost:11434/v1")
        api_key = os.getenv("GEN_API_KEY", "ollama")
        model = os.getenv("GEN_MODEL", "qwen2.5:1.5b-instruct")

        client = OpenAI(base_url=base_url, api_key=api_key)

        prompt = (
            "Classify the following user question into exactly one of two categories:\n"
            "1. 'RAG': The question asks about specific details, indexing, async python, code details, "
            "or requires search/document context to be answered accurately.\n"
            "2. 'GENERAL': The question is a greeting, conversational, or is a generic query that can be answered "
            "using standard developer knowledge without any specific documents (e.g., 'What is FAISS?', 'hello', 'who are you?').\n\n"
            f"User Question: {query}\n\n"
            "Respond with only the single word: 'RAG' or 'GENERAL' with no explanation."
        )

        print(f"[DEBUG] Routing query: '{query[:50]}'...")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
            timeout=5,
        )
        res = response.choices[0].message.content or ""
        res_clean = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip().upper()

        route = "GENERAL" if "GENERAL" in res_clean else "RAG"
        print(f"[DEBUG] Query classified route: {route}")
        return route
    except Exception as e:
        print(f"[DEBUG] Routing failed: {e}. Defaulting to RAG.")
        return "RAG"
