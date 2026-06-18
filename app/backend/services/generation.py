"""LLM generation service using Ollama."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterator, List

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

GEN_BASE_URL = os.getenv("GEN_BASE_URL", "http://localhost:11434/v1")
GEN_MODEL    = os.getenv("GEN_MODEL",     "gemma4:12b")
GEN_API_KEY  = os.getenv("GEN_API_KEY", os.getenv("GEMINI_API_KEY", "ollama"))

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _client() -> OpenAI:
    return OpenAI(base_url=GEN_BASE_URL, api_key=GEN_API_KEY)


def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> reasoning blocks emitted by deepseek-r1."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _dynamic_max_tokens(question: str, requested: int) -> int:
    """
    Adjust max_tokens based on question complexity.
    Short factual questions rarely need 2048 tokens.
    """
    q = question.strip()
    if len(q) < 40 and not any(kw in q.lower() for kw in (
        "explain", "describe", "how", "why", "summarize", "detail", "compare"
    )):
        return min(requested, 512)
    if len(q) > 120 or any(kw in q.lower() for kw in (
        "explain", "in depth", "step by step", "comprehensive", "detailed"
    )):
        return min(requested, 4096)
    return requested


def _build_context_header(chunk_index: int, filename: str, total: int) -> str:
    return f"[Source: {filename}, chunk {chunk_index + 1} of {total}]"


def _build_prompt(
    question: str,
    context: str,
    chat_history: list[dict] | None = None,
    doc_mode: bool = True,
) -> list[dict]:
    """
    Build the messages list for the OpenAI-compatible API.

    Returns a list of role/content dicts.
    """
    if doc_mode and context.strip():
        system = (
            "You are a knowledgeable assistant. Give thorough, well-structured answers "
            "based on the provided context. Guidelines:\n"
            "- Answer in as much detail as the context supports\n"
            "- Use bullet points, numbered lists, or headers where they aid clarity\n"
            "- Start with a one-sentence TL;DR, then elaborate\n"
            "- Quote or reference source labels when relevant\n"
            "- If context only partially answers, state what is and isn't covered\n"
            "- Do not fabricate information not present in the context"
        )
        user_content = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        system = (
            "You are a knowledgeable assistant. Answer thoroughly and clearly. "
            "Use structure (headers, bullets) where it aids understanding."
        )
        user_content = question

    messages: list[dict] = [{"role": "system", "content": system}]

    if chat_history:
        # Keep last 6 turns to avoid context overflow
        messages.extend(chat_history[-6:])

    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    context: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    chat_history: list[dict] | None = None,
    doc_mode: bool = True,
) -> tuple[str, dict]:
    """
    Generate a complete answer (non-streaming).

    Args:
        question:     User question
        context:      Retrieved context chunks with headers
        temperature:  LLM temperature
        max_tokens:   Upper bound on tokens to generate
        chat_history: Previous turns for multi-turn conversations
        doc_mode:     If True, constrain answer to context

    Returns:
        Tuple of (Clean answer string, usage dictionary)
    """
    try:
        effective_tokens = _dynamic_max_tokens(question, max_tokens)
        messages = _build_prompt(question, context, chat_history, doc_mode)
        response = _client().chat.completions.create(
            model=GEN_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=effective_tokens,
            timeout=300,
        )
        raw = response.choices[0].message.content or ""
        
        usage = response.usage
        usage_dict = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }
        return _strip_think_tags(raw), usage_dict
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}")


def generate_answer_stream(
    question: str,
    context: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    chat_history: list[dict] | None = None,
    doc_mode: bool = True,
) -> Iterator[str]:
    """
    Stream answer tokens, with think-tag filtering applied on-the-fly.

    Yields clean text delta strings.
    """
    try:
        effective_tokens = _dynamic_max_tokens(question, max_tokens)
        messages = _build_prompt(question, context, chat_history, doc_mode)
        stream = _client().chat.completions.create(
            model=GEN_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=effective_tokens,
            stream=True,
            timeout=300,
        )

        # Buffer to handle think-tag stripping across chunk boundaries
        buffer = ""
        in_think = False
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            buffer += delta

            # Strip complete <think>…</think> blocks
            while True:
                if in_think:
                    end = buffer.find("</think>")
                    if end == -1:
                        buffer = ""  # still inside think block, discard
                        break
                    buffer = buffer[end + len("</think>"):].lstrip("\n")
                    in_think = False
                else:
                    start = buffer.find("<think>")
                    if start == -1:
                        break
                    # Yield content before the think tag
                    if start > 0:
                        yield buffer[:start]
                    buffer = buffer[start + len("<think>"):]
                    in_think = True

            if not in_think and buffer:
                yield buffer
                buffer = ""

        # Yield any remaining buffer content
        if buffer and not in_think:
            yield buffer

    except Exception as e:
        raise RuntimeError(f"Streaming generation failed: {e}")


def generate_hypothetical_document(question: str) -> str:
    """
    HyDE: Generate a hypothetical document that would answer the question.
    Used to improve embedding-based retrieval for vague queries.
    """
    try:
        prompt = (
            "Write a short, factual passage (2-4 sentences) that would directly "
            f"answer the following question:\n\n{question}\n\nPassage:"
        )
        response = _client().chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256,
            timeout=300,
        )
        return _strip_think_tags(response.choices[0].message.content or "")
    except Exception:
        return question  # fall back to original query


def generate_query_expansions(question: str, n: int = 4) -> list[str]:
    """
    Generate semantically related search queries for multi-query retrieval.

    Returns an empty list on failure so retrieval can fall back to the original
    query without impacting the user-facing answer path.
    """
    try:
        prompt = (
            f"Generate exactly {n} concise search queries that are semantically "
            "similar to the user's question. Keep each query short. "
            "Return ONLY the queries, one per line, with no numbering or extra text.\n\n"
            f"Question: {question}\n\nSearch queries:"
        )
        response = _client().chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
            timeout=300,
        )
        raw = _strip_think_tags(response.choices[0].message.content or "")
        expansions: list[str] = []
        seen: set[str] = set()
        for line in raw.splitlines():
            cleaned = line.strip().lstrip("-•·123456789.) ").strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key == question.lower() or key in seen:
                continue
            seen.add(key)
            expansions.append(cleaned)
            if len(expansions) >= n:
                break
        return expansions
    except Exception:
        return []


def generate_document_summary(text: str, filename: str) -> str:
    """
    Generate a 2-3 sentence summary of a document for display in the UI.
    """
    try:
        snippet = text[:3000]
        prompt = (
            f"Summarize the following document '{filename}' in 2-3 sentences. "
            "Be specific about the main topics covered.\n\n"
            f"Document:\n{snippet}\n\nSummary:"
        )
        response = _client().chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150,
            timeout=300,
        )
        return _strip_think_tags(response.choices[0].message.content or "")
    except Exception:
        return ""


def generate_suggestions(
    question: str,
    answer: str,
    temperature: float = 0.3,
) -> list[str]:
    """
    Generate 3 follow-up question suggestions based on the Q&A exchange.
    """
    try:
        prompt = (
            "Based on the following question and answer, suggest exactly 3 concise "
            "follow-up questions the user might want to ask next. "
            "Return ONLY the 3 questions, one per line, no numbering, no extra text.\n\n"
            f"Question: {question}\n\nAnswer: {answer[:800]}\n\nFollow-up questions:"
        )
        response = _client().chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=200,
            timeout=300,
        )
        raw = _strip_think_tags(response.choices[0].message.content or "")
        return [
            line.strip().lstrip("-•·123456789.) ").strip()
            for line in raw.strip().splitlines()
            if line.strip() and len(line.strip()) > 5
        ][:3]
    except Exception:
        return []


def generate_search_autocomplete(partial: str, past_questions: list[str]) -> list[str]:
    """
    Return up to 5 autocomplete suggestions based on partial input and past questions.
    Pure string matching — no LLM call needed.
    """
    partial_lower = partial.lower()
    matches = [
        q for q in past_questions
        if partial_lower in q.lower() and q.lower() != partial_lower
    ]
    return matches[:5]
