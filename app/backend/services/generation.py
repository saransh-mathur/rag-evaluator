"""LLM generation service using Ollama."""

from openai import OpenAI
from typing import List, Iterator
import os
from dotenv import load_dotenv

load_dotenv()

GEN_BASE_URL = os.getenv("GEN_BASE_URL", "http://localhost:11434/v1")
GEN_MODEL = os.getenv("GEN_MODEL", "deepseek-r1:7b")
GEN_API_KEY = os.getenv("GEN_API_KEY", "ollama")


def _build_prompt(question: str, context: str) -> str:
    has_context = bool(context and context.strip())
    if has_context:
        return (
            "You are a knowledgeable assistant. Your job is to give thorough, "
            "well-structured answers based on the provided context.\n\n"
            "Guidelines:\n"
            "- Answer in as much detail as the context supports\n"
            "- Use bullet points, numbered lists, or headers where they aid clarity\n"
            "- Quote or reference specific parts of the context when relevant\n"
            "- If the context only partially answers the question, answer what you can "
            "and clearly state what is not covered\n"
            "- Do not fabricate information not present in the context\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
    else:
        return (
            "You are a knowledgeable assistant. Answer the following question as "
            "thoroughly and clearly as possible.\n\n"
            f"Question: {question}\n\nAnswer:"
        )


def generate_answer(
    question: str,
    context: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> str:
    """
    Generate an answer using local LLM based on context.

    Args:
        question: User question
        context: Retrieved context chunks
        temperature: LLM temperature (0 = deterministic, 1 = creative)
        max_tokens: Maximum tokens to generate (default 2048)

    Returns:
        Generated answer string
    """
    try:
        client = OpenAI(base_url=GEN_BASE_URL, api_key=GEN_API_KEY)
        response = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": _build_prompt(question, context)}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=180,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}")


def generate_answer_stream(
    question: str,
    context: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> Iterator[str]:
    """
    Stream answer tokens from the local LLM.

    Yields individual text delta strings as they arrive from the model.
    Raises RuntimeError on connection or API failure.
    """
    try:
        client = OpenAI(base_url=GEN_BASE_URL, api_key=GEN_API_KEY)
        stream = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": _build_prompt(question, context)}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            timeout=180,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        raise RuntimeError(f"Streaming generation failed: {e}")


def generate_with_history(
    question: str,
    context: str,
    chat_history: List[dict] = None,
    temperature: float = 0.1
) -> str:
    """
    Generate answer with chat history (multi-turn conversation).

    Args:
        question: Current question
        context: Retrieved context
        chat_history: Previous messages [{"role": "user"/"assistant", "content": "..."}]
        temperature: LLM temperature

    Returns:
        Generated answer string
    """
    try:
        client = OpenAI(base_url=GEN_BASE_URL, api_key=GEN_API_KEY)

        system_prompt = (
            "You are a helpful AI assistant. Answer the user's questions using the "
            "provided context. If context doesn't contain the answer, say you don't know."
        )
        messages = [{"role": "system", "content": system_prompt}]

        if chat_history:
            messages.extend(chat_history)

        messages.append({"role": "user", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model=GEN_MODEL,
            messages=messages,
            temperature=temperature,
            timeout=180,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}")
