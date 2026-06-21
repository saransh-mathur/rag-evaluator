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
GEN_MODEL    = os.getenv("GEN_MODEL",     "qwen2.5:1.5b-instruct")
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
    custom_instructions: str | None = None,
) -> list[dict]:
    """
    Build the messages list for the OpenAI-compatible API.

    Returns a list of role/content dicts.
    """
    if doc_mode and context.strip():
        system = (
            "You are a helpful assistant. Answer the question thoroughly based ONLY "
            "on the provided context. If the answer is not present, state that you do not know."
        )
        if custom_instructions and custom_instructions.strip():
            system += f"\n\nUser Custom Instructions:\n{custom_instructions}"
        user_content = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        system = "You are a helpful assistant. Answer the question thoroughly."
        if custom_instructions and custom_instructions.strip():
            system += f"\n\nUser Custom Instructions:\n{custom_instructions}"
        user_content = question

    messages: list[dict] = [{"role": "system", "content": system}]

    if chat_history:
        # Keep last 2 turns to minimize input token consumption
        messages.extend(chat_history[-2:])

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
    custom_instructions: str | None = None,
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
        custom_instructions: User custom instructions to guide LLM response

    Returns:
        Tuple of (Clean answer string, usage dictionary)
    """
    try:
        effective_tokens = _dynamic_max_tokens(question, max_tokens)
        messages = _build_prompt(question, context, chat_history, doc_mode, custom_instructions)
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
    custom_instructions: str | None = None,
) -> Iterator[str]:
    """
    Stream answer tokens, with think-tag filtering applied on-the-fly.

    Yields clean text delta strings.
    """
    try:
        effective_tokens = _dynamic_max_tokens(question, max_tokens)
        messages = _build_prompt(question, context, chat_history, doc_mode, custom_instructions)
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


def evaluate_chunks_relevance(
    question: str,
    chunks: list[dict]
) -> dict:
    """
    Assess the relevance of retrieved chunks to a question using the LLM.
    Returns a dict mapping chunk ID (str) to a dict with 'relevance' (1-5) and 'reason'.
    """
    import json
    evaluations = {}
    try:
        chunks_str = ""
        for c in chunks:
            text_val = c.get("text", "") or c.get("chunk_text", "")
            cid = c.get("chunk_id") or c.get("id")
            chunks_str += f"--- CHUNK ID: {cid} ---\n{text_val}\n\n"
            
        prompt = (
            f"You are a search relevance evaluator judge.\n"
            f"Question: {question}\n\n"
            f"Retrieved Chunks:\n{chunks_str}\n"
            f"Evaluate the relevance of each retrieved chunk to the question. "
            f"For each CHUNK ID, assign a relevance score between 1 (completely irrelevant) and 5 (highly relevant and directly answers the question) and a short reason explaining your rating.\n"
            f"You MUST respond ONLY with a JSON object in this exact format:\n"
            f"{{\n"
            f"  \"chunk_id_here\": {{\n"
            f"    \"relevance\": 5,\n"
            f"    \"reason\": \"Contains the exact definition requested.\"\n"
            f"  }}\n"
            f"}}\n"
            f"Do not add any preamble, markdown code blocks, or extra text. Return valid JSON only."
        )
        
        response = _client().chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content or ""
        clean_raw = raw.strip()
        if clean_raw.startswith("```json"):
            clean_raw = clean_raw[7:]
        if clean_raw.endswith("```"):
            clean_raw = clean_raw[:-3]
        clean_raw = clean_raw.strip()
        
        try:
            evaluations = json.loads(clean_raw)
        except Exception as json_err:
            # Fallback regex parsing to tolerate missing commas, preambles, etc.
            evaluations = {}
            import re
            matches = re.finditer(r'"([^"]+)"\s*:\s*\{([^}]+)\}', clean_raw)
            for m in matches:
                cid = m.group(1)
                body = m.group(2)
                
                rel_match = re.search(r'"relevance"\s*:\s*(\d+)', body)
                rel = int(rel_match.group(1)) if rel_match else 3
                
                reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', body)
                reason = reason_match.group(1) if reason_match else "No explanation provided."
                
                evaluations[cid] = {
                    "relevance": rel,
                    "reason": reason
                }
            
            if not evaluations:
                raise json_err
    except Exception as e:
        import traceback
        try:
            with open("/home/saranh/projects/rag-evaluator/app/eval_error.log", "w") as f:
                f.write(f"Exception: {str(e)}\n")
                traceback.print_exc(file=f)
        except Exception:
            pass
        print(f"[DEBUG] Chunk evaluation failed: {e}")
        for c in chunks:
            cid = str(c.get("chunk_id") or c.get("id"))
            evaluations[cid] = {
                "relevance": 3,
                "reason": f"Evaluation unavailable (error: {str(e)})."
            }
    return evaluations


def _run_judge_prompt(prompt: str) -> dict:
    """Helper to run a judge prompt and parse it using standard JSON or regex fallback."""
    import json
    import re
    try:
        response = _client().chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content or ""
        clean_raw = raw.strip()
        if clean_raw.startswith("```json"):
            clean_raw = clean_raw[7:]
        if clean_raw.endswith("```"):
            clean_raw = clean_raw[:-3]
        clean_raw = clean_raw.strip()
        
        try:
            return json.loads(clean_raw)
        except Exception as json_err:
            # Regex fallback for single score + reason format
            # Format expected: { "score": 5, "reason": "..." }
            score_match = re.search(r'"score"\s*:\s*(\d+)', clean_raw)
            reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', clean_raw)
            if score_match:
                return {
                    "score": int(score_match.group(1)),
                    "reason": reason_match.group(1) if reason_match else "No explanation provided."
                }
            raise json_err
    except Exception as e:
        print(f"[DEBUG] Judge execution failed: {e}")
        return {
            "score": 3,
            "reason": f"Evaluation failed: {str(e)}"
        }


def evaluate_rag_triad(question: str, answer: str, context: str) -> dict:
    """
    Evaluate the full RAG Triad (Faithfulness, Answer Relevance, Context Recall)
    using separate LLM prompts.
    """
    # Trim context to avoid blowing up model max token limit
    if len(context) > 4000:
        context = context[:4000] + "... [truncated]"
        
    faithfulness_prompt = (
        f"You are an AI quality judge evaluating RAG answers.\n"
        f"Context:\n{context}\n\n"
        f"Generated Answer:\n{answer}\n\n"
        f"Rate the faithfulness of the generated answer to the provided context. An answer is faithful if it contains no facts, claims, or information that cannot be directly derived from the context.\n"
        f"Assign a score between 1 (totally unfaithful, contains hallucinations) and 5 (completely faithful, grounded perfectly in context).\n"
        f"Respond ONLY with a JSON object in this exact format:\n"
        f"{{\n"
        f"  \"score\": 5,\n"
        f"  \"reason\": \"Explain why in 1 sentence.\"\n"
        f"}}\n"
        f"Do not add any preamble, markdown code blocks, or extra text. Return valid JSON only."
    )
    
    relevance_prompt = (
        f"You are an AI quality judge evaluating RAG answers.\n"
        f"Question:\n{question}\n\n"
        f"Generated Answer:\n{answer}\n\n"
        f"Rate the relevance of the generated answer to the question. The answer is relevant if it directly addresses the question asked, without adding irrelevant chatter or ignoring details.\n"
        f"Assign a score between 1 (completely irrelevant) and 5 (directly and fully answers the question).\n"
        f"Respond ONLY with a JSON object in this exact format:\n"
        f"{{\n"
        f"  \"score\": 5,\n"
        f"  \"reason\": \"Explain why in 1 sentence.\"\n"
        f"}}\n"
        f"Do not add any preamble, markdown code blocks, or extra text. Return valid JSON only."
    )
    
    recall_prompt = (
        f"You are an AI quality judge evaluating RAG retrieval.\n"
        f"Question:\n{question}\n\n"
        f"Retrieved Context Chunks:\n{context}\n\n"
        f"Rate the context recall. Recall is high if the retrieved chunks contain all the necessary details required to formulate a complete answer to the question.\n"
        f"Assign a score between 1 (completely irrelevant chunks, missing all key facts) and 5 (highly relevant chunks containing all facts).\n"
        f"Respond ONLY with a JSON object in this exact format:\n"
        f"{{\n"
        f"  \"score\": 5,\n"
        f"  \"reason\": \"Explain why in 1 sentence.\"\n"
        f"}}\n"
        f"Do not add any preamble, markdown code blocks, or extra text. Return valid JSON only."
    )
    
    return {
        "faithfulness": _run_judge_prompt(faithfulness_prompt),
        "answer_relevance": _run_judge_prompt(relevance_prompt),
        "context_recall": _run_judge_prompt(recall_prompt)
    }

