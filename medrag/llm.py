"""Answer generation — provider-agnostic.

Two interchangeable backends sit behind `answer_stream()`:
  - "ollama"    : a model running locally (free, offline). The default.
  - "anthropic" : Claude via API (higher quality, for serving a public site).

Selected by `config.LLM_PROVIDER`. Both ground the answer in the retrieved
sources and require inline [n] citations, and both stream so the UI can render
tokens as they arrive.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import requests

import config

SYSTEM_PROMPT = """You are MedRAG, a biomedical research assistant that answers \
questions using retrieved scientific literature.

Rules:
- Answer using ONLY the numbered sources provided. Do not rely on prior knowledge \
for factual claims.
- Cite every claim inline with the matching source number(s), e.g. "Amyloid-beta \
accumulation precedes tau pathology [1][3]."
- If the provided sources do not contain enough information to answer, say so \
explicitly rather than guessing.
- Be precise and use appropriate scientific language, but keep the answer readable.
- Never invent citations, PMIDs, or findings.
"""


def build_context(sources: list[dict]) -> str:
    """Render retrieved sources into a numbered context block for the prompt."""
    blocks = []
    for src in sources:
        meta = src["metadata"]
        header = f"[{src['n']}] {meta.get('title', 'Untitled')} — {meta.get('citation', '')}"
        blocks.append(f"{header}\n{src['text']}")
    return "\n\n".join(blocks)


def _user_message(question: str, sources: list[dict]) -> str:
    return (
        f"Sources:\n\n{build_context(sources)}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using the sources above, with inline [n] citations."
    )


def answer_stream(question: str, sources: list[dict]) -> Iterator[str]:
    """Yield answer text deltas, grounded in `sources`, from the active provider."""
    user_message = _user_message(question, sources)
    if config.LLM_PROVIDER == "ollama":
        yield from _ollama_stream(user_message)
    elif config.LLM_PROVIDER == "groq":
        yield from _groq_stream(user_message)
    elif config.LLM_PROVIDER == "anthropic":
        yield from _anthropic_stream(user_message)
    else:
        raise RuntimeError(
            f"Unknown MEDRAG_PROVIDER '{config.LLM_PROVIDER}'. "
            "Use 'ollama', 'groq', or 'anthropic'."
        )


def answer(question: str, sources: list[dict]) -> str:
    """Non-streaming convenience wrapper (collects the full answer)."""
    return "".join(answer_stream(question, sources))


# -- Ollama (local) -------------------------------------------------------
def _ollama_stream(user_message: str) -> Iterator[str]:
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
        "options": {"num_predict": config.ANSWER_MAX_TOKENS},
    }
    url = f"{config.OLLAMA_HOST}/api/chat"
    try:
        resp = requests.post(url, json=payload, stream=True, timeout=600)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {config.OLLAMA_HOST}. Install it from "
            f"https://ollama.com, start it, then run `ollama pull {config.OLLAMA_MODEL}`."
        ) from exc

    if resp.status_code == 404:
        raise RuntimeError(
            f"Ollama model '{config.OLLAMA_MODEL}' is not installed. "
            f"Run `ollama pull {config.OLLAMA_MODEL}`."
        )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        chunk = data.get("message", {}).get("content", "")
        if chunk:
            yield chunk
        if data.get("done"):
            break


# -- Groq (hosted, free tier) — uses the OpenAI-compatible chat endpoint ----
def _groq_stream(user_message: str) -> Iterator[str]:
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set (required when MEDRAG_PROVIDER=groq). "
            "Get a free key at https://console.groq.com/keys and add it to .env."
        )
    payload = {
        "model": config.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
        "max_tokens": config.ANSWER_MAX_TOKENS,
    }
    headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
    try:
        resp = requests.post(
            f"{config.GROQ_BASE}/chat/completions",
            json=payload,
            headers=headers,
            stream=True,
            timeout=120,
        )
    except requests.ConnectionError as exc:
        raise RuntimeError(f"Could not reach Groq at {config.GROQ_BASE}.") from exc

    if resp.status_code == 401:
        raise RuntimeError("Groq rejected the API key (401). Check GROQ_API_KEY.")
    if resp.status_code == 429:
        raise RuntimeError("Groq free-tier rate limit hit (429). Please try again shortly.")
    if resp.status_code == 404:
        raise RuntimeError(
            f"Groq model '{config.GROQ_MODEL}' not found. "
            "Pick a current model from https://console.groq.com and set MEDRAG_GROQ_MODEL."
        )
    resp.raise_for_status()

    # Server-Sent Events: each line is `data: {json}`, terminated by `data: [DONE]`.
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :].strip()
        if data == "[DONE]":
            break
        obj = json.loads(data)
        delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
        if delta:
            yield delta


# -- Anthropic (Claude) — dormant unless LLM_PROVIDER="anthropic" ----------
def _anthropic_stream(user_message: str) -> Iterator[str]:
    import anthropic  # imported lazily so local-only users never need it loaded

    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set (required when MEDRAG_PROVIDER=anthropic). "
            "Add it to .env, or use the default local Ollama provider."
        )
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    with client.messages.stream(
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.ANSWER_MAX_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        yield from stream.text_stream
