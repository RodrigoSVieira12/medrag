"""Answer generation with Claude.

Grounds the answer in retrieved sources and requires inline [n] citations so
every claim is traceable back to a PubMed record. Streams the response so the
UI can render tokens as they arrive.
"""

from __future__ import annotations

from collections.abc import Iterator

import anthropic

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


def _client() -> anthropic.Anthropic:
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def build_context(sources: list[dict]) -> str:
    """Render retrieved sources into a numbered context block for the prompt."""
    blocks = []
    for src in sources:
        meta = src["metadata"]
        header = f"[{src['n']}] {meta.get('title', 'Untitled')} — {meta.get('citation', '')}"
        blocks.append(f"{header}\n{src['text']}")
    return "\n\n".join(blocks)


def answer_stream(question: str, sources: list[dict]) -> Iterator[str]:
    """Yield answer text deltas grounded in `sources`."""
    context = build_context(sources)
    user_message = (
        f"Sources:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using the sources above, with inline [n] citations."
    )

    with _client().messages.stream(
        model=config.MODEL,
        max_tokens=config.ANSWER_MAX_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        yield from stream.text_stream


def answer(question: str, sources: list[dict]) -> str:
    """Non-streaming convenience wrapper (collects the full answer)."""
    return "".join(answer_stream(question, sources))
