"""Word-based chunking with overlap.

Overlap keeps a claim that straddles a boundary retrievable from either side.
Word-based (rather than character-based) chunks align reasonably with the
embedding model's token budget without needing a tokenizer dependency.
"""

from __future__ import annotations

import config


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE_WORDS,
    overlap: int = config.CHUNK_OVERLAP_WORDS,
) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks
