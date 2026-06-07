"""Local sentence embeddings.

Uses sentence-transformers so there's no second paid API key to manage —
embeddings run on your machine (CPU is fine for this scale). The model is
loaded lazily and cached, since the first load downloads weights (~80MB).
"""

from __future__ import annotations

from functools import lru_cache

import config


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so `import medrag` stays cheap and doesn't pull torch
    # until embeddings are actually needed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts into a list of float vectors."""
    if not texts:
        return []
    vectors = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
