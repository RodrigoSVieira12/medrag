"""Text embeddings via ChromaDB's built-in ONNX model.

Uses the all-MiniLM-L6-v2 model that ships with ChromaDB, run through ONNX
Runtime rather than PyTorch. Same embeddings, but a small fraction of the
memory — PyTorch's ~1GB resident footprint is what blew past a 512MB host.
The model (~80MB) downloads to a local cache on first use.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _embedder():
    # Imported lazily so `import medrag` stays cheap.
    from chromadb.utils import embedding_functions

    return embedding_functions.DefaultEmbeddingFunction()


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts into a list of float vectors."""
    if not texts:
        return []
    return [list(map(float, vec)) for vec in _embedder()(texts)]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
