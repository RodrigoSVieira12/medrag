"""Persistent vector store backed by ChromaDB.

Doubles as the cache: chunk IDs are deterministic (`<pmid>:<index>`), so a
paper that's already been indexed is detected via `has_paper()` and never
re-fetched or re-embedded. The Chroma collection persists to disk between
runs, so popular topics get faster over time.
"""

from __future__ import annotations

import chromadb

import config
from .pubmed import Article


class VectorStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def has_paper(self, pmid: str) -> bool:
        """True if this PMID's chunks are already indexed (cache hit)."""
        existing = self.collection.get(ids=[f"{pmid}:0"])
        return bool(existing["ids"])

    def add_paper(
        self, article: Article, chunks: list[str], embeddings: list[list[float]]
    ) -> None:
        if not chunks:
            return
        ids = [f"{article.pmid}:{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "pmid": article.pmid,
                "title": article.title,
                "journal": article.journal,
                "year": article.year,
                "citation": article.citation,
                "url": article.url,
                "oa_url": article.oa_url,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        self.collection.add(
            ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas
        )

    def query(
        self, query_embedding: list[float], top_k: int, pmids: list[str] | None = None
    ) -> list[dict]:
        """Return the top_k most similar chunks, optionally restricted to `pmids`."""
        where = {"pmid": {"$in": pmids}} if pmids else None
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        hits = []
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({"text": doc, "metadata": meta, "distance": dist})
        return hits
