"""The RAG pipeline: search PubMed -> enrich -> embed/cache -> retrieve -> answer.

This is the module the UI and CLI talk to. It exposes a small, callback-friendly
surface so the front end can show progress ("Searching PubMed...", "Indexing
paper 3/8...") without knowing the internals.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import config
from . import embeddings, fulltext, llm
from .chunking import chunk_text
from .pubmed import PubMedClient
from .vectorstore import VectorStore

# Progress callback: (message, current, total) -> None
ProgressFn = Callable[[str, int, int], None]


class MedRAG:
    def __init__(self) -> None:
        self.pubmed = PubMedClient()
        self.store = VectorStore()

    def index_query(
        self,
        question: str,
        max_papers: int = config.DEFAULT_MAX_PAPERS,
        use_full_text: bool = True,
        progress: ProgressFn | None = None,
    ) -> list[str]:
        """Search PubMed for the question and index any papers not already cached.

        Returns the list of PMIDs relevant to this question.
        """

        def report(msg: str, cur: int = 0, total: int = 0) -> None:
            if progress:
                progress(msg, cur, total)

        report("Searching PubMed...")
        pmids = self.pubmed.search(question, max_results=max_papers)
        if not pmids:
            return []

        # Only fetch + embed papers we haven't seen before (cache hit otherwise).
        new_pmids = [p for p in pmids if not self.store.has_paper(p)]
        if new_pmids:
            articles = self.pubmed.fetch(new_pmids)
            total = len(articles)
            for i, article in enumerate(articles, start=1):
                report(f"Indexing: {article.title[:60]}...", i, total)
                if use_full_text:
                    fulltext.enrich(article, self.pubmed)
                chunks = chunk_text(article.content)
                if not chunks:
                    continue
                vectors = embeddings.embed(chunks)
                self.store.add_paper(article, chunks, vectors)

        return pmids

    def retrieve(
        self, question: str, pmids: list[str], top_k: int = config.DEFAULT_TOP_K
    ) -> list[dict]:
        """Retrieve the top_k most relevant chunks, numbered for citation."""
        query_vec = embeddings.embed_one(question)
        hits = self.store.query(query_vec, top_k=top_k, pmids=pmids)
        # Number sources by PMID so repeated chunks from one paper share a citation.
        sources: list[dict] = []
        pmid_to_n: dict[str, int] = {}
        for hit in hits:
            pmid = hit["metadata"]["pmid"]
            if pmid not in pmid_to_n:
                pmid_to_n[pmid] = len(pmid_to_n) + 1
            sources.append({"n": pmid_to_n[pmid], **hit})
        return sources

    def answer_stream(self, question: str, sources: list[dict]) -> Iterator[str]:
        return llm.answer_stream(question, sources)

    def ask(
        self,
        question: str,
        max_papers: int = config.DEFAULT_MAX_PAPERS,
        top_k: int = config.DEFAULT_TOP_K,
        use_full_text: bool = True,
        progress: ProgressFn | None = None,
    ) -> tuple[str, list[dict]]:
        """End-to-end convenience: returns (answer_text, sources)."""
        pmids = self.index_query(question, max_papers, use_full_text, progress)
        if not pmids:
            return "No PubMed results found for that question.", []
        sources = self.retrieve(question, pmids, top_k)
        answer_text = llm.answer(question, sources)
        return answer_text, sources
