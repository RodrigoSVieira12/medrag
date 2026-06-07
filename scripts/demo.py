"""Command-line demo / smoke test for the MedRAG pipeline.

Usage:
    python scripts/demo.py "What is the role of amyloid-beta in Alzheimer's disease?"
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.rag import MedRAG  # noqa: E402


def main() -> None:
    question = (
        " ".join(sys.argv[1:])
        or "What is the role of amyloid-beta in Alzheimer's disease?"
    )
    print(f"\nQ: {question}\n")

    engine = MedRAG()

    def progress(msg: str, cur: int, total: int) -> None:
        suffix = f" ({cur}/{total})" if total else ""
        print(f"  · {msg}{suffix}")

    pmids = engine.index_query(question, progress=progress)
    if not pmids:
        print("No PubMed results found.")
        return

    sources = engine.retrieve(question, pmids)

    print("\nAnswer:\n")
    for delta in engine.answer_stream(question, sources):
        print(delta, end="", flush=True)
    print("\n")

    print("Sources:")
    seen = {}
    for src in sources:
        seen.setdefault(src["n"], src["metadata"])
    for n in sorted(seen):
        meta = seen[n]
        print(f"  [{n}] {meta.get('title', '')[:70]} — {meta.get('url', '')}")


if __name__ == "__main__":
    main()
