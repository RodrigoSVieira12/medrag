"""MedRAG — Streamlit UI.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import config
from medrag.rag import MedRAG

st.set_page_config(page_title="MedRAG — Biomedical Literature Q&A")


@st.cache_resource
def get_engine() -> MedRAG:
    # Cached across reruns so the embedding model + Chroma client load once.
    return MedRAG()


def render_sources(sources: list[dict]) -> None:
    """Show the cited sources, de-duplicated by citation number."""
    seen: dict[int, dict] = {}
    for src in sources:
        seen.setdefault(src["n"], src["metadata"])
    with st.expander(f"Sources ({len(seen)})", expanded=False):
        for n in sorted(seen):
            meta = seen[n]
            line = f"**[{n}]** {meta.get('title', 'Untitled')}  \n{meta.get('citation', '')}"
            links = [f"[PubMed]({meta.get('url', '')})"]
            if meta.get("oa_url"):
                links.append(f"[Open access]({meta['oa_url']})")
            st.markdown(f"{line}  \n{' · '.join(links)}")


st.title("MedRAG")
st.caption(
    "Ask a biomedical question. MedRAG searches PubMed in real time, retrieves "
    "abstracts and open-access full text, and answers with cited sources."
)

# --- Sidebar settings ----------------------------------------------------
with st.sidebar:
    st.header("Settings")
    max_papers = st.slider("Papers to fetch", 3, 20, config.DEFAULT_MAX_PAPERS)
    top_k = st.slider("Context chunks", 3, 12, config.DEFAULT_TOP_K)
    use_full_text = st.toggle("Use open-access full text", value=True)
    st.divider()
    st.caption(f"Provider: `{config.LLM_PROVIDER}`")
    st.caption(f"Model: `{config.active_model()}`")
    st.caption(f"Embeddings: `{config.EMBEDDING_MODEL}`")

# --- Provider check ------------------------------------------------------
if config.LLM_PROVIDER == "anthropic" and not config.ANTHROPIC_API_KEY:
    st.error(
        "Provider is `anthropic` but no `ANTHROPIC_API_KEY` is set. Add it to "
        "`.env`, or set `MEDRAG_PROVIDER=ollama` to run locally for free."
    )
    st.stop()

# --- Main query ----------------------------------------------------------
question = st.text_input(
    "Your question",
    placeholder="e.g. What is the role of amyloid-beta in Alzheimer's disease?",
)

if st.button("Ask", type="primary") and question:
    engine = get_engine()

    status = st.status("Working...", expanded=True)

    def progress(msg: str, cur: int, total: int) -> None:
        status.update(label=f"{msg} ({cur}/{total})" if total else msg)

    pmids = engine.index_query(question, max_papers, use_full_text, progress)
    if not pmids:
        status.update(label="No results", state="error")
        st.warning("No PubMed results found for that question.")
        st.stop()

    status.update(label="Retrieving relevant passages...")
    sources = engine.retrieve(question, pmids, top_k)
    status.update(label="Done", state="complete")

    render_sources(sources)
    st.subheader("Answer")
    st.write_stream(engine.answer_stream(question, sources))
