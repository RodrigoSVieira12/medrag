# MedRAG — Biomedical Literature Q&A Assistant

MedRAG answers biomedical questions by retrieving and reasoning over real
scientific papers from **PubMed**, in real time, with **inline citations** back
to every source. It's a Retrieval-Augmented Generation (RAG) system built on
[Claude](https://www.anthropic.com/claude) for generation and local embeddings
for retrieval.

Ask *"What is the role of amyloid-beta in Alzheimer's disease?"* and get a
grounded, cited answer drawn from the latest literature — not the model's
memory.

## Why it's different

- **Dynamic, always up to date.** Queries PubMed live via the NCBI E-utilities
  API. A paper published yesterday is answerable today — no static snapshot.
- **Legal full text.** Pulls open-access full text from **PubMed Central** and
  finds legal open-access links via **Unpaywall**. Falls back to abstracts
  (always free). It deliberately does **not** scrape paywalled content.
- **Cited and verifiable.** Every claim carries an inline `[n]` citation linking
  to the PubMed record, so answers are auditable.
- **Runs free and offline by default.** Answer generation uses a local model via
  [Ollama](https://ollama.com) out of the box — no API key, no per-question cost.
  Optionally switch to Claude (one config line) for higher quality when serving a
  public site.
- **Caches as it goes.** Papers are embedded once and persisted in a local
  vector store; popular topics get faster on repeat.

## Architecture

```
            ┌─────────────┐
  question  │   PubMed    │  esearch → relevant PMIDs
 ──────────▶│ E-utilities │  efetch  → abstracts + metadata + PMCIDs
            └──────┬──────┘
                   │
                   ▼
            ┌─────────────┐  PubMed Central open-access full text
            │  full-text  │  Unpaywall open-access link
            │  enrichment │  (abstract fallback)
            └──────┬──────┘
                   ▼
            ┌─────────────┐  chunk → embed (sentence-transformers)
            │  ChromaDB   │  persistent vector store + cache
            └──────┬──────┘
                   │  top-k similar chunks (cosine)
                   ▼
            ┌─────────────┐
            │   Claude    │  grounded answer with [n] citations
            └─────────────┘
```

| Module | Responsibility |
|---|---|
| `medrag/pubmed.py` | E-utilities search + fetch, XML parsing |
| `medrag/fulltext.py` | PMC full text + Unpaywall OA links |
| `medrag/chunking.py` | Word-based chunking with overlap |
| `medrag/embeddings.py` | Local sentence-transformer embeddings |
| `medrag/vectorstore.py` | ChromaDB persistence + paper-level cache |
| `medrag/llm.py` | Claude answer generation (streamed, cited) |
| `medrag/rag.py` | Orchestration |
| `app.py` | Streamlit UI |
| `server/main.py` | FastAPI web backend (custom site) |
| `server/limits.py` | Cost-control rate limiting |
| `web/` | Front end (HTML/CSS/JS) |
| `scripts/demo.py` | CLI demo / smoke test |

## Setup

Requires Python 3.10+.

```bash
# 1. Install Python dependencies (first run downloads a small embedding model)
pip install -r requirements.txt

# 2. Install a local model with Ollama (free, no API key) — the default backend
#    Install Ollama from https://ollama.com, then:
ollama pull llama3.1

# 3. Configure
cp .env.example .env          # then edit .env
#   - CONTACT_EMAIL      (recommended; required for Unpaywall)
#   - NCBI_API_KEY       (optional; raises PubMed rate limit)
```

### Answer-generation backend

By default MedRAG generates answers with a **local Ollama model** — free,
offline, no API key. To use **Claude** instead (higher quality; needed to serve
a public website), set in `.env`:

```bash
MEDRAG_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

**Web app:**

```bash
streamlit run app.py
```

**Command line:**

```bash
python scripts/demo.py "What is the role of tau protein in neurodegeneration?"
```

**Website (FastAPI backend + front end):**

```bash
uvicorn server.main:app --reload
# open http://127.0.0.1:8000
```

The web backend holds the API key server-side — visitors never log in or supply
a key (they use the site's single account). Usage is bounded by:

- **Per-session / global / cooldown limits** in `config.py`
  (`PUBLIC_MAX_QUESTIONS_PER_SESSION`, `PUBLIC_MAX_QUESTIONS_PER_DAY_GLOBAL`,
  `PUBLIC_COOLDOWN_SECONDS`), enforced in `server/limits.py`.
- A **monthly spend cap** set in the [Anthropic Console](https://console.anthropic.com/)
  (Settings → Limits) — the ultimate backstop.

The public default model is `claude-sonnet-4-6` (see `config.py`); set
`MEDRAG_MODEL=claude-opus-4-8` in `.env` for higher-quality local testing.

## Deploying the public site

The repo includes a `render.yaml` blueprint and a `Procfile`. The deployed site
uses the **Groq** free-tier backend (`MEDRAG_PROVIDER=groq`) — local Ollama can't
serve a public URL. Steps:

1. Push to GitHub.
2. On [render.com](https://render.com): **New → Blueprint** → pick this repo.
3. Set secrets in the dashboard: `GROQ_API_KEY` (free, from
   [console.groq.com](https://console.groq.com/keys)) and `CONTACT_EMAIL`.

> **Note:** the embedding stack (PyTorch + sentence-transformers) is memory-heavy.
> A free 512 MB instance may be tight; if it runs out of memory, use a small paid
> instance or switch the embeddings to an ONNX/CPU-light backend.

## A note on data sources

MedRAG uses only freely and legally available content:

- **Abstracts** — free for every PubMed record.
- **PubMed Central open-access subset** — full text where authors/funders have
  made it openly available (most NIH-funded work qualifies).
- **Unpaywall** — a database of *legal* open-access versions (author
  manuscripts, preprints).

Paywalled full text is never scraped. Where only an abstract is available,
that's what's used, and the source is linked so you can read further.

## Roadmap

- Reranking of retrieved chunks for higher precision
- bioRxiv/medRxiv preprint support
- A small evaluation harness for answer quality
- Conversation memory for follow-up questions
