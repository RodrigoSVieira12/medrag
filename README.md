# MedRAG: Biomedical Literature Q&A Assistant

MedRAG answers biomedical questions by retrieving and reasoning over real
scientific papers from **PubMed**, in real time, with **inline citations** back
to every source. It is a Retrieval-Augmented Generation (RAG) system: a language
model writes the answer, but only from passages retrieved from the literature.

Ask *"What is the role of amyloid-beta in Alzheimer's disease?"* and get a
grounded, cited answer drawn from the latest literature rather than the model's
memory.

## Why it's different

- **Dynamic, always up to date.** Queries PubMed live via the NCBI E-utilities
  API, so a paper published yesterday is answerable today. There is no static
  snapshot.
- **Legal full text.** Pulls open-access full text from **PubMed Central** and
  finds legal open-access links via **Unpaywall**, falling back to abstracts
  (always free). It deliberately does **not** scrape paywalled content.
- **Cited and verifiable.** Every claim carries an inline `[n]` citation linking
  to the PubMed record, so answers are auditable.
- **Runs free and offline by default.** Answer generation uses a local model via
  [Ollama](https://ollama.com) out of the box, with no API key and no
  per-question cost. A hosted backend (Groq or Claude) can be enabled with one
  config line for a public deployment.
- **Caches as it goes.** Papers are embedded once and persisted in a local
  vector store, so repeated topics get faster over time.

## Architecture

```
            ┌─────────────┐
  question  │   PubMed    │  esearch: relevant PMIDs
 ─────────▶ │ E-utilities │  efetch: abstracts, metadata, PMCIDs
            └──────┬──────┘
                   │
                   ▼
            ┌─────────────┐  PubMed Central open-access full text
            │  full-text  │  Unpaywall open-access link
            │  enrichment │  (abstract fallback)
            └──────┬──────┘
                   ▼
            ┌─────────────┐  chunk, then embed (ONNX MiniLM)
            │  ChromaDB   │  persistent vector store and cache
            └──────┬──────┘
                   │  top-k similar chunks (cosine)
                   ▼
            ┌─────────────┐
            │     LLM     │  grounded answer with [n] citations
            │ Ollama/Groq │  (provider is configurable)
            │  or Claude  │
            └─────────────┘
```

| Module | Responsibility |
|---|---|
| `medrag/pubmed.py` | E-utilities search and fetch, query cleaning, XML parsing |
| `medrag/fulltext.py` | PMC full text and Unpaywall open-access links |
| `medrag/chunking.py` | Word-based chunking with overlap |
| `medrag/embeddings.py` | ONNX MiniLM embeddings (via onnxruntime) |
| `medrag/vectorstore.py` | ChromaDB persistence and paper-level cache |
| `medrag/llm.py` | Answer generation (Ollama, Groq, or Claude), streamed and cited |
| `medrag/rag.py` | Orchestration |
| `app.py` | Streamlit UI |
| `server/main.py` | FastAPI web backend (custom site) |
| `server/limits.py` | Cost-control rate limiting |
| `web/` | Front end (HTML, CSS, JS) |
| `scripts/demo.py` | CLI demo and smoke test |

## Setup

Requires Python 3.10+.

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install a local model with Ollama (free, no API key); the default backend.
#    Install Ollama from https://ollama.com, then:
ollama pull llama3.1

# 3. Configure (optional for local Ollama use)
cp .env.example .env
#   CONTACT_EMAIL   recommended, and required for Unpaywall links
#   NCBI_API_KEY    optional, raises the PubMed rate limit
```

### Answer-generation backend

The backend is selected by `MEDRAG_PROVIDER` in `.env`:

- `ollama` (default): a model running locally. Free, offline, no API key.
- `groq`: Llama hosted on Groq's free tier. Good for a public website.
  Set `GROQ_API_KEY` (free from [console.groq.com](https://console.groq.com/keys)).
- `anthropic`: Claude via API. Highest quality. Set `ANTHROPIC_API_KEY`.

## Usage

**Web app (Streamlit):**

```bash
streamlit run app.py
```

**Command line:**

```bash
python scripts/demo.py "What is the role of tau protein in neurodegeneration?"
```

**Website (FastAPI backend and front end):**

```bash
uvicorn server.main:app --reload
# open http://127.0.0.1:8000
```

The web backend holds any API key server-side, so visitors never log in or
supply a key of their own. Usage is bounded by:

- **Per-session, global, and cooldown limits** in `config.py`
  (`PUBLIC_MAX_QUESTIONS_PER_SESSION`, `PUBLIC_MAX_QUESTIONS_PER_DAY_GLOBAL`,
  `PUBLIC_COOLDOWN_SECONDS`), enforced in `server/limits.py`.
- For the Claude backend, a **monthly spend cap** set in the
  [Anthropic Console](https://console.anthropic.com/) (Settings, then Limits).

## Deploying the public site

The repo includes a `render.yaml` blueprint and a `Procfile`. The deployed site
uses the **Groq** free-tier backend (`MEDRAG_PROVIDER=groq`), because a local
Ollama model cannot serve a public URL. Steps:

1. Push to GitHub.
2. On [render.com](https://render.com), choose **New**, then **Blueprint**, and
   pick this repo.
3. Set the secrets in the dashboard: `GROQ_API_KEY` (free, from
   [console.groq.com](https://console.groq.com/keys)) and `CONTACT_EMAIL`.

Embeddings run on a lightweight ONNX model rather than PyTorch, so the app fits
on small and free instances. The MiniLM model (about 80 MB) downloads on the
first request, which adds a little latency to the very first query after a
deploy.

## A note on data sources

MedRAG uses only freely and legally available content:

- **Abstracts.** Free for every PubMed record.
- **PubMed Central open-access subset.** Full text where authors or funders have
  made it openly available (most NIH-funded work qualifies).
- **Unpaywall.** A database of *legal* open-access versions, such as author
  manuscripts and preprints.

Paywalled full text is never scraped. Where only an abstract is available, that
is what is used, and the source is linked so you can read further.

## Roadmap

- Reranking of retrieved chunks for higher precision
- bioRxiv and medRxiv preprint support
- A small evaluation harness for answer quality
- Conversation memory for follow-up questions
