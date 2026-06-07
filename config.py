"""Central configuration for MedRAG.

All tunable knobs live here so the rest of the codebase stays declarative.
Values are read from the environment (optionally via a local .env file).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / ".chroma"
DATA_DIR.mkdir(exist_ok=True)

# --- Secrets / contact ---------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# --- LLM -----------------------------------------------------------------
# Which engine generates answers:
#   "ollama"    -> a model running locally on your machine. Free, no API key,
#                  offline. Default for local dev. Can't serve a public site.
#   "groq"      -> Llama hosted on Groq's free tier. Free, fast, no GPU to host.
#                  Good for the public website. Rate-limited; needs GROQ_API_KEY.
#   "anthropic" -> Claude via API. Highest quality; costs ~cents/question and
#                  requires ANTHROPIC_API_KEY.
LLM_PROVIDER = os.getenv("MEDRAG_PROVIDER", "ollama")

# Ollama (local) settings.
OLLAMA_MODEL = os.getenv("MEDRAG_OLLAMA_MODEL", "llama3.1")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Groq (hosted, free tier) settings — used only when LLM_PROVIDER="groq".
# Free API key: https://console.groq.com/keys . If the model name ever changes,
# pick a current one from the Groq console and set MEDRAG_GROQ_MODEL.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("MEDRAG_GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE = "https://api.groq.com/openai/v1"

# Anthropic (Claude) settings — used only when LLM_PROVIDER="anthropic".
ANTHROPIC_MODEL = os.getenv("MEDRAG_MODEL", "claude-sonnet-4-6")

# Caps the length (and, for paid providers, the cost) of any single answer.
ANSWER_MAX_TOKENS = 1024


def active_model() -> str:
    """The model name in use for the current provider (for display)."""
    return {
        "ollama": OLLAMA_MODEL,
        "groq": GROQ_MODEL,
        "anthropic": ANTHROPIC_MODEL,
    }.get(LLM_PROVIDER, LLM_PROVIDER)

# --- Retrieval / embeddings ---------------------------------------------
# all-MiniLM-L6-v2: small (~80MB), fast, strong general-purpose embeddings.
EMBEDDING_MODEL = os.getenv("MEDRAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "medrag"

# Chunking (word-based, with overlap to preserve context across boundaries).
CHUNK_SIZE_WORDS = 250
CHUNK_OVERLAP_WORDS = 50

# Defaults for a single question (overridable in the UI).
DEFAULT_MAX_PAPERS = 8     # how many PubMed hits to fetch + index per query
DEFAULT_TOP_K = 6          # how many chunks to feed the LLM as context

# --- Public-demo limits (Option A: you pay, with hard caps) ---------------
# These bound how much visitors can spend on your behalf. They are ENFORCED BY
# THE BACKEND (the FastAPI layer we'll build for the website) — defined here now
# so the limits live in one place and are ready to wire up. They do NOT replace
# the monthly spend cap you set in the Anthropic Console (your ultimate backstop).
PUBLIC_MAX_QUESTIONS_PER_SESSION = 5      # per visitor, per browser session
PUBLIC_MAX_QUESTIONS_PER_DAY_GLOBAL = 100  # site-wide daily ceiling across all users
PUBLIC_COOLDOWN_SECONDS = 5                # min seconds between a visitor's questions

# --- PubMed / E-utilities ------------------------------------------------
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
PUBMED_TOOL_NAME = "medrag"
