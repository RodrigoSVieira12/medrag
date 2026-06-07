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
# Public-site default is Sonnet 4.6 (good quality, ~6x cheaper than Opus).
# For your own local testing, override with MEDRAG_MODEL=claude-opus-4-8 in .env.
MODEL = os.getenv("MEDRAG_MODEL", "claude-sonnet-4-6")
# Caps the size (and therefore cost) of any single answer. A grounded, cited
# RAG answer rarely needs more than this; raise it only if answers get cut off.
ANSWER_MAX_TOKENS = 1024

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
