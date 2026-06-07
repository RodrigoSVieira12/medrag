"""Legal open-access full-text enrichment.

Order of preference for each article:
  1. PubMed Central open-access full text (via E-utilities) — real body text.
  2. Unpaywall — a legal open-access URL (author manuscript / publisher OA),
     surfaced as a link for the user. We do NOT scrape paywalled PDFs.
  3. Abstract only (always free) — the fallback.

Deliberately excludes Sci-Hub and any source that distributes paywalled
content without permission.
"""

from __future__ import annotations

import requests

import config
from .pubmed import Article, PubMedClient


def enrich(article: Article, client: PubMedClient) -> Article:
    """Mutate `article` in place with full text + an open-access link, then return it."""
    # 1. PMC open-access full text (best: real body text we can index).
    if article.pmcid:
        body = client.fetch_pmc_fulltext(article.pmcid)
        if body:
            article.full_text = body
            article.oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{article.pmcid}/"

    # 2. Unpaywall — find a legal OA link when we don't already have one.
    if not article.oa_url and article.doi and config.CONTACT_EMAIL:
        article.oa_url = _unpaywall_oa_url(article.doi) or ""

    # 3. (abstract is already present; Article.content falls back to it)
    return article


def _unpaywall_oa_url(doi: str) -> str | None:
    """Return the best legal open-access URL for a DOI, or None."""
    try:
        resp = requests.get(
            f"{config.UNPAYWALL_BASE}/{doi}",
            params={"email": config.CONTACT_EMAIL},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None
    best = data.get("best_oa_location") or {}
    return best.get("url_for_pdf") or best.get("url")
