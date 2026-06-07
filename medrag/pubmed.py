"""PubMed access via NCBI E-utilities.

Searches PubMed in real time (esearch) and fetches structured article
metadata + abstracts (efetch). Everything here uses only freely available
endpoints; abstracts are always free, and we surface PMCIDs so the
full-text layer can pull open-access bodies where they exist.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests

import config

# Conversational filler + generic research words. PubMed ANDs every term, so
# leaving these in ("can you find sources for ...") can drive results to zero.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with",
    "about", "regarding", "concerning", "from", "by", "as", "at", "into",
    "i", "you", "we", "they", "it", "me", "my", "our", "your",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "do", "does", "did", "can",
    "could", "would", "should", "will", "shall", "may", "might", "please",
    "find", "search", "show", "give", "tell", "get", "need", "want", "looking",
    "look", "source", "sources", "paper", "papers", "article", "articles",
    "study", "studies", "research", "reference", "references", "information",
    "info", "literature", "any", "some", "all", "more", "most", "related", "list",
}


def clean_query(text: str) -> str:
    """Reduce a natural-language question to PubMed-friendly keyword terms.

    Strips punctuation and conversational stopwords. If that leaves nothing
    (e.g. a query made only of stopwords), returns the original text.
    """
    cleaned = re.sub(r"[^\w\s-]", " ", text)
    tokens = [t for t in cleaned.split() if t.lower() not in _STOPWORDS]
    return " ".join(tokens).strip() or text.strip()


@dataclass
class Article:
    """Structured PubMed record. `full_text` is filled in later (see fulltext.py)."""

    pmid: str
    title: str = ""
    abstract: str = ""
    journal: str = ""
    year: str = ""
    authors: list[str] = field(default_factory=list)
    doi: str = ""
    pmcid: str = ""          # e.g. "PMC1234567" if an open-access full text exists
    full_text: str = ""      # populated from PMC when available
    oa_url: str = ""         # legal open-access link (Unpaywall / PMC)

    @property
    def citation(self) -> str:
        first_author = self.authors[0] if self.authors else "Unknown"
        et_al = " et al" if len(self.authors) > 1 else ""
        return f"{first_author}{et_al}. {self.journal} ({self.year}). PMID:{self.pmid}"

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    @property
    def content(self) -> str:
        """Best available text for indexing: full text if we have it, else abstract."""
        return self.full_text or self.abstract


class PubMedClient:
    """Thin, polite wrapper over E-utilities with built-in rate limiting."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"{config.PUBMED_TOOL_NAME}/0.1"})
        # NCBI allows 3 req/s without a key, 10 req/s with one.
        self._min_interval = 0.11 if config.NCBI_API_KEY else 0.34
        self._last_request = 0.0

    # -- internals --------------------------------------------------------
    def _params(self, **extra: str) -> dict:
        params = {"tool": config.PUBMED_TOOL_NAME, "db": "pubmed"}
        if config.CONTACT_EMAIL:
            params["email"] = config.CONTACT_EMAIL
        if config.NCBI_API_KEY:
            params["api_key"] = config.NCBI_API_KEY
        params.update(extra)
        return params

    def _get(self, endpoint: str, params: dict) -> requests.Response:
        # Simple client-side throttle to respect NCBI rate limits.
        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        resp = self.session.get(
            f"{config.EUTILS_BASE}/{endpoint}", params=params, timeout=30
        )
        self._last_request = time.monotonic()
        resp.raise_for_status()
        return resp

    # -- public API -------------------------------------------------------
    def search(self, query: str, max_results: int = 10) -> list[str]:
        """Return PMIDs most relevant to `query`.

        Natural-language questions are reduced to keyword terms first (PubMed
        ANDs every word, so filler kills recall). Falls back to the raw query
        if the cleaned one finds nothing.
        """
        terms = clean_query(query)
        pmids = self._esearch(terms, max_results)
        if not pmids and terms.lower() != query.strip().lower():
            pmids = self._esearch(query, max_results)
        return pmids

    def _esearch(self, term: str, max_results: int) -> list[str]:
        params = self._params(
            term=term, retmax=str(max_results), retmode="json", sort="relevance"
        )
        data = self._get("esearch.fcgi", params).json()
        return data.get("esearchresult", {}).get("idlist", [])

    def fetch(self, pmids: list[str]) -> list[Article]:
        """Fetch full metadata + abstracts for a batch of PMIDs."""
        if not pmids:
            return []
        params = self._params(id=",".join(pmids), retmode="xml", rettype="abstract")
        xml = self._get("efetch.fcgi", params).text
        return self._parse_articles(xml)

    def fetch_pmc_fulltext(self, pmcid: str) -> str:
        """Fetch open-access full-text body from PubMed Central (JATS XML).

        Only articles in the PMC open-access subset return a usable body;
        others 403/return empty, in which case we fall back to the abstract.
        """
        numeric = pmcid.replace("PMC", "")
        params = {
            "tool": config.PUBMED_TOOL_NAME,
            "db": "pmc",
            "id": numeric,
            "retmode": "xml",
        }
        if config.CONTACT_EMAIL:
            params["email"] = config.CONTACT_EMAIL
        if config.NCBI_API_KEY:
            params["api_key"] = config.NCBI_API_KEY
        try:
            xml = self._get("efetch.fcgi", params).text
        except requests.HTTPError:
            return ""
        return self._parse_pmc_body(xml)

    # -- parsing ----------------------------------------------------------
    @staticmethod
    def _text(el: ET.Element | None) -> str:
        if el is None:
            return ""
        return " ".join("".join(el.itertext()).split())

    def _parse_articles(self, xml: str) -> list[Article]:
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return []

        articles: list[Article] = []
        for node in root.findall(".//PubmedArticle"):
            medline = node.find("MedlineCitation")
            if medline is None:
                continue
            pmid = self._text(medline.find("PMID"))
            art = medline.find("Article")
            if art is None or not pmid:
                continue

            # Abstracts can be split into labeled sections; join them.
            abstract_parts = []
            for ab in art.findall(".//Abstract/AbstractText"):
                label = ab.get("Label")
                text = self._text(ab)
                abstract_parts.append(f"{label}: {text}" if label else text)

            # Publication year (PubDate may use Year or MedlineDate).
            year = self._text(art.find(".//Journal/JournalIssue/PubDate/Year"))
            if not year:
                medline_date = self._text(art.find(".//Journal/JournalIssue/PubDate/MedlineDate"))
                year = medline_date.split(" ")[0] if medline_date else ""

            authors = []
            for author in art.findall(".//AuthorList/Author"):
                last = self._text(author.find("LastName"))
                initials = self._text(author.find("Initials"))
                if last:
                    authors.append(f"{last} {initials}".strip())

            # IDs (DOI + PMCID) live under PubmedData/ArticleIdList.
            doi, pmcid = "", ""
            for aid in node.findall(".//PubmedData/ArticleIdList/ArticleId"):
                id_type = aid.get("IdType")
                if id_type == "doi":
                    doi = self._text(aid)
                elif id_type == "pmc":
                    pmcid = self._text(aid)

            articles.append(
                Article(
                    pmid=pmid,
                    title=self._text(art.find("ArticleTitle")),
                    abstract=" ".join(abstract_parts),
                    journal=self._text(art.find(".//Journal/Title")),
                    year=year,
                    authors=authors,
                    doi=doi,
                    pmcid=pmcid,
                )
            )
        return articles

    def _parse_pmc_body(self, xml: str) -> str:
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return ""
        # JATS body paragraphs. Skip references/tables for cleaner context.
        paragraphs = []
        for body in root.findall(".//body"):
            for p in body.findall(".//p"):
                text = self._text(p)
                if len(text) > 40:  # drop tiny fragments
                    paragraphs.append(text)
        return "\n\n".join(paragraphs)
