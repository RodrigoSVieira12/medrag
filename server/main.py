"""MedRAG web backend.

The browser talks only to this server; this server holds the Anthropic API key
and calls Claude. Visitors never see a key and never log in (Option A billing).
Every request passes through the rate limiter before any tokens are spent.

Run locally:
    uvicorn server.main:app --reload
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from medrag.rag import MedRAG
from server.limits import RateLimiter

COOKIE = "medrag_session"
WEB_DIR = config.ROOT / "web"

app = FastAPI(title="MedRAG")
limiter = RateLimiter()


@lru_cache(maxsize=1)
def get_engine() -> MedRAG:
    # Lazy singleton: built on first request so the server starts instantly
    # and the embedding model only loads when actually needed.
    return MedRAG()


class AskRequest(BaseModel):
    question: str
    max_papers: int | None = None
    top_k: int | None = None


def _ndjson(obj: dict) -> str:
    return json.dumps(obj) + "\n"


def _serialize_sources(sources: list[dict]) -> list[dict]:
    """De-duplicate by citation number for the client."""
    seen: dict[int, dict] = {}
    for src in sources:
        meta = src["metadata"]
        seen.setdefault(
            src["n"],
            {
                "n": src["n"],
                "title": meta.get("title", "Untitled"),
                "citation": meta.get("citation", ""),
                "url": meta.get("url", ""),
                "oa_url": meta.get("oa_url", ""),
            },
        )
    return [seen[n] for n in sorted(seen)]


def _stream_answer(question: str, max_papers: int, top_k: int, remaining: int) -> Iterator[str]:
    """Sync generator yielding newline-delimited JSON events to the browser."""
    engine = get_engine()
    try:
        pmids = engine.index_query(question, max_papers=max_papers)
        if not pmids:
            yield _ndjson({"type": "error", "message": "No PubMed results found for that question."})
            return
        sources = engine.retrieve(question, pmids, top_k=top_k)
        yield _ndjson({"type": "sources", "sources": _serialize_sources(sources), "remaining": remaining})
        for delta in engine.answer_stream(question, sources):
            yield _ndjson({"type": "delta", "text": delta})
        yield _ndjson({"type": "done"})
    except RuntimeError as exc:
        # e.g. missing API key — surface the message cleanly.
        yield _ndjson({"type": "error", "message": str(exc)})
    except Exception as exc:  # noqa: BLE001 - last-resort guard for the stream
        yield _ndjson({"type": "error", "message": f"Something went wrong: {exc}"})


def _session_id(request: Request) -> str:
    return request.cookies.get(COOKIE) or uuid.uuid4().hex


@app.get("/api/status")
async def status(request: Request) -> JSONResponse:
    sid = _session_id(request)
    resp = JSONResponse({"remaining": limiter.remaining(sid)})
    resp.set_cookie(COOKIE, sid, httponly=True, samesite="lax", max_age=86400)
    return resp


@app.post("/api/ask")
async def ask(payload: AskRequest, request: Request):
    sid = _session_id(request)

    question = payload.question.strip()
    if not question:
        return JSONResponse({"error": "Please enter a question."}, status_code=400)

    # Enforce cost-control limits BEFORE doing any paid work.
    decision = limiter.check(sid)
    if not decision.allowed:
        resp = JSONResponse(
            {"error": decision.reason, "remaining": decision.remaining_session},
            status_code=429,
        )
        resp.set_cookie(COOKIE, sid, httponly=True, samesite="lax", max_age=86400)
        return resp

    # Clamp request parameters so a crafted request can't blow up cost/latency.
    max_papers = min(max(payload.max_papers or config.DEFAULT_MAX_PAPERS, 1), 20)
    top_k = min(max(payload.top_k or config.DEFAULT_TOP_K, 1), 12)

    resp = StreamingResponse(
        _stream_answer(question, max_papers, top_k, decision.remaining_session),
        media_type="application/x-ndjson",
    )
    resp.set_cookie(COOKIE, sid, httponly=True, samesite="lax", max_age=86400)
    return resp


# Serve the static front end. Mounted last so /api/* routes take precedence.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
