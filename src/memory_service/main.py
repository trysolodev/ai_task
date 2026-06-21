"""HTTP entrypoint implementing the §3 contract.

Endpoints are synchronous (`def`) so FastAPI runs them in a worker threadpool —
simpler and safer than async around CPU-bound embedding and blocking DB calls,
and plenty for the "a few concurrent sessions" the eval exercises. The backing
store is ACID, so once `/turns` returns, every read path sees the data.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from . import __version__, db
from .config import settings
from .embeddings import build_embedder
from .extraction import build_extractor
from .memory_store import MemoryStore, build_turn_text, parse_timestamp
from .models import (
    MemoriesResponse,
    RecallRequest,
    RecallResponse,
    SearchRequest,
    SearchResponse,
    TurnRequest,
    TurnResponse,
)
from .recall import RecallEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("memory.api")

MAX_BODY_BYTES = 16 * 1024 * 1024  # 16 MB; reject oversized payloads with 413

# Built during lifespan startup.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting memory-service v%s", __version__)
    db.init_db(settings.database_url, settings.embed_dim)
    embedder = build_embedder(
        settings.embed_provider,
        settings.embed_dim,
        openai_key=settings.openai_api_key,
        voyage_key=settings.voyage_api_key,
    )
    state["embedder"] = embedder
    state["store"] = MemoryStore(embedder)
    state["recall"] = RecallEngine(embedder)
    state["extractor"] = build_extractor(settings)
    state["ready"] = True
    logger.info(
        "ready | extraction=%s embed=%s dim=%d auth=%s",
        state["extractor"].mode, settings.embed_provider, settings.embed_dim,
        "on" if settings.auth_enabled else "off",
    )
    try:
        yield
    finally:
        db.close_db()


app = FastAPI(title="memory-service", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def _guard_body_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > MAX_BODY_BYTES:
                return JSONResponse(status_code=413, content={"detail": "payload too large"})
        except ValueError:
            pass
    return await call_next(request)


def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not settings.auth_enabled:
        return
    if authorization != f"Bearer {settings.auth_token}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> JSONResponse:
    ok = bool(state.get("ready")) and db.healthy()
    body = {
        "status": "ok" if ok else "starting",
        "version": __version__,
        "extraction": state.get("extractor").mode if state.get("extractor") else None,
        "embeddings": settings.embed_provider,
    }
    return JSONResponse(status_code=200 if ok else 503, content=body)


@app.get("/")
def root() -> dict:
    return {"service": "memory-service", "version": __version__}


@app.post("/turns", status_code=201, response_model=TurnResponse, dependencies=[Depends(require_auth)])
def post_turn(req: TurnRequest) -> TurnResponse:
    store: MemoryStore = state["store"]
    msgs = [(m.role, m.content, m.name) for m in req.messages]
    content = build_turn_text(msgs)
    ts = parse_timestamp(req.timestamp)

    # 1) Persist the raw turn (committed before extraction).
    turn_id = store.store_turn(
        session_id=req.session_id,
        user_id=req.user_id,
        messages_json=[m.model_dump() for m in req.messages],
        content=content,
        metadata=req.metadata,
        ts=ts,
    )

    # 2) Extract structured memories and apply supersession. Best-effort: a
    #    failure here must not lose the turn we already stored.
    try:
        known = store.known_active_facts(req.user_id, req.session_id)
        extracted = state["extractor"].extract(msgs, known)
        if extracted:
            store.apply_extracted(
                extracted, user_id=req.user_id, session_id=req.session_id,
                turn_id=turn_id, ts=ts,
            )
    except Exception:  # pragma: no cover - extraction must never 500 a write
        logger.exception("extraction failed for turn %s (turn still stored)", turn_id)

    return TurnResponse(id=turn_id)


@app.post("/recall", response_model=RecallResponse, dependencies=[Depends(require_auth)])
def post_recall(req: RecallRequest) -> RecallResponse:
    engine: RecallEngine = state["recall"]
    try:
        result = engine.recall(req.query, req.user_id, req.session_id, req.max_tokens)
    except Exception:  # never error on a recall; degrade to empty context
        logger.exception("recall failed; returning empty context")
        return RecallResponse(context="", citations=[])
    return RecallResponse(**result)


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(require_auth)])
def post_search(req: SearchRequest) -> SearchResponse:
    engine: RecallEngine = state["recall"]
    try:
        results = engine.search(req.query, req.session_id, req.user_id, req.limit)
    except Exception:
        logger.exception("search failed; returning no results")
        results = []
    return SearchResponse(results=results)


@app.get(
    "/users/{user_id}/memories",
    response_model=MemoriesResponse,
    dependencies=[Depends(require_auth)],
)
def get_user_memories(user_id: str) -> MemoriesResponse:
    store: MemoryStore = state["store"]
    return MemoriesResponse(memories=store.list_memories(user_id))


@app.delete("/sessions/{session_id}", status_code=204, dependencies=[Depends(require_auth)])
def delete_session(session_id: str) -> Response:
    state["store"].delete_session(session_id)
    return Response(status_code=204)


@app.delete("/users/{user_id}", status_code=204, dependencies=[Depends(require_auth)])
def delete_user(user_id: str) -> Response:
    state["store"].delete_user(user_id)
    return Response(status_code=204)
