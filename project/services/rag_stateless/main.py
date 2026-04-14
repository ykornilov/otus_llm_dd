"""
Stateless RAG service — FastAPI, port 8002.

The embedding model and BM25 index are loaded once at startup.
The FAISS index is rebuilt from scratch on EVERY /retrieve request —
this is the key difference vs the stateful service.

Usage:
    cd project/
    EMBEDDER=bge-m3 uvicorn services.rag_stateless.main:app --port 8002

Retrieval modes (pass via request body):
    alpha=1.0   — pure dense (FAISS cosine similarity)
    alpha=0.0   — pure BM25 (lexical)
    alpha=0.5   — hybrid via Reciprocal Rank Fusion (default)
"""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (one level above project/)
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent.parent / ".env", override=True)

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.common.chunker import get_chunks
from services.common.embedder import Embedder
from services.common.tokenizer import tokenize as _tokenize

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDER_NAME = os.environ.get("EMBEDDER", "multilingual-e5-large")
TOP_K_DEFAULT = 6
RRF_K         = 60
BM25_FETCH    = 50


# ── Startup — load model + chunks + BM25, NO dense index ─────────────────────
print(f"[stateless] Initialising with embedder='{EMBEDDER_NAME}'")

embedder = Embedder(EMBEDDER_NAME)
chunks = get_chunks()
texts = [c["text"] for c in chunks]

print("[stateless] Building BM25 index …")
tokenized_corpus = [_tokenize(t) for t in texts]
bm25 = BM25Okapi(tokenized_corpus)

print(f"[stateless] Model ready — {len(chunks)} chunks loaded, FAISS index built per request")

# ── Schemas ───────────────────────────────────────────────────────────────────
class RetrieveRequest(BaseModel):
    query: str
    k:     int   = Field(default=TOP_K_DEFAULT, ge=1, le=50)
    alpha: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Blend weight: 1.0 = pure dense, 0.0 = pure BM25, 0.5 = hybrid RRF",
    )


class RetrieveResponse(BaseModel):
    chunks:         list[str]
    titles:         list[str]
    scores:         list[float]
    index_build_ms: float   # overhead of building the FAISS index per request


# ── Retrieval helpers ─────────────────────────────────────────────────────────
def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


def hybrid_retrieve(query: str, k: int, alpha: float, index):
    fetch_n = max(k * 4, BM25_FETCH)

    if alpha >= 1.0:
        q_emb = embedder.embed_query(query).reshape(1, -1)
        scores, indices = index.search(q_emb, k)
        return indices[0].tolist(), scores[0].tolist()

    if alpha <= 0.0:
        bm25_scores = bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(bm25_scores)[::-1][:k]
        return top_idx.tolist(), bm25_scores[top_idx].tolist()

    # Hybrid RRF
    q_emb = embedder.embed_query(query).reshape(1, -1)
    _, dense_indices = index.search(q_emb, fetch_n)
    dense_ranks = {int(idx): rank for rank, idx in enumerate(dense_indices[0])}

    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_top = np.argsort(bm25_scores)[::-1][:fetch_n]
    bm25_ranks = {int(idx): rank for rank, idx in enumerate(bm25_top)}

    all_ids = set(dense_ranks) | set(bm25_ranks)
    fused: dict[int, float] = {}
    for doc_id in all_ids:
        score = 0.0
        if doc_id in dense_ranks:
            score += alpha * _rrf_score(dense_ranks[doc_id])
        if doc_id in bm25_ranks:
            score += (1 - alpha) * _rrf_score(bm25_ranks[doc_id])
        fused[doc_id] = score

    top_k = sorted(fused, key=fused.__getitem__, reverse=True)[:k]
    return top_k, [fused[i] for i in top_k]


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="DataLens RAG — Stateless", version="2.0")


@app.get("/info")
def info():
    return {
        "mode":      "stateless",
        "embedder":  EMBEDDER_NAME,
        "model":     embedder.model_name,
        "n_chunks":  len(chunks),
        "dim":       embedder.dim,
        "bm25_docs": len(tokenized_corpus),
    }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    t0 = time.perf_counter()

    # Rebuild FAISS index from scratch on every request
    doc_embeddings = embedder.embed_docs(texts, batch_size=64).astype(np.float32)
    index = faiss.IndexFlatIP(embedder.dim)
    index.add(doc_embeddings)

    index_build_ms = round((time.perf_counter() - t0) * 1000, 1)

    final_indices, final_scores = hybrid_retrieve(req.query, req.k, req.alpha, index)

    return RetrieveResponse(
        chunks=[chunks[i]["text"] for i in final_indices],
        titles=[chunks[i]["title"] for i in final_indices],
        scores=final_scores,
        index_build_ms=index_build_ms,
    )
