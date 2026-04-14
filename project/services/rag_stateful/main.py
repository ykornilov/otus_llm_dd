"""
Stateful RAG service — FastAPI, port 8001.

RAG index is built ONCE at startup and kept in memory.
Restart the service to switch embedder.

Usage:
    cd project/
    EMBEDDER=bge-m3 uvicorn services.rag_stateful.main:app --port 8001
    EMBEDDER=multilingual-e5-large uvicorn services.rag_stateful.main:app --port 8001

    # qodo via vLLM (start vLLM first on port 8003, then):
    EMBEDDER=qodo-1.5b VLLM_URL=http://localhost:8003 \\
        uvicorn services.rag_stateful.main:app --port 8001

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
from services.common.embedder import Embedder, VLLMEmbedder
from services.common.tokenizer import tokenize as _tokenize

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDER_NAME = os.environ.get("EMBEDDER", "multilingual-e5-large")
VLLM_URL      = os.environ.get("VLLM_URL", "")
TOP_K_DEFAULT = 6
RRF_K         = 60          # RRF constant — higher = smoother blend
BM25_FETCH    = 50          # how many candidates to pull from each retriever before fusion


# ── Startup — build indexes once ─────────────────────────────────────────────
print(f"[stateful] Initialising with embedder='{EMBEDDER_NAME}'" +
      (f" via vLLM at {VLLM_URL}" if VLLM_URL else ""))
t_start = time.perf_counter()

embedder = VLLMEmbedder(EMBEDDER_NAME, VLLM_URL) if VLLM_URL else Embedder(EMBEDDER_NAME)
chunks = get_chunks()
texts = [c["text"] for c in chunks]

# Dense index
print(f"[stateful] Embedding {len(texts)} chunks …")
doc_embeddings = embedder.embed_docs(texts)
index = faiss.IndexFlatIP(embedder.dim)   # cosine similarity on L2-normalised vectors
index.add(doc_embeddings)

# BM25 index
print("[stateful] Building BM25 index …")
tokenized_corpus = [_tokenize(t) for t in texts]
bm25 = BM25Okapi(tokenized_corpus)

startup_ms = round((time.perf_counter() - t_start) * 1000)
print(f"[stateful] Ready — {index.ntotal} vectors, dim={embedder.dim}, startup={startup_ms} ms")

# ── Schemas ───────────────────────────────────────────────────────────────────
class RetrieveRequest(BaseModel):
    query: str
    k:     int   = Field(default=TOP_K_DEFAULT, ge=1, le=50)
    alpha: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Blend weight: 1.0 = pure dense, 0.0 = pure BM25, 0.5 = hybrid RRF",
    )


class RetrieveResponse(BaseModel):
    chunks: list[str]
    titles: list[str]
    scores: list[float]

# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _dense_ranks(query: str, n: int) -> dict[int, int]:
    """Return {chunk_index: rank} for top-n dense results (rank starts at 0)."""
    q_emb = embedder.embed_query(query).reshape(1, -1)
    _, indices = index.search(q_emb, n)
    return {int(idx): rank for rank, idx in enumerate(indices[0])}


def _bm25_ranks(query: str, n: int) -> dict[int, int]:
    """Return {chunk_index: rank} for top-n BM25 results (rank starts at 0)."""
    scores = bm25.get_scores(_tokenize(query))
    top_indices = np.argsort(scores)[::-1][:n]
    return {int(idx): rank for rank, idx in enumerate(top_indices)}


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


def hybrid_retrieve(query: str, k: int, alpha: float):
    """
    Reciprocal Rank Fusion of dense + BM25 retrieval.

    RRF score = alpha * rrf(dense_rank) + (1-alpha) * rrf(bm25_rank)

    Documents that don't appear in a retriever get no contribution from it.
    """
    fetch_n = max(k * 4, BM25_FETCH)

    if alpha >= 1.0:
        # Pure dense — skip BM25 entirely
        q_emb = embedder.embed_query(query).reshape(1, -1)
        scores, indices = index.search(q_emb, k)
        return indices[0].tolist(), scores[0].tolist()

    if alpha <= 0.0:
        # Pure BM25 — skip dense entirely
        bm25_scores = bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(bm25_scores)[::-1][:k]
        return top_idx.tolist(), bm25_scores[top_idx].tolist()

    # Hybrid
    dense_ranks = _dense_ranks(query, fetch_n)
    bm25_ranks  = _bm25_ranks(query, fetch_n)

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
app = FastAPI(title="DataLens RAG — Stateful", version="2.0")


@app.get("/info")
def info():
    return {
        "mode":         "stateful",
        "embedder":     EMBEDDER_NAME,
        "model":        embedder.model_name,
        "n_chunks":     len(chunks),
        "index_size":   index.ntotal,
        "dim":          embedder.dim,
        "bm25_docs":    len(tokenized_corpus),
        "startup_ms":   startup_ms,
    }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    final_indices, final_scores = hybrid_retrieve(req.query, req.k, req.alpha)

    return RetrieveResponse(
        chunks=[chunks[i]["text"] for i in final_indices],
        titles=[chunks[i]["title"] for i in final_indices],
        scores=final_scores,
    )
