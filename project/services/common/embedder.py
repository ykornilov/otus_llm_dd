"""
Embedding model wrapper with MPS / CUDA / CPU auto-detection.

Supported embedders (set via EMBEDDER env var):
  - bge-m3               BAAI/bge-m3               ~570M params
  - multilingual-e5-large intfloat/multilingual-e5-large  ~560M params
  - qodo                 Qodo/Qodo-Embed-1-7B       ~7B params  (slow on MPS)

VLLMEmbedder — calls an external vLLM /v1/embeddings endpoint.
  Set VLLM_URL=http://localhost:8003 to use it via rag_stateful.
"""
import numpy as np
import torch
import httpx
from sentence_transformers import SentenceTransformer

# ── Model configs ─────────────────────────────────────────────────────────────
# query_prefix / doc_prefix: prepended before encoding per model's recommendation
EMBEDDER_CONFIGS: dict[str, dict] = {
    "bge-m3": {
        "model_name": "BAAI/bge-m3",
        "query_prefix": "",
        "doc_prefix": "",
        "model_kwargs": {},
    },
    "multilingual-e5-large": {
        "model_name": "intfloat/multilingual-e5-large",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "model_kwargs": {},
    },
    "qodo": {
        "model_name": "Qodo/Qodo-Embed-1-7B",
        # Instruction-following query format per Qodo model card
        "query_prefix": (
            "Instruct: Given a DataLens BI formula question, "
            "retrieve relevant formula documentation\nQuery: "
        ),
        "doc_prefix": "",
        "model_kwargs": {"torch_dtype": torch.float16},
    },
    "qodo-1.5b": {
        "model_name": "Qodo/Qodo-Embed-1-1.5B",
        # Same instruction format as 7B per model card
        "query_prefix": (
            "Instruct: Given a DataLens BI formula question, "
            "retrieve relevant formula documentation\nQuery: "
        ),
        "doc_prefix": "",
        "model_kwargs": {"torch_dtype": torch.float16},
    },
}


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    def __init__(self, name: str):
        cfg = EMBEDDER_CONFIGS.get(name)
        if cfg is None:
            raise ValueError(
                f"Unknown embedder: '{name}'. "
                f"Choose from: {list(EMBEDDER_CONFIGS)}"
            )
        self.name = name
        self.model_name: str = cfg["model_name"]
        self.query_prefix: str = cfg["query_prefix"]
        self.doc_prefix: str = cfg["doc_prefix"]

        device = cfg.get("device") or get_device()
        print(f"[embedder] Loading {self.model_name} on {device} …")
        self.model = SentenceTransformer(
            self.model_name,
            device=device,
            model_kwargs=cfg["model_kwargs"] or {},
        )
        print(f"[embedder] Ready — dim={self.model.get_sentence_embedding_dimension()}")

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_docs(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a corpus of documents. Returns float32 array (N, dim), L2-normalised."""
        prefixed = [self.doc_prefix + t for t in texts]
        return self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns float32 array (dim,), L2-normalised."""
        prefixed = self.query_prefix + text
        return self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32)


# ── vLLM-backed embedder ──────────────────────────────────────────────────────

class VLLMEmbedder:
    """
    Same interface as Embedder, but delegates to a vLLM /v1/embeddings endpoint.

    Usage:
        VLLM_URL=http://localhost:8003 EMBEDDER=qodo-1.5b \\
            uvicorn services.rag_stateful.main:app --port 8001

    The vLLM server must already be running:
        VLLM_HOST_IP=127.0.0.1 vllm serve Qodo/Qodo-Embed-1-1.5B \\
            --runner pooling --port 8003 --dtype float16
    """

    def __init__(self, name: str, vllm_url: str):
        cfg = EMBEDDER_CONFIGS.get(name)
        if cfg is None:
            raise ValueError(
                f"Unknown embedder: '{name}'. "
                f"Choose from: {list(EMBEDDER_CONFIGS)}"
            )
        self.name = name
        self.model_name: str = cfg["model_name"]
        self.query_prefix: str = cfg["query_prefix"]
        self.doc_prefix: str = cfg["doc_prefix"]
        self._url = vllm_url.rstrip("/") + "/v1/embeddings"

        print(f"[vllm-embedder] Connecting to {vllm_url} (model={self.model_name}) …")
        # Probe: embed one string to get dimension and verify connectivity
        test = self._post([" "])
        self._dim = len(test[0])
        print(f"[vllm-embedder] Ready — dim={self._dim}")

    @property
    def dim(self) -> int:
        return self._dim

    def _post(self, texts: list[str]) -> list[list[float]]:
        r = httpx.post(
            self._url,
            json={"model": self.model_name, "input": texts},
            timeout=120,
        )
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]

    def _normalise(self, arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9)

    def embed_docs(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed corpus. Returns float32 array (N, dim), L2-normalised."""
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [self.doc_prefix + t for t in texts[i : i + batch_size]]
            embs = np.array(self._post(batch), dtype=np.float32)
            out.append(self._normalise(embs))
            print(f"[vllm-embedder] {min(i + batch_size, len(texts))}/{len(texts)} docs embedded")
        return np.vstack(out)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed single query. Returns float32 array (dim,), L2-normalised."""
        prefixed = self.query_prefix + text
        emb = np.array(self._post([prefixed])[0], dtype=np.float32)
        return emb / max(np.linalg.norm(emb), 1e-9)
