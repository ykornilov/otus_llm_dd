import os
import re
from pathlib import Path

import chromadb
import nltk
import httpx
from chromadb import EmbeddingFunction, Documents, Embeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("russian")) | set(stopwords.words("english"))

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
DOCS_DIR = Path(__file__).parent / "docs"
COLLECTION_NAME = "hw9_docs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ---------------------------------------------------------------------------
# Embedding function
# ---------------------------------------------------------------------------

class E5EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        device = "mps" if _mps_available() else "cpu"
        self._model = SentenceTransformer(model_name, device=device)

    def __call__(self, input: Documents) -> Embeddings:
        texts_with_prefix = []
        for text in input:
            if len(text) < 200:
                texts_with_prefix.append(f"query: {text}")
            else:
                texts_with_prefix.append(f"passage: {text}")
        embeddings = self._model.encode(
            texts_with_prefix,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Document processing helpers
# ---------------------------------------------------------------------------

def clean_doc_text(text: str) -> str:
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\{%.*?%\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_metadata(source_path: str) -> dict:
    parts = Path(source_path).parts
    category = "unknown"
    section = "unknown"
    for i, part in enumerate(parts):
        if part in ("concepts", "operations"):
            category = part
            if i + 1 < len(parts):
                section = parts[i + 1]
            break
    return {"category": category, "section": section, "filename": Path(source_path).stem}


def tokenize(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalnum() and t not in STOPWORDS]


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(docs_dir: Path = DOCS_DIR, chroma_dir: Path = CHROMA_DIR) -> chromadb.Collection:
    """Load documents, embed them, and store in ChromaDB. Returns collection."""
    embed_fn = E5EmbeddingFunction()
    client = chromadb.PersistentClient(path=str(chroma_dir))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine", "hnsw:M": 16, "hnsw:construction_ef": 100, "hnsw:search_ef": 50},
    )

    md_files = list(docs_dir.rglob("*.md"))
    all_chunks, all_ids, all_metas = [], [], []
    chunk_counter = 0

    for file_path in md_files:
        raw = file_path.read_text(encoding="utf-8")
        cleaned = clean_doc_text(raw)
        meta_base = extract_metadata(str(file_path))
        meta_base["source"] = str(file_path)
        for chunk in _split_text(cleaned):
            chunk_id = f"chunk_{chunk_counter:04d}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metas.append({**meta_base, "chunk_id": chunk_id})
            chunk_counter += 1

    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metas)
    return collection


def get_or_build_index(docs_dir: Path = DOCS_DIR, chroma_dir: Path = CHROMA_DIR) -> chromadb.Collection:
    embed_fn = E5EmbeddingFunction()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
        if collection.count() > 0:
            return collection
    except Exception:
        pass
    return build_index(docs_dir, chroma_dir)


# ---------------------------------------------------------------------------
# BM25 index (built lazily from Chroma docs)
# ---------------------------------------------------------------------------

class BM25Index:
    def __init__(self, collection: chromadb.Collection):
        result = collection.get(include=["documents", "metadatas"])
        self._docs = result["documents"]
        self._ids = result["ids"]
        self._metas = result["metadatas"]
        self._bm25 = BM25Okapi([tokenize(d) for d in self._docs])

    def search(self, query: str, k: int = 5) -> list[dict]:
        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {
                "id": self._ids[i],
                "document": self._docs[i],
                "metadata": self._metas[i],
                "score": float(scores[i]),
            }
            for i in top_indices
        ]


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------

def search_dense(collection: chromadb.Collection, query: str, k: int = 5) -> list[dict]:
    results = collection.query(query_texts=[query], n_results=k, include=["documents", "metadatas", "distances"])
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]
    return [
        {"id": ids[i], "document": docs[i], "metadata": metas[i], "score": 1.0 - distances[i]}
        for i in range(len(docs))
    ]


def reciprocal_rank_fusion(dense_results: list[dict], sparse_results: list[dict], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    docs_map: dict[str, dict] = {}
    for rank, item in enumerate(dense_results):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        docs_map[doc_id] = item
    for rank, item in enumerate(sparse_results):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        docs_map[doc_id] = item
    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
    return [{"score": scores[did], **{k: v for k, v in docs_map[did].items() if k != "score"}} for did in sorted_ids]


def search_hybrid(collection: chromadb.Collection, bm25_index: BM25Index, query: str, k: int = 5) -> list[dict]:
    dense = search_dense(collection, query, k)
    sparse = bm25_index.search(query, k)
    fused = reciprocal_rank_fusion(dense, sparse)
    return fused[:k]


# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------

RAG_SYSTEM = (
    "You are a Yandex DataLens documentation assistant.\n"
    "Answer user questions using ONLY the provided context.\n"
    "If the context doesn't contain an answer, say: "
    "'There is no information about this in the documentation.'\n"
    "Don't make up facts."
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.1,
        http_client=httpx.Client(timeout=60.0, verify=False),
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )


def format_context(results: list[dict]) -> str:
    parts = []
    for r in results:
        meta = r.get("metadata", {})
        label = f"{meta.get('category', '')}/{meta.get('filename', '')}"
        parts.append(f"[{label}]\n{r['document']}")
    return "\n\n".join(parts)


class RAGPipeline:
    def __init__(self, docs_dir: Path = DOCS_DIR, chroma_dir: Path = CHROMA_DIR):
        self.collection = get_or_build_index(docs_dir, chroma_dir)
        self.bm25 = BM25Index(self.collection)
        self.llm = build_llm()
        self.chain = RAG_PROMPT | self.llm

    def retrieve(self, question: str, k: int = 5) -> list[dict]:
        return search_hybrid(self.collection, self.bm25, question, k)

    def ask(self, question: str, k: int = 5) -> str:
        results = self.retrieve(question, k)
        context = format_context(results)
        response = self.chain.invoke({"context": context, "question": question})
        return response.content


if __name__ == "__main__":
    pipeline = RAGPipeline()
    answer = pipeline.ask("Как создать чарт в Yandex DataLens?")
    print(answer)
