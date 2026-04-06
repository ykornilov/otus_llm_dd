"""
Unit tests for RAG pipeline components.
No LLM or network calls — fast, runs offline.

Run: pytest tests/test_unit.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rag import (
    BM25Index,
    RAGPipeline,
    clean_doc_text,
    extract_metadata,
    format_context,
    reciprocal_rank_fusion,
    tokenize,
)

DOCS_DIR = ROOT / "docs"


# ---------------------------------------------------------------------------
# clean_doc_text
# ---------------------------------------------------------------------------

def test_clean_doc_text_removes_template_syntax():
    raw = "Hello {{ variable }} world {% if x %} skip {% endif %} end"
    result = clean_doc_text(raw)
    assert "{{" not in result
    assert "{%" not in result
    assert "Hello" in result
    assert "end" in result


def test_clean_doc_text_collapses_blank_lines():
    raw = "line1\n\n\n\n\nline2"
    result = clean_doc_text(raw)
    assert "\n\n\n" not in result


def test_clean_doc_text_empty_string():
    assert clean_doc_text("") == ""


# ---------------------------------------------------------------------------
# extract_metadata
# ---------------------------------------------------------------------------

def test_extract_metadata_concepts():
    meta = extract_metadata("docs/concepts/chart/index.md")
    assert meta["category"] == "concepts"
    assert meta["section"] == "chart"
    assert meta["filename"] == "index"


def test_extract_metadata_operations():
    meta = extract_metadata("docs/operations/chart/create-chart.md")
    assert meta["category"] == "operations"
    assert meta["section"] == "chart"
    assert meta["filename"] == "create-chart"


def test_extract_metadata_unknown_path():
    meta = extract_metadata("some/random/file.md")
    assert meta["category"] == "unknown"


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

def test_tokenize_removes_stopwords():
    tokens = tokenize("как создать чарт в DataLens")
    assert "в" not in tokens
    assert "как" not in tokens


def test_tokenize_lowercases():
    tokens = tokenize("DataLens Chart")
    assert all(t == t.lower() for t in tokens)


def test_tokenize_removes_punctuation():
    tokens = tokenize("чарт, диаграмма. настройки!")
    assert all(t.isalnum() for t in tokens)


def test_tokenize_empty():
    assert tokenize("") == []


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------

def _make_results(ids: list[str]) -> list[dict]:
    return [{"id": i, "document": f"doc {i}", "metadata": {}, "score": 1.0} for i in ids]


def test_rrf_combines_results():
    dense = _make_results(["a", "b", "c"])
    sparse = _make_results(["b", "c", "d"])
    fused = reciprocal_rank_fusion(dense, sparse)
    ids = [r["id"] for r in fused]
    # b and c appear in both — should rank higher than a and d
    assert ids.index("b") < ids.index("a")
    assert ids.index("c") < ids.index("d")


def test_rrf_returns_all_unique():
    dense = _make_results(["a", "b"])
    sparse = _make_results(["b", "c"])
    fused = reciprocal_rank_fusion(dense, sparse)
    ids = [r["id"] for r in fused]
    assert sorted(ids) == ["a", "b", "c"]


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([], []) == []


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------

def test_format_context_includes_label_and_text():
    results = [
        {"document": "some text", "metadata": {"category": "concepts", "filename": "index"}},
    ]
    ctx = format_context(results)
    assert "concepts/index" in ctx
    assert "some text" in ctx


def test_format_context_multiple_docs_separated():
    results = [
        {"document": "doc1", "metadata": {"category": "c", "filename": "f1"}},
        {"document": "doc2", "metadata": {"category": "c", "filename": "f2"}},
    ]
    ctx = format_context(results)
    assert "doc1" in ctx
    assert "doc2" in ctx
    assert "\n\n" in ctx


# ---------------------------------------------------------------------------
# Docs directory
# ---------------------------------------------------------------------------

def test_docs_dir_exists():
    assert DOCS_DIR.exists(), f"docs/ not found at {DOCS_DIR}"


def test_docs_contains_markdown_files():
    md_files = list(DOCS_DIR.rglob("*.md"))
    assert len(md_files) >= 10, f"Expected ≥10 .md files, found {len(md_files)}"


def test_docs_key_files_present():
    expected = [
        "concepts/chart/index.md",
        "operations/chart/create-chart.md",
        "concepts/chart/ql-charts.md",
    ]
    for rel_path in expected:
        assert (DOCS_DIR / rel_path).exists(), f"Missing: {rel_path}"


# ---------------------------------------------------------------------------
# BM25Index (requires index built — uses chroma_db if present)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline():
    return RAGPipeline()


def test_pipeline_retrieve_returns_results(pipeline):
    results = pipeline.retrieve("создать чарт", k=3)
    assert len(results) > 0
    assert len(results) <= 3


def test_pipeline_retrieve_result_has_required_keys(pipeline):
    results = pipeline.retrieve("иерархия в чарте", k=1)
    assert len(results) == 1
    r = results[0]
    assert "document" in r
    assert "metadata" in r
    assert "score" in r


def test_pipeline_retrieve_relevant_doc(pipeline):
    """Hybrid search should find the chart creation doc for relevant query."""
    results = pipeline.retrieve("Как создать чарт", k=5)
    docs_text = " ".join(r["document"] for r in results)
    assert "чарт" in docs_text.lower()


def test_bm25_search(pipeline):
    results = pipeline.bm25.search("создать чарт DataLens", k=3)
    assert len(results) == 3
    assert all("document" in r for r in results)
