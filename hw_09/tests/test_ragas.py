"""
Ragas evaluation: Faithfulness, Answer Relevancy, Context Recall.
Run:  pytest tests/test_ragas.py -v -s
Or standalone: python tests/test_ragas.py
"""

import json
import os
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rag import RAGPipeline

GOLDENS_PATH = Path(__file__).parent / "goldens.json"
RESULTS_PATH = Path(__file__).parent / "results.json"

# Minimum acceptable scores (0..1)
THRESHOLDS = {
    "faithfulness": 0.9,
    "answer_relevancy": 0.7,
    "context_recall": 0.8,
}


# ---------------------------------------------------------------------------
# Ragas setup
# ---------------------------------------------------------------------------

def _build_ragas_llm():
    """Build ragas LLM evaluator"""
    import warnings
    from langchain_openai import ChatOpenAI
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.llms import LangchainLLMWrapper

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.0,
        http_client=httpx.Client(timeout=60.0, verify=False),
        http_async_client=httpx.AsyncClient(timeout=60.0, verify=False),
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings():
    import warnings
    from langchain_openai import OpenAIEmbeddings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.embeddings import LangchainEmbeddingsWrapper

    emb = OpenAIEmbeddings(
        model="text-embedding-3-small",
        http_client=httpx.Client(timeout=60.0, verify=False),
        http_async_client=httpx.AsyncClient(timeout=60.0, verify=False),
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    return LangchainEmbeddingsWrapper(emb)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_goldens() -> list[dict]:
    with open(GOLDENS_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_on_goldens(pipeline: RAGPipeline, goldens: list[dict]) -> list[dict]:
    """For each golden: retrieve contexts + generate answer"""
    rows = []
    for g in goldens:
        question = g["question"]
        retrieved = pipeline.retrieve(question, k=5)
        contexts = [r["document"] for r in retrieved]
        answer = pipeline.ask(question, k=5)
        rows.append(
            {
                "id": g["id"],
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": g["expected_answer"],
            }
        )
        print(f"[{g['id']}] Q: {question[:60]}...")
        print(f"       A: {answer[:80]}...")
    return rows


def _mean(values) -> float:
    """Mean of a list, ignoring None values"""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def evaluate_with_ragas(rows: list[dict]) -> dict:
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import AnswerRelevancy, ContextRecall, Faithfulness  # noqa: deprecated path ok for 0.4.x

    ragas_llm = _build_ragas_llm()
    ragas_emb = _build_ragas_embeddings()

    faithfulness = Faithfulness(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)
    context_recall = ContextRecall(llm=ragas_llm)

    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        for r in rows
    ]
    dataset = EvaluationDataset(samples=samples)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
    )
    return result


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pipeline():
    return RAGPipeline()


@pytest.fixture(scope="session")
def eval_results(pipeline):
    goldens = load_goldens()
    rows = run_pipeline_on_goldens(pipeline, goldens)
    result = evaluate_with_ragas(rows)

    df = result.to_pandas()
    scores = {
        "faithfulness": _mean(df["faithfulness"].tolist()),
        "answer_relevancy": _mean(df["answer_relevancy"].tolist()),
        "context_recall": _mean(df["context_recall"].tolist()),
    }

    # Per-sample breakdown
    per_sample = df[
        ["user_input", "faithfulness", "answer_relevancy", "context_recall"]
    ].to_dict(orient="records")

    output = {"scores": scores, "per_sample": per_sample}
    RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Scores: {scores}")
    return scores


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_faithfulness(eval_results):
    score = eval_results["faithfulness"]
    assert score >= THRESHOLDS["faithfulness"], (
        f"Faithfulness {score:.3f} < threshold {THRESHOLDS['faithfulness']}"
    )


def test_answer_relevancy(eval_results):
    score = eval_results["answer_relevancy"]
    assert score >= THRESHOLDS["answer_relevancy"], (
        f"Answer Relevancy {score:.3f} < threshold {THRESHOLDS['answer_relevancy']}"
    )


def test_context_recall(eval_results):
    score = eval_results["context_recall"]
    assert score >= THRESHOLDS["context_recall"], (
        f"Context Recall {score:.3f} < threshold {THRESHOLDS['context_recall']}"
    )


# ---------------------------------------------------------------------------
# Standalone run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    rag = RAGPipeline()
    goldens = load_goldens()
    rows = run_pipeline_on_goldens(rag, goldens)
    result = evaluate_with_ragas(rows)

    df = result.to_pandas()
    scores = {
        "faithfulness": _mean(df["faithfulness"].tolist()),
        "answer_relevancy": _mean(df["answer_relevancy"].tolist()),
        "context_recall": _mean(df["context_recall"].tolist()),
    }
    per_sample = df[
        ["user_input", "faithfulness", "answer_relevancy", "context_recall"]
    ].to_dict(orient="records")

    output = {"scores": scores, "per_sample": per_sample}
    RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== Ragas Evaluation Results ===")
    for metric, score in scores.items():
        threshold = THRESHOLDS[metric]
        status = "PASS" if score >= threshold else "FAIL"
        print(f"  {metric:<20} {score:.3f}  [{status}]")
    print(f"\nFull results saved to {RESULTS_PATH}")
