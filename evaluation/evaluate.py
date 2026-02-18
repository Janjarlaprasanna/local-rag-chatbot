"""Evaluation script for the RAG pipeline.

Ingests sample documents, runs test questions, and scores answers.
Requires Ollama running with the configured model.

Usage:
    python -m evaluation.evaluate
"""

import json
import sys
import time
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.document_loader import load_and_chunk
from rag.vector_store import add_documents, clear_collection, get_document_count
from rag.chain import ask

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_DATASET = Path(__file__).resolve().parent / "eval_dataset.json"


def ingest_sample_docs():
    """Ingest all sample documents."""
    clear_collection()
    total = 0
    for file in sorted(DATA_DIR.glob("sample.*")):
        if file.suffix in {".txt", ".csv", ".pdf"}:
            print(f"  Ingesting {file.name}...")
            chunks = load_and_chunk(str(file))
            count = add_documents(chunks)
            total += count
            print(f"    -> {count} chunks")
    print(f"  Total: {total} chunks indexed\n")
    return total


def check_answer(answer: str, expected_keywords: list[str], expect_refusal: bool = False) -> dict:
    """Check if answer contains expected keywords."""
    answer_lower = answer.lower()

    if expect_refusal:
        refusal_phrases = ["don't have enough information", "cannot answer",
                           "not enough information", "don't know", "no information"]
        passed = any(phrase in answer_lower for phrase in refusal_phrases)
        return {"passed": passed, "reason": "refusal detected" if passed else "expected refusal but got answer"}

    found = []
    missing = []
    for kw in expected_keywords:
        if kw.lower() in answer_lower:
            found.append(kw)
        else:
            missing.append(kw)

    # Pass if at least half the keywords are found
    threshold = max(1, len(expected_keywords) // 2)
    passed = len(found) >= threshold

    return {
        "passed": passed,
        "found": found,
        "missing": missing,
        "score": len(found) / len(expected_keywords) if expected_keywords else 0,
    }


def run_evaluation():
    """Run the full evaluation pipeline."""
    print("=" * 60)
    print("LOCAL RAG CHATBOT — EVALUATION")
    print("=" * 60)

    # Load eval dataset
    with open(EVAL_DATASET) as f:
        questions = json.load(f)

    print(f"\nLoaded {len(questions)} test questions\n")

    # Ingest documents
    print("Step 1: Ingesting sample documents...")
    ingest_sample_docs()

    # Run questions
    print("Step 2: Running questions through RAG pipeline...\n")
    results = []
    total_time = 0

    for i, q in enumerate(questions, 1):
        print(f"  Q{i}: {q['question']}")
        start = time.time()

        try:
            response = ask(q["question"])
            elapsed = time.time() - start
            total_time += elapsed

            answer = response["answer"]
            sources = response["sources"]

            check = check_answer(
                answer,
                q["expected_keywords"],
                q.get("expect_refusal", False),
            )

            status = "PASS" if check["passed"] else "FAIL"
            print(f"      [{status}] ({elapsed:.1f}s) Sources: {sources}")
            if not check["passed"]:
                print(f"      Answer: {answer[:100]}...")
                if "missing" in check:
                    print(f"      Missing keywords: {check['missing']}")

            results.append({
                "question": q["question"],
                "passed": check["passed"],
                "time": elapsed,
                "sources": sources,
                **check,
            })

        except Exception as e:
            elapsed = time.time() - start
            print(f"      [ERROR] ({elapsed:.1f}s) {e}")
            results.append({
                "question": q["question"],
                "passed": False,
                "time": elapsed,
                "error": str(e),
            })

        print()

    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    avg_time = total_time / total if total else 0

    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Passed:       {passed}/{total} ({100 * passed / total:.0f}%)")
    print(f"  Failed:       {total - passed}/{total}")
    print(f"  Avg latency:  {avg_time:.1f}s per question")
    print(f"  Total time:   {total_time:.1f}s")
    print("=" * 60)

    # Cleanup
    clear_collection()

    return passed == total


if __name__ == "__main__":
    success = run_evaluation()
    sys.exit(0 if success else 1)
