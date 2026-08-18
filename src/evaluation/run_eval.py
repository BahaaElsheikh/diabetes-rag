"""
Retrieval evaluation harness (Bi-encoder vs Cross-encoder Reranker).

Loads test_queries.json, runs each query through the retrieval pipeline,
computes Mean Precision@k, Mean Recall@k, Mean F1@k, Refusal accuracy, and Average latency,
then saves per-query results to a timestamped JSON file.

Usage:
    python -m src.evaluation.run_eval --k 5
    python -m src.evaluation.run_eval --k 5 --no-rerank
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
TEST_QUERIES_PATH = EVAL_DIR / "test_queries.json"


def _run_single_query(
    query: str,
    top_k: int,
    score_threshold: float,
    use_reranker: bool,
    collection_name: str | None,
):
    """Run one retrieval query, return (results, latency_ms)."""
    from src.retrieval.search import search

    t0 = time.perf_counter()
    results = search(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        use_reranker=use_reranker,
        candidate_k=20,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return results, latency_ms


def run_eval(
    top_k: int = 5,
    score_threshold: float = 0.35,
    use_reranker: bool = True,
    collection_name: str | None = None,
    label: str = "",
) -> dict:
    """Execute the full evaluation and return the summary dict."""
    from src.config import QDRANT_COLLECTION

    if collection_name is None:
        collection_name = QDRANT_COLLECTION

    with open(TEST_QUERIES_PATH, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    in_scope = [q for q in test_queries if not q.get("out_of_scope", False)]
    out_of_scope = [q for q in test_queries if q.get("out_of_scope", False)]

    per_query_results = []
    total_latency = 0.0

    precisions = []
    recalls = []
    f1s = []
    refusal_correct = 0

    # --- In-scope queries ---
    for q in in_scope:
        query_text = q["query"]
        relevant_sections = q.get("relevant_sections", [])
        if not relevant_sections and q.get("expected_section"):
            relevant_sections = [q["expected_section"]]

        results, latency_ms = _run_single_query(
            query_text, top_k, score_threshold, use_reranker, collection_name
        )
        total_latency += latency_ms

        returned_sections = [r.section_number for r in results]
        top_score = results[0].score if results else 0.0

        # Calculate Precision@k, Recall@k, F1@k
        relevant_set = set(relevant_sections)
        matches = [s for s in returned_sections if s in relevant_set]
        precision_k = len(matches) / top_k if top_k > 0 else 0.0

        retrieved_relevant_unique = set(returned_sections).intersection(relevant_set)
        recall_k = (
            len(retrieved_relevant_unique) / len(relevant_set)
            if relevant_set
            else 0.0
        )

        if (precision_k + recall_k) > 0:
            f1_k = (2 * precision_k * recall_k) / (precision_k + recall_k)
        else:
            f1_k = 0.0

        precisions.append(precision_k)
        recalls.append(recall_k)
        f1s.append(f1_k)

        per_query_results.append({
            "query": query_text,
            "out_of_scope": False,
            "relevant_sections": relevant_sections,
            "returned_sections": returned_sections,
            "top_score": round(top_score, 4),
            "all_scores": [round(r.score, 4) for r in results],
            "precision_at_k": round(precision_k, 4),
            "recall_at_k": round(recall_k, 4),
            "f1_at_k": round(f1_k, 4),
            "latency_ms": round(latency_ms, 1),
        })

    # --- Out-of-scope queries ---
    for q in out_of_scope:
        query_text = q["query"]
        results, latency_ms = _run_single_query(
            query_text, top_k, score_threshold, use_reranker, collection_name
        )
        total_latency += latency_ms

        is_correctly_refused = len(results) == 0
        if is_correctly_refused:
            refusal_correct += 1

        top_score = results[0].score if results else 0.0

        per_query_results.append({
            "query": query_text,
            "out_of_scope": True,
            "relevant_sections": [],
            "returned_sections": [r.section_number for r in results],
            "top_score": round(top_score, 4),
            "all_scores": [round(r.score, 4) for r in results],
            "correctly_refused": is_correctly_refused,
            "latency_ms": round(latency_ms, 1),
        })

    n_in = len(in_scope)
    n_out = len(out_of_scope)

    mean_precision = sum(precisions) / n_in if n_in else 0.0
    mean_recall = sum(recalls) / n_in if n_in else 0.0
    mean_f1 = sum(f1s) / n_in if n_in else 0.0
    refusal_acc = refusal_correct / n_out if n_out else 0.0
    mean_latency = total_latency / len(test_queries) if test_queries else 0.0

    mode_label = "With reranker" if use_reranker else "Bi-encoder only"
    final_label = label if label else mode_label

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": final_label,
        "mode": "reranker" if use_reranker else "bi-encoder",
        "config": {
            "collection_name": collection_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "use_reranker": use_reranker,
        },
        "metrics": {
            "mean_precision_at_k": round(mean_precision, 4),
            "mean_recall_at_k": round(mean_recall, 4),
            "mean_f1_at_k": round(mean_f1, 4),
            "refusal_accuracy": round(refusal_acc, 4),
            "mean_latency_ms": round(mean_latency, 1),
            "in_scope_queries": n_in,
            "out_of_scope_queries": n_out,
        },
        "per_query": per_query_results,
    }

    return summary


def save_results(summary: dict, suffix: str = "") -> Path:
    """Save results to a timestamped JSON file."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{suffix}" if suffix else ""
    path = EVAL_DIR / f"results_{ts}{tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path


def print_summary(summary: dict) -> None:
    """Pretty-print the evaluation summary to stdout."""
    m = summary["metrics"]
    c = summary["config"]
    label = summary.get("label", "")
    k = c["top_k"]

    print(f"\n{'='*65}")
    print(f"  EVALUATION SUMMARY: {label}")
    print(f"  Mode       : {'With Reranker' if c['use_reranker'] else 'Bi-encoder Only'}")
    print(f"  Collection : {c['collection_name']}")
    print(f"  top_k (k)  : {k}")
    print(f"  threshold  : {c['score_threshold']}")
    print(f"{'='*65}")
    print(f"  Mean Precision@{k} : {m['mean_precision_at_k']:.4f} ({m['mean_precision_at_k']:.2%})")
    print(f"  Mean Recall@{k}    : {m['mean_recall_at_k']:.4f} ({m['mean_recall_at_k']:.2%})")
    print(f"  Mean F1@{k}        : {m['mean_f1_at_k']:.4f} ({m['mean_f1_at_k']:.2%})")
    print(f"  Refusal accuracy  : {m['refusal_accuracy']:.4f} ({m['refusal_accuracy']:.2%})")
    print(f"  Mean latency      : {m['mean_latency_ms']:.1f} ms")
    print(f"{'='*65}\n")

    print(f"{'PER-QUERY BREAKDOWN':^65}")
    print("-" * 65)
    for qres in summary["per_query"]:
        if qres["out_of_scope"]:
            status = "REFUSED (OK)" if qres["correctly_refused"] else "FAILED TO REFUSE"
            print(f"[OOS] {qres['query'][:45]:45s} | {status:18s} | {qres['latency_ms']:6.1f}ms")
        else:
            p = qres["precision_at_k"]
            r = qres["recall_at_k"]
            f1 = qres["f1_at_k"]
            ret_secs = ", ".join(qres["returned_sections"][:3])
            print(f"[IN ] {qres['query'][:35]:35s} | P@{k}={p:.2f} R@{k}={r:.2f} F1={f1:.2f} | secs:[{ret_secs}] | {qres['latency_ms']:6.1f}ms")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument("-k", "--k", type=int, default=5, help="top_k results (default: 5)")
    parser.add_argument("--top_k", type=int, default=None, help="Alias for --k")
    parser.add_argument("--threshold", type=float, default=0.35, help="Score threshold")
    parser.add_argument("--no-rerank", action="store_true", default=False, help="Disable CrossEncoder reranker")
    parser.add_argument("--collection", type=str, default=None, help="Override Qdrant collection name")
    parser.add_argument("--label", type=str, default="", help="Label for this run")
    parser.add_argument("--save", action="store_true", default=True, help="Save results to JSON (default: True)")
    args = parser.parse_args()

    top_k = args.top_k if args.top_k is not None else args.k
    use_reranker = not args.no_rerank
    mode_tag = "no_rerank" if args.no_rerank else "with_reranker"
    run_label = args.label if args.label else mode_tag

    summary = run_eval(
        top_k=top_k,
        score_threshold=args.threshold,
        use_reranker=use_reranker,
        collection_name=args.collection,
        label=run_label,
    )

    print_summary(summary)

    if args.save:
        path = save_results(summary, mode_tag)
        print(f"Results saved to: {path}")


if __name__ == "__main__":
    main()
