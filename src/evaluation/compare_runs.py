"""
Compare two evaluation result JSON files and print a side-by-side metrics table.

Usage:
    python -m src.evaluation.compare_runs <previous_results_file> <new_results_file>
"""

import sys
import json
from pathlib import Path


def load_result(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        # Try resolving relative to project root or data/eval
        alt = Path("data/eval") / path_str
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Result file not found: {path_str}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare(file1: str, file2: str):
    res1 = load_result(file1)
    res2 = load_result(file2)

    m1 = res1["metrics"]
    m2 = res2["metrics"]
    c1 = res1.get("config", {})
    c2 = res2.get("config", {})

    lbl1 = res1.get("label", Path(file1).stem)
    lbl2 = res2.get("label", Path(file2).stem)

    print("\n" + "=" * 72)
    print(f"{'EVALUATION RUN COMPARISON':^72}")
    print("=" * 72)
    print(f" Previous: {lbl1} ({Path(file1).name})")
    print(f" New     : {lbl2} ({Path(file2).name})")
    print("-" * 72)

    headers = ["Metric", "Previous", "New", "Delta / Change"]
    rows = [
        (
            "Mean Precision@5",
            f"{m1['mean_precision_at_k']:.4f} ({m1['mean_precision_at_k']:.2%})",
            f"{m2['mean_precision_at_k']:.4f} ({m2['mean_precision_at_k']:.2%})",
            f"{m2['mean_precision_at_k'] - m1['mean_precision_at_k']:+.4f}",
        ),
        (
            "Mean Recall@5",
            f"{m1['mean_recall_at_k']:.4f} ({m1['mean_recall_at_k']:.2%})",
            f"{m2['mean_recall_at_k']:.4f} ({m2['mean_recall_at_k']:.2%})",
            f"{m2['mean_recall_at_k'] - m1['mean_recall_at_k']:+.4f}",
        ),
        (
            "Mean F1@5",
            f"{m1['mean_f1_at_k']:.4f} ({m1['mean_f1_at_k']:.2%})",
            f"{m2['mean_f1_at_k']:.4f} ({m2['mean_f1_at_k']:.2%})",
            f"{m2['mean_f1_at_k'] - m1['mean_f1_at_k']:+.4f}",
        ),
        (
            "Refusal Accuracy",
            f"{m1['refusal_accuracy']:.4f} ({m1['refusal_accuracy']:.2%})",
            f"{m2['refusal_accuracy']:.4f} ({m2['refusal_accuracy']:.2%})",
            f"{m2['refusal_accuracy'] - m1['refusal_accuracy']:+.4f}",
        ),
        (
            "Mean Latency (ms)",
            f"{m1['mean_latency_ms']:.1f} ms",
            f"{m2['mean_latency_ms']:.1f} ms",
            f"{m2['mean_latency_ms'] - m1['mean_latency_ms']:+.1f} ms ({((m2['mean_latency_ms'] - m1['mean_latency_ms'])/m1['mean_latency_ms'])*100:+.1f}%)",
        ),
    ]

    # Print table
    row_format = "{:<22} | {:<18} | {:<18} | {:<15}"
    print(row_format.format(*headers))
    print("-" * 72)
    for row in rows:
        print(row_format.format(*row))
    print("=" * 72 + "\n")

    # Check for leaked out-of-scope queries in new run
    oos_leaks = [
        q for q in res2.get("per_query", [])
        if q.get("out_of_scope") and not q.get("correctly_refused")
    ]
    if oos_leaks:
        print("⚠️ WARNING: Refusal Accuracy dropped! Out-of-scope query leaked:")
        for q in oos_leaks:
            print(f"  - Query: {q['query']}")
            print(f"    Returned sections: {q['returned_sections']}")
            print(f"    Top score: {q['top_score']}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.evaluation.compare_runs <previous_results_file> <new_results_file>")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
