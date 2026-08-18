"""
Day 2 — Run all threshold and top-k sweep experiments in a single process.

This avoids repeated model-loading overhead (sentence-transformers model
loads once, then all experiments run back-to-back).

Usage:
    python -m src.evaluation.run_sweep
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Force model preload before any experiments
print("Pre-loading embedding model...")
from src.ingestion.embedder import get_embedding_model
get_embedding_model()
print("Model loaded.\n")

from src.evaluation.run_eval import run_eval, save_results, print_summary


def run_threshold_sweep():
    """Sweep score_threshold values with fixed top_k=5."""
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    results = []

    print("=" * 70)
    print("  EXPERIMENT: Score Threshold Sweep (top_k=5)")
    print("=" * 70)

    for t in thresholds:
        label = f"threshold_{t:.2f}"
        summary = run_eval(top_k=5, score_threshold=t, label=label)
        print_summary(summary)
        path = save_results(summary, label)
        print(f"  -> Saved: {path.name}")
        results.append(summary)

    # Print comparison table
    print("\n" + "=" * 90)
    print("  THRESHOLD SWEEP COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Threshold':>10} {'P@1':>8} {'P@3':>8} {'Refusal':>10} {'Latency':>10} {'MinCorrect':>12} {'MaxOOS':>10} {'Gap':>8}")
    print("-" * 90)
    for s in results:
        m = s["metrics"]
        c = s["config"]
        print(
            f"{c['score_threshold']:>10.2f} "
            f"{m['precision_at_1']:>8.2%} "
            f"{m['precision_at_3']:>8.2%} "
            f"{m['refusal_accuracy']:>10.2%} "
            f"{m['avg_latency_ms']:>9.1f}ms "
            f"{m['in_scope_min_correct_score']:>12.4f} "
            f"{m['out_of_scope_max_score']:>10.4f} "
            f"{m['score_gap']:>8.4f}"
        )
    print("=" * 90)

    return results


def run_topk_sweep():
    """Sweep top_k values with fixed score_threshold=0.35."""
    topk_values = [3, 5, 8, 10]
    results = []

    print("\n" + "=" * 70)
    print("  EXPERIMENT: Top-K Sweep (score_threshold=0.35)")
    print("=" * 70)

    for k in topk_values:
        label = f"topk_{k}"
        summary = run_eval(top_k=k, score_threshold=0.35, label=label)
        print_summary(summary)
        path = save_results(summary, label)
        print(f"  -> Saved: {path.name}")
        results.append(summary)

    # Print comparison table
    print("\n" + "=" * 90)
    print("  TOP-K SWEEP COMPARISON TABLE")
    print("=" * 90)
    print(f"{'top_k':>8} {'P@1':>8} {'P@3':>8} {'P@k':>8} {'Refusal':>10} {'Latency':>10}")
    print("-" * 90)
    for s in results:
        m = s["metrics"]
        c = s["config"]
        # P@k: expected section appears anywhere in top-k
        n_in = m["in_scope_queries"]
        hit_at_k = sum(
            1 for pq in s["per_query"]
            if not pq["out_of_scope"] and pq["expected_section"] in pq["returned_sections"]
        )
        p_at_k = hit_at_k / n_in if n_in else 0
        print(
            f"{c['top_k']:>8} "
            f"{m['precision_at_1']:>8.2%} "
            f"{m['precision_at_3']:>8.2%} "
            f"{p_at_k:>8.2%} "
            f"{m['refusal_accuracy']:>10.2%} "
            f"{m['avg_latency_ms']:>9.1f}ms"
        )
    print("=" * 90)

    return results


def main():
    print("Starting Day 2 experiment sweep...\n")

    threshold_results = run_threshold_sweep()
    topk_results = run_topk_sweep()

    print("\n\nAll experiments complete. Results saved to data/eval/")


if __name__ == "__main__":
    main()
