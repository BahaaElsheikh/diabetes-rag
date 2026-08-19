"""
Test script for verifying upgraded Day 3 /ask live responses.
"""

from src.api.main import AskRequest, ask_endpoint

test_queries = [
    "What HbA1c target should be offered to a patient managed by lifestyle and diet alone?",
    "best restaurants in Cairo",
    "What treatment is recommended for adults with type 2 diabetes and atherosclerotic cardiovascular disease?",
]

print("=" * 75)
print(" DAY 3 UPGRADED /ASK LIVE AUDIT RESPONSES")
print("=" * 75)

import sys

import time

for idx, q in enumerate(test_queries, 1):
    req = AskRequest(query=q, top_k=5)
    resp = ask_endpoint(req)
    print(f"\n--- QUERY [{idx}]: {q!r} ---", flush=True)
    print(f"Refused          : {resp.refused} (Reason: {resp.refusal_reason})", flush=True)
    print(f"Recommendation   : {resp.recommendation}", flush=True)
    print(f"Excerpt          : {resp.supporting_excerpt!r}", flush=True)
    print(f"Citations ({len(resp.citations)}): {[c.section_number for c in resp.citations]}", flush=True)
    print(f"Subgroups        : {[ch.patient_subgroup_tags for ch in resp.retrieved_chunks]}", flush=True)
    print(f"Latency          : {resp.latency_ms.total_ms:.1f} ms (Retrieval: {resp.latency_ms.retrieval_and_rerank_ms:.1f}ms, LLM: {resp.latency_ms.llm_ms:.1f}ms)", flush=True)
    time.sleep(3)
