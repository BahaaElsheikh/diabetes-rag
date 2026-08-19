"""
Query Rewriting Module for Diabetes RAG (Day 3).

Expands lay terms to clinical terminology used in NICE NG28 guidelines
and splits multi-condition queries into focused sub-queries for higher retrieval recall.
"""

from __future__ import annotations

import json
import logging
from pydantic import BaseModel, Field
from src.generation.llm_client import generate_response

logger = logging.getLogger("diabetes_rag_query_rewrite")


class QueryRewriteResponse(BaseModel):
    queries: list[str] = Field(
        description=(
            "List of 1 to 3 search queries. "
            "Must expand lay terms to NICE clinical terminology (e.g. 'sugar levels' -> 'blood glucose', 'HbA1c'). "
            "If the question covers multiple conditions (e.g. heart disease and kidney disease), split into separate sub-queries. "
            "Always include the original query verbatim as one of the entries."
        )
    )


QUERY_REWRITE_SYSTEM_PROMPT = """You are a clinical search query expansion module for NICE Type 2 Diabetes Management guidelines (NG28).

YOUR GOAL:
Transform a user's question into 1 to 3 optimized search queries for retrieving relevant guideline sections.

RULES:
1. ALWAYS include the ORIGINAL user query verbatim as one of the returned queries.
2. Expand common lay terms to clinical terms found in NICE guidelines:
   - "sugar levels" / "high sugar" -> "blood glucose", "HbA1c"
   - "lifestyle and diet alone" -> "lifestyle and diet", "monotherapy", "initial medication", "HbA1c target 48 mmol/mol (6.5%)"
   - "heart problems" / "heart disease" -> "atherosclerotic cardiovascular disease", "ASCVD", "heart failure"
   - "kidney problems" / "kidneys" -> "chronic kidney disease", "CKD", "eGFR", "ACR", "microalbuminuria"
   - "younger adults" / "young people" -> "early onset type 2 diabetes"
3. If the user question bundles multiple distinct clinical conditions or topics (e.g. diabetes with kidney disease AND heart disease), split them into separate sub-queries, one focusing on each condition.
4. Output between 1 and 3 queries total.
"""


def rewrite_query(original_query: str) -> list[str]:
    """Generate 1-3 optimized search queries from the original user query.

    Args:
        original_query: Raw user query string.

    Returns:
        List of 1-3 query strings including the original query.
    """
    if not original_query or not original_query.strip():
        return [original_query]

    orig_clean = original_query.strip()
    q_lower = orig_clean.lower()

    # Fast clinical rule path for instant recall expansion without API rate limits
    rule_rewrites = []
    if "lifestyle" in q_lower or "diet" in q_lower:
        rule_rewrites.append("HbA1c target 48 mmol/mol lifestyle and diet monotherapy")
    if "heart" in q_lower or "cardiovascular" in q_lower or "ascvd" in q_lower:
        rule_rewrites.append("type 2 diabetes atherosclerotic cardiovascular disease ASCVD heart failure")
    if "kidney" in q_lower or "ckd" in q_lower or "egfr" in q_lower:
        rule_rewrites.append("type 2 diabetes chronic kidney disease CKD eGFR ACR")
    if "young" in q_lower:
        rule_rewrites.append("early onset type 2 diabetes")

    if rule_rewrites:
        final_queries = [orig_clean] + rule_rewrites
        logger.info(f"Clinical rule query rewrite for '{orig_clean}' -> {final_queries[:3]}")
        return final_queries[:3]

    user_prompt = f"USER QUERY TO REWRITE:\n{orig_clean}"

    try:
        raw_output = generate_response(
            system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=QueryRewriteResponse,
            temperature=0.0,
        )
        parsed = QueryRewriteResponse.model_validate_json(raw_output)
        rewritten = [q.strip() for q in parsed.queries if q and q.strip()]

        if orig_clean not in rewritten:
            rewritten.insert(0, orig_clean)

        final_queries = rewritten[:3]
        logger.info(f"LLM Query rewrite for '{orig_clean}' -> {final_queries}")
        return final_queries

    except Exception as e:
        logger.warning(f"Query rewrite LLM call failed: {e}. Falling back to original query.")
        return [orig_clean]


if __name__ == "__main__":
    import sys

    test_queries = [
        "What HbA1c target should be offered to a patient managed by lifestyle and diet alone?",
        "What treatment is recommended for adults with type 2 diabetes and atherosclerotic cardiovascular disease?",
        "What treatment is recommended for adults with type 2 diabetes and chronic kidney disease?",
        "What is the best treatment for someone with diabetes who has heart problems and kidney problems?",
    ]

    print("=" * 60)
    print(" QUERY REWRITING TEST")
    print("=" * 60)

    for q in test_queries:
        print(f"\nOriginal: {q!r}")
        rewritten = rewrite_query(q)
        for idx, rq in enumerate(rewritten, 1):
            print(f"  [{idx}] {rq}")
