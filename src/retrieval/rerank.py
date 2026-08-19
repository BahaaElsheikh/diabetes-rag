"""
Layer 2 - Reranking: Cross-Encoder reranking over candidate retrieved chunks.
"""

from __future__ import annotations

import time
from sentence_transformers import CrossEncoder
from src.config import RERANKER_MODEL_NAME
from src.retrieval.search import RetrievedChunk

_reranker_model: CrossEncoder | None = None
_loaded_model_name: str | None = None


def get_reranker() -> CrossEncoder:
    global _reranker_model, _loaded_model_name
    from src.config import RERANKER_MODEL_NAME

    if _reranker_model is None or _loaded_model_name != RERANKER_MODEL_NAME:
        try:
            _reranker_model = CrossEncoder(RERANKER_MODEL_NAME, local_files_only=True)
        except Exception:
            _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
        _loaded_model_name = RERANKER_MODEL_NAME
    return _reranker_model


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """Rerank candidates using a Cross-Encoder model.

    Args:
        query: User input query string.
        candidates: List of candidate RetrievedChunk objects from bi-encoder search.
        top_k: Number of reranked results to return.
        score_threshold: Minimum cross-encoder score for refusal/filtering.

    Returns:
        Sorted list of top_k RetrievedChunk objects with updated reranker scores.
    """
    if not candidates:
        return []

    from src.config import RERANK_SCORE_THRESHOLD, RERANK_TOP_K
    if top_k is None:
        top_k = RERANK_TOP_K
    if score_threshold is None:
        score_threshold = RERANK_SCORE_THRESHOLD

    model = get_reranker()
    pairs = [[query, c.text] for c in candidates]
    scores = model.predict(pairs)

    reranked = [
        RetrievedChunk(
            text=c.text,
            document_name=c.document_name,
            section_number=c.section_number,
            section_title=c.section_title,
            page_number=c.page_number,
            score=float(s),
            related_sections=c.related_sections,
            patient_subgroup_tags=c.patient_subgroup_tags,
        )
        for c, s in zip(candidates, scores)
    ]

    reranked.sort(key=lambda x: x.score, reverse=True)

    if score_threshold is not None:
        reranked = [c for c in reranked if c.score >= score_threshold]

    return reranked[:top_k]


if __name__ == "__main__":
    import sys
    from src.retrieval.search import search

    query = " ".join(sys.argv[1:]) or "HbA1c target for type 2 diabetes"
    results = search(query, use_reranker=True)

    if not results:
        print(f"Query: {query!r}")
        print("Status: Refused (no results above threshold)")
    else:
        print(f"Query: {query!r}")
        print(f"Status: Accepted ({len(results)} results)")
        for r in results:
            print(f"  rerank_score={r.score:.4f}  {r.citation()}")
            print(f"    {r.text[:150]}...\n")

