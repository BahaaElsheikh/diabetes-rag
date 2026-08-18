"""
Layer 2 - Reranking: Cross-Encoder reranking over candidate retrieved chunks.
"""

from __future__ import annotations

import time
from sentence_transformers import CrossEncoder
from src.retrieval.search import RetrievedChunk

_reranker_model: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int = 5,
    score_threshold: float | None = -1.0,
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
        )
        for c, s in zip(candidates, scores)
    ]

    reranked.sort(key=lambda x: x.score, reverse=True)

    if score_threshold is not None:
        reranked = [c for c in reranked if c.score >= score_threshold]

    return reranked[:top_k]
