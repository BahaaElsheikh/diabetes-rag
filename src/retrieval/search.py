"""
Layer 2 - Retrieval: semantic search over the indexed guideline chunks.

Every result carries the metadata needed to build a citation:
document name, section number, page number.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import QDRANT_COLLECTION, RERANK_CANDIDATE_K
from src.ingestion.embedder import embed_query, get_qdrant_client


from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    text: str
    document_name: str
    section_number: str | None
    section_title: str | None
    page_number: int
    score: float
    related_sections: list[str] = field(default_factory=list)
    patient_subgroup_tags: list[str] = field(default_factory=list)

    def citation(self) -> str:
        loc = f"Section {self.section_number}" if self.section_number else f"Page {self.page_number}"
        return f"{self.document_name}, {loc} (p.{self.page_number})"


def search(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.35,
    use_reranker: bool = False,
    candidate_k: int | None = None,
    use_query_rewrite: bool = True,
) -> list[RetrievedChunk]:
    """Return the top_k most relevant chunks for a query.

    score_threshold filters out weak matches - this is the retrieval-side
    half of the Safety Layer's "insufficient evidence -> refuse" behavior.
    """
    client = get_qdrant_client()

    if candidate_k is None:
        from src.config import RERANK_CANDIDATE_K
        cand_k = RERANK_CANDIDATE_K
    else:
        cand_k = candidate_k

    limit_k = cand_k if use_reranker else top_k
    thresh = 0.0 if use_reranker else score_threshold

    if use_query_rewrite:
        from src.retrieval.query_rewrite import rewrite_query
        search_queries = rewrite_query(query)
    else:
        search_queries = [query]

    seen_keys = set()
    candidates: list[RetrievedChunk] = []

    for q in search_queries:
        query_vector = embed_query(q)
        hits = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=limit_k,
            score_threshold=thresh,
        )
        for h in hits:
            key = (
                h.payload["document_name"],
                h.payload.get("section_number"),
                h.payload["page_number"],
                h.payload["text"],
            )
            if key not in seen_keys:
                seen_keys.add(key)
                candidates.append(
                    RetrievedChunk(
                        text=h.payload["text"],
                        document_name=h.payload["document_name"],
                        section_number=h.payload.get("section_number"),
                        section_title=h.payload.get("section_title"),
                        page_number=h.payload["page_number"],
                        score=h.score,
                        related_sections=h.payload.get("related_sections", []),
                        patient_subgroup_tags=h.payload.get("patient_subgroup_tags", []),
                    )
                )

    if use_reranker and candidates:
        from src.retrieval.rerank import rerank
        return rerank(query, candidates, top_k=top_k, score_threshold=score_threshold)

    return candidates[:top_k]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "HbA1c target for type 2 diabetes"
    results = search(query)

    if not results:
        print(f"No sufficiently confident evidence found for: {query!r}")
        print("(This is the intended refusal path - not a bug.)")
    else:
        print(f"Query: {query!r}\n")
        for r in results:
            print(f"score={r.score:.3f}  {r.citation()}")
            print(f"  {r.text[:200]}...\n")
