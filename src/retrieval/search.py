"""
Layer 2 - Retrieval: semantic search over the indexed guideline chunks.

Every result carries the metadata needed to build a citation:
document name, section number, page number.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import QDRANT_COLLECTION
from src.ingestion.embedder import embed_query, get_qdrant_client


@dataclass
class RetrievedChunk:
    text: str
    document_name: str
    section_number: str | None
    section_title: str | None
    page_number: int
    score: float

    def citation(self) -> str:
        loc = f"Section {self.section_number}" if self.section_number else f"Page {self.page_number}"
        return f"{self.document_name}, {loc} (p.{self.page_number})"


def search(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.35,
    use_reranker: bool = False,
    candidate_k: int = 20,
) -> list[RetrievedChunk]:
    """Return the top_k most relevant chunks for a query.

    score_threshold filters out weak matches - this is the retrieval-side
    half of the Safety Layer's "insufficient evidence -> refuse" behavior.
    """
    client = get_qdrant_client()
    query_vector = embed_query(query)

    limit_k = candidate_k if use_reranker else top_k
    thresh = 0.0 if use_reranker else score_threshold

    hits = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=limit_k,
        score_threshold=thresh,
    )

    candidates = [
        RetrievedChunk(
            text=h.payload["text"],
            document_name=h.payload["document_name"],
            section_number=h.payload.get("section_number"),
            section_title=h.payload.get("section_title"),
            page_number=h.payload["page_number"],
            score=h.score,
        )
        for h in hits
    ]

    if use_reranker and candidates:
        from src.retrieval.rerank import rerank
        return rerank(query, candidates, top_k=top_k, score_threshold=-1.0)

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
