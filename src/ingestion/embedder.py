"""
Layer 1/2 - Embeddings + Qdrant indexing.

Turns each Chunk into a vector and upserts it into Qdrant along with the
metadata we need to produce a citation later: document name, section
number, section title, page number.
"""

from __future__ import annotations

# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)
from src.ingestion.chunker import Chunk

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_qdrant_client() -> QdrantClient:
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2.0)
        client.get_collections()
        return client
    except Exception:
        from src.config import PROJECT_ROOT
        return QdrantClient(path=str(PROJECT_ROOT / "data" / "qdrant_db"))



def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def index_chunks(chunks: list[Chunk]) -> int:
    """Embed and upsert all chunks. Returns number of points written."""
    if not chunks:
        return 0

    model = get_embedding_model()
    client = get_qdrant_client()
    ensure_collection(client)

    # bge models expect a "passage:" style prefix isn't required for bge-small,
    # but a light instruction prefix improves retrieval quality in practice.
    texts = [c.text for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "chunk_id": chunks[i].chunk_id,
                "document_name": chunks[i].document_name,
                "section_number": chunks[i].section_number,
                "section_title": chunks[i].section_title,
                "page_number": chunks[i].page_number,
                "text": chunks[i].text,
            },
        )
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)


def embed_query(query: str):
    """Embed a single user query for search."""
    model = get_embedding_model()
    return model.encode([f"Represent this sentence for searching relevant passages: {query}"], normalize_embeddings=True)[0].tolist()

