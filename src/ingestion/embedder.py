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
_model_name: str | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model, _model_name
    from src.config import EMBEDDING_MODEL_NAME
    if _model is None or _model_name != EMBEDDING_MODEL_NAME:
        try:
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except Exception:
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        _model_name = EMBEDDING_MODEL_NAME
    return _model


_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    import socket
    try:
        with socket.create_connection((QDRANT_HOST, QDRANT_PORT), timeout=0.2):
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2.0)
            client.get_collections()
            _qdrant_client = client
            return client
    except Exception:
        from src.config import PROJECT_ROOT
        _qdrant_client = QdrantClient(path=str(PROJECT_ROOT / "data" / "qdrant_db"))
        return _qdrant_client


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        info = client.get_collection(QDRANT_COLLECTION)
        # Check current vector size
        vectors_config = info.config.params.vectors
        if hasattr(vectors_config, "size"):
            curr_size = vectors_config.size
        elif isinstance(vectors_config, dict):
            curr_size = vectors_config.get("size")
        else:
            curr_size = None

        if curr_size is not None and curr_size != EMBEDDING_DIM:
            print(f"Recreating collection {QDRANT_COLLECTION}: dimension changed from {curr_size} to {EMBEDDING_DIM}")
            client.delete_collection(QDRANT_COLLECTION)
            if hasattr(client, "_client") and hasattr(client._client, "collections"):
                client._client.collections.pop(QDRANT_COLLECTION, None)
            existing.remove(QDRANT_COLLECTION)

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

    texts = [c.text for c in chunks]
    vectors = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)

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
                "related_sections": chunks[i].related_sections,
                "patient_subgroup_tags": chunks[i].patient_subgroup_tags,
            },
        )
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)


def embed_query(query: str):
    """Embed a single user query for search."""
    model = get_embedding_model()
    return model.encode(
        [f"Represent this sentence for searching relevant passages: {query}"],
        normalize_embeddings=True,
    )[0].tolist()
