"""Pre-download HuggingFace models for embedding and reranking during Docker build."""

from sentence_transformers import CrossEncoder, SentenceTransformer

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def main():
    print(f"Pre-downloading embedding model: {EMBEDDING_MODEL_NAME}...", flush=True)
    SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model cached successfully.", flush=True)

    print(f"Pre-downloading reranker model: {RERANKER_MODEL_NAME}...", flush=True)
    CrossEncoder(RERANKER_MODEL_NAME)
    print("Reranker model cached successfully.", flush=True)

if __name__ == "__main__":
    main()
