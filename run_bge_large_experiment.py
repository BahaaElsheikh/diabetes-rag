"""Run the embedding model comparison experiment (bge-large)."""

import sys
import json
import time
from pathlib import Path

# Patch config before other imports
import src.config
src.config.EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
src.config.EMBEDDING_DIM = 1024
src.config.QDRANT_COLLECTION = "diabetes_guidelines_bge_large"

import src.ingestion.embedder
src.ingestion.embedder._model = None

from src.ingestion.chunker import Chunk
from src.config import CHUNKS_JSON_PATH
from src.evaluation.run_eval import run_eval, print_summary, save_results

def main():
    print("="*60)
    print(" EXPERIMENT: Embedding Model Comparison (bge-large)")
    print("="*60)
    
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    # Some fields like 'metadata' might not exist in the json depending on how it was saved
    chunks = []
    for c in raw_chunks:
        # Ignore extra kwargs safely
        chunks.append(Chunk(
            chunk_id=c["chunk_id"],
            document_name=c["document_name"],
            section_number=c.get("section_number"),
            section_title=c.get("section_title"),
            page_number=c["page_number"],
            text=c["text"]
        ))

    print(f"Indexing {len(chunks)} chunks with BAAI/bge-large-en-v1.5...")
    t0 = time.perf_counter()
    n = src.ingestion.embedder.index_chunks(chunks)
    ingestion_time = time.perf_counter() - t0
    print(f"Indexed {n} chunks in {ingestion_time:.2f} seconds.\n")

    print("Evaluating bge-large-en-v1.5 at score_threshold=0.60, top_k=5...")
    summary = run_eval(top_k=5, score_threshold=0.60, label="bge-large_threshold_0.60")
    
    # Store timing metrics for comparison
    summary["metrics"]["ingestion_time_sec"] = round(ingestion_time, 2)
    
    print_summary(summary)
    path = save_results(summary, "bge-large_threshold_0.60")
    print(f"Results saved to: {path.name}")

if __name__ == "__main__":
    main()
