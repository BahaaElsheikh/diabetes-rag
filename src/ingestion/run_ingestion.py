"""
Day 1 entry point: run the full ingestion pipeline end to end.

Usage:
    python -m src.ingestion.run_ingestion [path/to/ng28.pdf]

Steps:
    1. Parse the PDF page by page (pdf_parser).
    2. Split into section-aware chunks (chunker).
    3. Save chunks to data/processed/chunks.json (for manual inspection).
    4. Embed + upsert everything into Qdrant (embedder).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

from src.config import CHUNKS_JSON_PATH, PROCESSED_DIR, RAW_PDF_DIR
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import index_chunks
from src.ingestion.pdf_parser import parse_pdf


def main(pdf_path: str) -> None:
    print(f"[1/4] Parsing PDF: {pdf_path}")
    pages = parse_pdf(pdf_path)
    print(f"      -> {len(pages)} pages extracted")

    print("[2/4] Chunking by section number")
    chunks = chunk_pages(pages)
    print(f"      -> {len(chunks)} chunks produced")

    print(f"[3/4] Saving chunks to {CHUNKS_JSON_PATH}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)

    print("[4/4] Embedding + indexing into Qdrant")
    n = index_chunks(chunks)
    print(f"      -> {n} vectors written to Qdrant")

    print("\nDone. Try a test query with:")
    print("    python -m src.retrieval.search \"HbA1c target for type 2 diabetes\"")


if __name__ == "__main__":
    pdf_arg = sys.argv[1] if len(sys.argv) > 1 else str(RAW_PDF_DIR / "ng28.pdf")
    main(pdf_arg)
