"""Shared configuration for the Diabetes RAG pipeline (Day 1)."""

import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_JSON_PATH = PROCESSED_DIR / "chunks.json"

# --- Document identity (update per source PDF) ---
DOCUMENT_NAME = "NICE NG28 - Type 2 Diabetes in Adults: Management"
DOCUMENT_SOURCE_URL = (
    "https://www.nice.org.uk/guidance/ng28/resources/"
    "type-2-diabetes-in-adults-management-pdf-1837338615493"
)

# --- Qdrant ---
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "diabetes_guidelines"

# --- Embeddings ---
# Small, CPU-friendly model. Swap for "BAAI/bge-large-en-v1.5" if you have a GPU.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # must match the model above

# --- Chunking ---
# NICE numbers every recommendation like "1.6.1", "1.6.2" ... we use that
# numbering as the natural chunk boundary instead of a fixed character count.
SECTION_NUMBER_REGEX = r"(?m)^(\d{1,2}(?:\.\d{1,2}){1,3})\s+"

# Minimum characters for a chunk to be indexed (filters out stray headers/footers)
MIN_CHUNK_CHARS = 40
