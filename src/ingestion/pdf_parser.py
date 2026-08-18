"""
Layer 1 - Document Ingestion: PDF parsing.

Extracts text from the guideline PDF page by page (so we never lose the
page-number metadata we need for citations later), and does light cleanup
of headers/footers that NICE repeats on every page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# pyrefly: ignore [missing-import]
import fitz  # PyMuPDF


@dataclass
class PageText:
    page_number: int  # 1-indexed, matches what a human sees in the PDF
    text: str


# Repeating boilerplate NICE prints on every page - strip it so it doesn't
# pollute chunk boundaries or get mistaken for guideline content.
_BOILERPLATE_PATTERNS = [
    r"©\s*NICE\s*\d{4}\.?\s*All rights reserved\.?",
    r"Subject to Notice of rights.*?notice-of-rights\)\.?",
    r"Page\s*\d+\s*of\s*[\dA-Za-z]+",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE | re.DOTALL)


def clean_page_text(raw_text: str) -> str:
    """Remove repeated headers/footers and normalize whitespace."""
    text = _BOILERPLATE_RE.sub(" ", raw_text)
    # Collapse the weird mid-word breaks PyMuPDF sometimes introduces from
    # justified NICE layouts (e.g. "structured\n· curriculum").
    text = re.sub(r"\n·\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def parse_pdf(pdf_path: str | Path) -> list[PageText]:
    """Extract cleaned text from every page of the PDF, keeping page numbers."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}. Download NG28 from NICE and place it there."
        )

    pages: list[PageText] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            raw = page.get_text("text")
            cleaned = clean_page_text(raw)
            if cleaned:
                pages.append(PageText(page_number=i + 1, text=cleaned))
    return pages


if __name__ == "__main__":
    import sys

    from src.config import RAW_PDF_DIR

    pdf_file = sys.argv[1] if len(sys.argv) > 1 else RAW_PDF_DIR / "ng28.pdf"
    result = parse_pdf(pdf_file)
    print(f"Parsed {len(result)} pages.")
    if result:
        print("--- Sample (page 1) ---")
        print(result[0].text[:500])
