"""
Layer 1 - Document Ingestion: section-aware chunking.

Instead of splitting by a fixed character count, we split on NICE's own
numbered recommendation scheme (e.g. "1.6.1", "1.6.2", ...). Each chunk is
one recommendation, tagged with its section number, the nearest section
title we can find above it, and the page it started on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import DOCUMENT_NAME, MIN_CHUNK_CHARS, SECTION_NUMBER_REGEX
from src.ingestion.pdf_parser import PageText

_SECTION_RE = re.compile(SECTION_NUMBER_REGEX)
# A "title" line: short, no trailing period, mostly letters -> heuristic for
# section headers like "Blood glucose management" vs. body prose.
_TITLE_LINE_RE = re.compile(r"^(?:\d{1,2}\.\d{1,2}\s+)?[A-Z][A-Za-z ,/&()-]{3,60}$")


@dataclass
class Chunk:
    chunk_id: str
    document_name: str
    section_number: str | None
    section_title: str | None
    page_number: int
    text: str
    metadata: dict = field(default_factory=dict)


def _build_offset_to_page_map(pages: list[PageText]) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate all pages into one string; return (full_text, [(offset, page_number)])."""
    full_text_parts = []
    offset_map: list[tuple[int, int]] = []
    running_offset = 0
    for p in pages:
        offset_map.append((running_offset, p.page_number))
        full_text_parts.append(p.text)
        running_offset += len(p.text) + 1  # +1 for the joining newline
    return "\n".join(full_text_parts), offset_map


def _page_for_offset(offset: int, offset_map: list[tuple[int, int]]) -> int:
    page = offset_map[0][1]
    for start_offset, page_number in offset_map:
        if start_offset <= offset:
            page = page_number
        else:
            break
    return page


def _nearest_title_above(full_text: str, offset: int) -> str | None:
    """Look a little above the match for a heuristically-detected title line."""
    window = full_text[max(0, offset - 300):offset]
    lines = [ln.strip() for ln in window.splitlines() if ln.strip()]
    for line in reversed(lines):
        if _TITLE_LINE_RE.match(line):
            return line
    return None


def chunk_pages(pages: list[PageText], document_name: str = DOCUMENT_NAME) -> list[Chunk]:
    full_text, offset_map = _build_offset_to_page_map(pages)
    matches = list(_SECTION_RE.finditer(full_text))

    chunks: list[Chunk] = []
    for i, m in enumerate(matches):
        section_number = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()

        if len(body) < MIN_CHUNK_CHARS:
            continue

        page_number = _page_for_offset(start, offset_map)
        title = _nearest_title_above(full_text, start)

        chunks.append(
            Chunk(
                chunk_id=f"{document_name}::{section_number}",
                document_name=document_name,
                section_number=section_number,
                section_title=title,
                page_number=page_number,
                text=body,
            )
        )

    # Fallback: if the numbering pattern found nothing (e.g. a differently
    # formatted PDF), fall back to one chunk per page so the pipeline never
    # produces zero chunks silently.
    if not chunks:
        for p in pages:
            if len(p.text) >= MIN_CHUNK_CHARS:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_name}::page-{p.page_number}",
                        document_name=document_name,
                        section_number=None,
                        section_title=None,
                        page_number=p.page_number,
                        text=p.text,
                    )
                )
    return chunks


if __name__ == "__main__":
    from src.config import RAW_PDF_DIR
    from src.ingestion.pdf_parser import parse_pdf

    pages = parse_pdf(RAW_PDF_DIR / "ng28.pdf")
    result = chunk_pages(pages)
    print(f"Produced {len(result)} chunks from {len(pages)} pages.")
    for c in result[:3]:
        print(f"[{c.section_number}] (p.{c.page_number}) {c.section_title!r} -> {c.text[:120]!r}")
