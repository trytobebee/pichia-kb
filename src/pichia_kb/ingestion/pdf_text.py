"""Shared PDF text extraction with on-disk caching.

All ingestion stages (chunking, entity extraction, synthesis, figure context)
read full-paper plain text via :func:`read_pdf_text`. The first call parses
the PDF with pdfplumber; subsequent calls return the cached text from
``cache_dir/<pdf_stem>.txt`` so we don't re-parse on every rerun and can
inspect what the LLM actually sees.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber


def read_pdf_text(pdf_path: Path, cache_dir: Path | None = None) -> str:
    """Return the full plain text of a PDF, using ``cache_dir`` if provided."""
    cache_file: Path | None = None
    if cache_dir is not None:
        cache_file = cache_dir / f"{pdf_path.stem}.txt"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    full_text = "\n".join(pages)

    if cache_file is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(full_text, encoding="utf-8")

    return full_text
