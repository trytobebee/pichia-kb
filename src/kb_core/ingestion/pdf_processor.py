"""PDF parsing and chunking for research papers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..schema_engine import KnowledgeChunk
from .pdf_text import read_pdf_text


# Sections we want to track as context signals
_SECTION_PATTERNS = [
    r"abstract",
    r"introduction",
    r"materials?\s+and\s+methods?",
    r"results?",
    r"discussion",
    r"conclusion",
    r"fermentation",
    r"expression",
    r"purification",
    r"characterization",
    r"glycosylation",
    r"strain",
    r"vector",
    r"medium|media",
]
_SECTION_RE = re.compile(
    r"^(?:" + "|".join(_SECTION_PATTERNS) + r")\b",
    re.IGNORECASE,
)

# Domain keywords are now per-project; loaded from project config
# (data/projects/<slug>/config.yaml: keywords). Empty default = match nothing.

# Preferred natural break points, in priority order. Chunking will try to
# end at one of these within the trailing half of a chunk window so the LLM
# sees coherent passages instead of mid-sentence cuts.
_BREAK_CHARS = ("\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ")


class PDFProcessor:
    """Extracts text from PDF papers and splits into retrievable chunks.

    Chunking is **character-based** rather than whitespace-tokenized: Chinese
    text has very few spaces, so ``str.split()`` produced single chunks of
    whole papers. ``chunk_size`` and ``overlap`` are now in characters, with
    natural sentence breaks preferred.
    """

    def __init__(
        self,
        chunk_size: int = 1800,
        overlap: int = 250,
        cache_dir: Path | None = None,
        keywords: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.cache_dir = cache_dir
        self.keywords = keywords or []

    def process(self, pdf_path: Path) -> list[KnowledgeChunk]:
        """Return a list of KnowledgeChunk objects from the PDF."""
        full_text = read_pdf_text(pdf_path, cache_dir=self.cache_dir)
        chunks = self._split_text(full_text)
        source_file = pdf_path.name
        result: list[KnowledgeChunk] = []
        current_section: str | None = None

        for i, chunk_text in enumerate(chunks):
            detected_section = self._detect_section(chunk_text, current_section)
            if detected_section:
                current_section = detected_section

            chunk_id = hashlib.md5(
                f"{source_file}:{i}:{chunk_text[:50]}".encode()
            ).hexdigest()[:12]

            keywords = self._extract_keywords(chunk_text)

            result.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    source_file=source_file,
                    section=current_section,
                    content=chunk_text.strip(),
                    keywords=keywords,
                )
            )

        return result

    # ── private ──────────────────────────────────────────────────────────────

    def _split_text(self, text: str) -> list[str]:
        """Split into ``chunk_size``-char windows, preferring natural breaks."""
        n = len(text)
        if n == 0:
            return []

        chunks: list[str] = []
        start = 0
        # Allow break-point search to start halfway through the window so a
        # chunk is never less than half its target size when a break is found.
        min_break_pos = self.chunk_size // 2

        while start < n:
            end = min(start + self.chunk_size, n)
            if end < n:
                best_cut = -1
                for sep in _BREAK_CHARS:
                    cut = text.rfind(sep, start + min_break_pos, end)
                    if cut != -1:
                        best_cut = max(best_cut, cut + len(sep))
                        break
                if best_cut != -1:
                    end = best_cut

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= n:
                break
            # Step forward, leaving ``overlap`` chars of context, but never
            # stand still (overlap >= chunk_size would loop).
            start = max(end - self.overlap, start + 1)

        return chunks

    def _detect_section(self, text: str, current: str | None) -> str | None:
        for line in text.splitlines()[:5]:
            line = line.strip()
            if 3 < len(line) < 80 and _SECTION_RE.match(line):
                return line
        return None

    def _extract_keywords(self, text: str) -> list[str]:
        lower = text.lower()
        return [kw for kw in self.keywords if kw.lower() in lower]
