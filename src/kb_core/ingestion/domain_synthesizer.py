"""
Domain knowledge synthesizer.

Reads ALL papers together and extracts a cross-paper overview:
target proteins, substrates, fermentation conditions, yields,
technical challenges, innovations, and field maturity assessment.
"""

from __future__ import annotations

import sys
import textwrap
from datetime import date
from pathlib import Path

from ..config import DomainContext
from ..llm import get_llm
from .pdf_text import read_pdf_text


_DOMAIN_SYSTEM = textwrap.dedent("""
You are an expert in {field_summary}. You synthesize insights across multiple
research papers to build actionable domain knowledge for engineers and researchers.

Papers may be in Chinese or English. Extract everything regardless of language.
Be quantitative wherever possible. Return ONLY valid JSON.
""").strip()

_DOMAIN_PROMPT = textwrap.dedent("""
You are given summaries / full text from {n_papers} research papers on
{paper_topic}. Synthesize cross-paper domain knowledge.

Return a JSON object with exactly these keys:

{{
  "target_proteins": [
    {{
      "name": "protein name (Chinese or English)",
      "aliases": ["other names used"],
      "protein_type": "full-length | fragment | engineered variant",
      "subtype": "domain-specific subtype (e.g. type I/II/III for collagens) or null",
      "sequence_origin": "source organism / engineered / spliced fragment",
      "molecular_weight_kda": "e.g. '~25 kDa' or null",
      "key_features": ["list of distinctive features"],
      "post_translational_modifications_required": true/false/null,
      "papers": ["paper filenames that study this protein"]
    }}
  ],

  "production_substrates": [
    {{
      "name": "substrate name",
      "role": "carbon_source | inducer | co-feed | nitrogen_source | other",
      "phase": "growth | induction | both | null",
      "typical_concentration": "e.g. '10-30 g/L' or null",
      "notes": "any important notes"
    }}
  ],

  "fermentation_conditions": [
    {{
      "parameter": "parameter name (Chinese ok)",
      "typical_range": "e.g. '25-30°C'",
      "optimal_value": "best value found across papers or null",
      "effect": "what goes wrong outside this range",
      "applies_to_phase": "growth | induction | both | null",
      "confidence": "high | medium | low"
    }}
  ],

  "yield_benchmarks": [
    {{
      "protein": "protein name",
      "expression_system": "e.g. '5L发酵罐高密度发酵'",
      "yield_value": "e.g. '10.3 g/L'",
      "key_strategies": ["what made this yield possible"],
      "source_paper": "paper filename"
    }}
  ],

  "technical_challenges": [
    {{
      "challenge": "challenge description",
      "root_cause": "underlying mechanism",
      "affected_proteins": ["which proteins are affected"],
      "solutions": ["solution 1", "solution 2"],
      "benefit_if_solved": "quantified benefit if possible, e.g. 'yield +33%' or 'enables full-length expression'",
      "status": "unsolved | partially_solved | solved",
      "source_papers": ["papers that address this challenge"]
    }}
  ],

  "innovations": [
    {{
      "title": "short title",
      "description": "what was done",
      "innovation_type": "genetic | process | analytical | regulatory",
      "result": "quantified outcome if available",
      "transferability": "can this be applied to other proteins/processes?",
      "source_paper": "paper filename"
    }}
  ],

  "field_maturity": "one paragraph assessment of how mature this field is",
  "industrialization_readiness": "assessment of readiness for industrial scale-up",
  "key_open_questions": ["question 1", "question 2", ...]
}}

PAPERS:
{papers_content}
""").strip()


class DomainKnowledgeSynthesizer:
    """Synthesizes cross-paper domain knowledge from all available papers."""

    def __init__(
        self,
        domain: DomainContext,
        model: str = "gemini-2.5-flash",
        cache_dir: Path | None = None,
    ) -> None:
        self.llm = get_llm(model)
        self.model = model
        self.cache_dir = cache_dir
        self.domain = domain
        self._system = _DOMAIN_SYSTEM.format(field_summary=domain.field_summary)

    def synthesize(self, papers_dir: Path) -> dict:
        """Read all PDFs in papers_dir and synthesize domain knowledge."""
        pdfs = sorted(papers_dir.glob("*.pdf"))
        if not pdfs:
            return {}

        papers_content = self._read_papers(pdfs)
        prompt = _DOMAIN_PROMPT.format(
            n_papers=len(pdfs),
            paper_topic=self.domain.paper_topic,
            papers_content=papers_content,
        )

        print(f"  Synthesizing domain knowledge from {len(pdfs)} papers "
              f"({len(papers_content):,} chars)...")

        try:
            data = self.llm.chat_json(
                prompt, system=self._system,
                temperature=0.1, max_tokens=16384,
            )
        except Exception as e:
            print(f"  [warn] domain synthesis failed: {type(e).__name__}: {e}", file=sys.stderr)
            data = {}

        data["synthesis_date"] = str(date.today())
        data["papers_analyzed"] = [p.name for p in pdfs]
        return data

    def _read_papers(self, pdfs: list[Path]) -> str:
        parts = []
        for pdf in pdfs:
            text = read_pdf_text(pdf, cache_dir=self.cache_dir)
            # Keep first 6000 chars per paper (abstract + methods + results summary)
            parts.append(f"=== PAPER: {pdf.name} ===\n{text[:6000]}")
        return "\n\n".join(parts)
