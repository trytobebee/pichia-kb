"""
Domain knowledge synthesizer.

Reads ALL papers together and extracts a cross-paper overview:
target proteins, substrates, fermentation conditions, yields,
technical challenges, innovations, and field maturity assessment.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from datetime import date
from pathlib import Path

from google import genai
from google.genai import types

from .pdf_text import read_pdf_text


_DOMAIN_SYSTEM = textwrap.dedent("""
You are an expert in Pichia pastoris bioprocess engineering with deep knowledge of
recombinant collagen production. You synthesize insights across multiple research papers
to build actionable domain knowledge for industrial fermentation engineers.

Papers may be in Chinese or English. Extract everything regardless of language.
Be quantitative wherever possible. Return ONLY valid JSON.
""").strip()

_DOMAIN_PROMPT = textwrap.dedent("""
You are given summaries / full text from {n_papers} research papers on Pichia pastoris
recombinant collagen production. Synthesize cross-paper domain knowledge.

Return a JSON object with exactly these keys:

{{
  "target_proteins": [
    {{
      "name": "protein name (Chinese or English)",
      "aliases": ["other names used"],
      "protein_type": "full-length | fragment | human-like",
      "collagen_type": "I | II | III | null",
      "sequence_origin": "human gene | engineered | spliced fragment",
      "molecular_weight_kda": "e.g. '~25 kDa' or null",
      "key_features": ["list of distinctive features"],
      "hydroxylation_required": true/false/null,
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
        model: str = "gemini-2.5-flash",
        cache_dir: Path | None = None,
    ) -> None:
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model
        self.cache_dir = cache_dir

    def synthesize(self, papers_dir: Path) -> dict:
        """Read all PDFs in papers_dir and synthesize domain knowledge."""
        pdfs = sorted(papers_dir.glob("*.pdf"))
        if not pdfs:
            return {}

        papers_content = self._read_papers(pdfs)
        prompt = _DOMAIN_PROMPT.format(
            n_papers=len(pdfs),
            papers_content=papers_content,
        )

        print(f"  Synthesizing domain knowledge from {len(pdfs)} papers "
              f"({len(papers_content):,} chars)...")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_DOMAIN_SYSTEM,
                max_output_tokens=16384,
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  [warn] JSON parse error: {e}", file=sys.stderr)
            print(f"  [warn] raw ({len(raw)} chars): {raw[:300]}", file=sys.stderr)
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
