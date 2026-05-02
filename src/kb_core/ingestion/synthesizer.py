"""
Process knowledge synthesizer.

Unlike the chunk-by-chunk extractor, this synthesizer reads a whole paper
and asks Gemini to distil control principles, protocols, and quality factors
— the higher-level knowledge that guides fermentation decision-making.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from ..config import DomainContext
from ..llm import get_llm
from .pdf_text import read_pdf_text


_SYNTHESIS_SYSTEM = textwrap.dedent("""
You are an expert in {field_summary}.
Your task is to read research paper content and extract actionable control knowledge —
the principles, protocols, and quality factors that help engineers control process
outcomes (product yield, structure, modification, activity).

Papers may be in Chinese or English. Extract knowledge regardless of language.
Return ONLY valid JSON. Be specific and quantitative wherever the text supports it.
""").strip()

_SYNTHESIS_PROMPT = textwrap.dedent("""
Read the following research paper content and extract process control knowledge.
Return a JSON object with these keys (omit any key with an empty array):

{{
  "control_principles": [
    {{
      "title": "concise principle name (e.g. 'Low temperature reduces proteolysis')",
      "stage": "seed_culture|glycerol_batch|glycerol_fed_batch|methanol_transition|methanol_induction|harvest|downstream|null",
      "parameter": "the controlled variable",
      "observation": "what is observed / the trigger condition",
      "mechanism": "biological/biochemical reason",
      "recommendation": "concrete actionable guidance",
      "target_value": "specific value or range if mentioned",
      "consequence_if_ignored": "what goes wrong",
      "priority": "critical|important|advisory",
      "applies_to_product": ["specific product name", "general", etc.]
    }}
  ],

  "process_stages": [
    {{
      "stage": "seed_culture|glycerol_batch|glycerol_fed_batch|methanol_transition|methanol_induction|harvest|downstream",
      "description": "what happens in this stage",
      "carbon_source": "glycerol / methanol / glucose",
      "typical_duration": "e.g. 24 h",
      "temperature_celsius": "value or range",
      "ph": "set-point",
      "dissolved_oxygen_percent": "e.g. >20%",
      "agitation_rpm": "value",
      "aeration_vvm": "value",
      "feed_rate": "e.g. 3.6 mL/h/L or exponential",
      "transition_trigger": "condition to advance to next stage",
      "key_monitoring": ["DO", "pH", "methanol concentration"],
      "common_problems": ["problem1", "problem2"]
    }}
  ],

  "fermentation_protocols": [
    {{
      "name": "protocol name",
      "target_product": "product name",
      "host_strain": "strain name",
      "expression_vector": "vector name",
      "total_duration_hours": "total time",
      "expected_yield": "e.g. 500 mg/L",
      "critical_success_factors": ["factor1", "factor2"],
      "quality_checkpoints": ["SDS-PAGE at 24h", "activity assay at harvest"]
    }}
  ],

  "troubleshooting": [
    {{
      "problem": "observable symptom",
      "stage": "stage where it occurs or null",
      "root_causes": ["cause1", "cause2"],
      "diagnostic_steps": ["check X", "measure Y"],
      "solutions": ["solution1", "solution2"],
      "prevention": "how to prevent"
    }}
  ],

  "product_quality_factors": [
    {{
      "factor": "process variable",
      "stage": "stage or null",
      "effect_on_structure": "effect on folding/aggregation",
      "effect_on_modification": "effect on glycosylation/hydroxylation/cleavage",
      "effect_on_activity": "effect on biological activity",
      "optimal_range": "best value or range"
    }}
  ]
}}

SOURCE FILE: {source_file}

PAPER CONTENT:
{content}
""").strip()


class ProcessKnowledgeSynthesizer:
    """Synthesizes fermentation control knowledge from full paper content."""

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
        self._system = _SYNTHESIS_SYSTEM.format(field_summary=domain.field_summary)

    def synthesize(self, full_text: str, source_file: str) -> dict:
        """Return raw dict of synthesized process knowledge."""
        content = full_text[:60000]
        prompt = _SYNTHESIS_PROMPT.format(
            source_file=source_file,
            content=content,
        )
        try:
            return self.llm.chat_json(
                prompt, system=self._system,
                temperature=0.1, max_tokens=16384,
            )
        except json.JSONDecodeError as e:
            import sys
            print(f"[synthesizer] JSON parse error for {source_file}: {e}", file=sys.stderr)
            return {}

    def synthesize_from_pdf(self, pdf_path: Path) -> dict:
        """Read PDF and synthesize process knowledge."""
        full_text = read_pdf_text(pdf_path, cache_dir=self.cache_dir)
        return self.synthesize(full_text, pdf_path.name)
