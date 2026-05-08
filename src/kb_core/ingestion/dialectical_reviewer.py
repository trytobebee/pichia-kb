"""
Cross-paper dialectical reviewer.

Reads all per-paper extracted knowledge, groups findings by topic,
then uses Gemini to identify consensus, contradictions, and synthesise
evidence-weighted recommendations.
"""

from __future__ import annotations

import json
import textwrap
from datetime import date

from ..config import DomainContext
from ..llm import get_llm
from ..schema_engine import DialecticalReview


_REVIEW_SYSTEM = textwrap.dedent("""
You are a senior scientist and systematic reviewer specialising in {field_summary}.

Your task is to perform a rigorous cross-paper dialectical analysis:
  1. Compare what multiple papers say about the same experimental topics.
  2. Where papers AGREE → identify consensus, cite all supporting papers,
     assess evidence strength, and state practical implications.
  3. Where papers DISAGREE or give different values → identify the conflict,
     explain the likely reason (different strains, scales, products, methods),
     assess risk for experiment design, and suggest resolution.
  4. Be quantitative wherever possible: include specific values from each paper.
  5. Distinguish findings that are well-replicated from those based on a single
     experiment.

Free-text field values (summary, consensus_claim, divergence_explanation, recommended_approach, etc.) MUST MATCH the source papers' language (write in Chinese if the papers are in Chinese; do NOT translate). Keep technical names as printed: strain IDs (GS115, X-33), vector / promoter / gene symbols (pPIC9K, AOX1, COL3A1), units (g/L, °C, vvm), and chemical abbreviations.

Return ONLY valid JSON. No markdown fences. No explanation outside JSON.
""").strip()

_REVIEW_PROMPT = textwrap.dedent("""
Below is all the process control knowledge extracted from {n_papers} research papers
on {paper_topic}. The papers cover: {paper_list}.

Perform a dialectical cross-paper review. Identify the key experimental topics,
then for each topic synthesize consensus and conflict across papers.

Return a JSON object matching this schema exactly:

{{
  "papers_reviewed": ["paper1.pdf", "paper2.pdf", ...],
  "review_date": "{today}",
  "overall_summary": "2-3 sentence overview of the collective state of knowledge",
  "highest_confidence_findings": [
    "Finding 1 — cite papers",
    "Finding 2 — cite papers"
  ],
  "most_uncertain_areas": [
    "Uncertain area 1",
    "Uncertain area 2"
  ],
  "topic_syntheses": [
    {{
      "topic_area": "e.g. Induction Temperature",
      "summary": "What the literature collectively says on this topic",
      "overall_confidence": "high|medium|low|conflicting|uncertain",
      "key_uncertainties": ["uncertainty 1", "uncertainty 2"],
      "actionable_recommendation": "Concrete guidance accounting for consensus and conflicts",
      "consensus_points": [
        {{
          "topic": "specific sub-topic",
          "consensus_claim": "unified statement all supporting papers agree on",
          "recommended_value": "specific value or range",
          "evidence_strength": "high|medium|low|conflicting|uncertain",
          "applies_to": ["specific product name", "general"],
          "practical_implication": "what this means for experiment design",
          "supporting_papers": [
            {{
              "paper": "filename.pdf",
              "claim": "what this paper specifically says",
              "value": "numeric value if any",
              "experimental_context": "strain/product/scale used"
            }}
          ]
        }}
      ],
      "conflict_points": [
        {{
          "topic": "specific sub-topic",
          "divergence_explanation": "why results differ",
          "risk_level": "high|medium|low",
          "recommended_approach": "what to do given the conflict",
          "open_questions": ["question to resolve this"],
          "positions": [
            {{
              "paper": "filename.pdf",
              "claim": "what this paper says",
              "value": "value if any",
              "experimental_context": "context"
            }}
          ]
        }}
      ]
    }}
  ]
}}

Identify the most important topic areas from the actual content of the
extracted knowledge below. Pick at least the recurring themes that appear
in multiple papers.

=== EXTRACTED PROCESS KNOWLEDGE FROM ALL PAPERS ===

{knowledge_dump}
""").strip()


class DialecticalReviewer:
    """Performs cross-paper dialectical synthesis of process knowledge."""

    def __init__(self, domain: DomainContext, model: str = "gemini-2.5-pro") -> None:
        self.llm = get_llm(model)
        self.model = model
        self.domain = domain
        self._system = _REVIEW_SYSTEM.format(field_summary=domain.field_summary)

    def review(self, all_process_knowledge: list[dict]) -> DialecticalReview:
        """
        all_process_knowledge: list of per-paper dicts from structured_store.
        Each dict has keys: source_file, control_principles, process_stages,
        fermentation_protocols, troubleshooting, product_quality_factors.
        """
        papers = [e.get("source_file", "unknown") for e in all_process_knowledge]
        knowledge_dump = json.dumps(all_process_knowledge, ensure_ascii=False, indent=2)

        # Trim if too large (Gemini 2.5 Pro has large context, but be safe)
        if len(knowledge_dump) > 80000:
            knowledge_dump = knowledge_dump[:80000] + "\n... [truncated]"

        prompt = _REVIEW_PROMPT.format(
            n_papers=len(papers),
            paper_list=", ".join(papers),
            paper_topic=self.domain.paper_topic,
            today=date.today().isoformat(),
            knowledge_dump=knowledge_dump,
        )

        print(f"  Sending {len(knowledge_dump):,} chars to {self.model} for dialectical review...")
        data = self.llm.chat_json(
            prompt, system=self._system,
            temperature=0.15, max_tokens=16384,
        )
        return DialecticalReview(**data)
