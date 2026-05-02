"""
Experiment-lineage extraction.

Reads the already-extracted experiments of a single paper and asks Gemini to
identify directed edges between them — which experiment builds on which and
how. Single LLM call per paper; produces a list[ExperimentLineageEdge].

Output is stored on the same `<paper>.experiments.json` (lineage field), so
downstream code (Streamlit page, cross-paper comparison) can read it from one
place.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from typing import Any

from google import genai
from google.genai import types

from ..schema.experiments import (
    ExperimentLineageEdge,
    ExperimentRun,
    PaperExperiments,
)
from ..config import DomainContext


_LINEAGE_SYSTEM = textwrap.dedent("""
You are an expert in {field_summary}. Your task is to
read a list of experiments extracted from one paper and identify the narrative
relationships between them — which experiment builds on which, and how.

Most thesis-style papers describe a sequential optimization: parameter A screening
→ parameter B tuning → component co-expression → scale-up. The "best" strain or
condition from each stage is typically carried into the next stage. Your job is
to make these implicit dependencies explicit as edges.

Be conservative. Only emit an edge when the evidence (parent_strain text,
fixed_parameters citing earlier results, or sequential section numbering with a
clear continuation) is clear. Do NOT invent fully linear chains; independent
parallel experiments should remain disconnected.

Return ONLY valid JSON.
""").strip()


_LINEAGE_PROMPT = textwrap.dedent("""
Source paper: {source_file}

Below are the {n} experiments already extracted from this paper. Each block
shows the experiment_id, title, paper_section, goal, the strain construct
(esp. parent_strain), and the outcome.

{experiments_block}

Identify directed edges between these experiments. Return JSON:

{{
  "edges": [
    {{
      "from_id": "<parent experiment_id>",
      "to_id": "<child experiment_id>",
      "relation": "derives_from | scales_up | applies_optimum | varies_parameter | replaces_component | branches | parallel",
      "summary": "one sentence (Chinese OK) describing how the child differs from the parent"
    }}
  ]
}}

Relation vocabulary (pick the closest):
- applies_optimum:   child takes the parent's best strain/condition as its starting point
- scales_up:         child runs the same experiment at a larger scale (e.g. shake flask → 5L bioreactor)
- varies_parameter:  child explores a new independent variable using the parent's strain/setup
- replaces_component: child swaps one component (e.g. medium BMGY → BSM, signal peptide MFα → OST1)
- derives_from:      generic "child uses material/insight from parent" when no narrower label fits
- branches:          two children both derive from one parent (parallel exploration)
- parallel:          loosely related but no clear dependency — usually OMIT instead

Rules:
- Use ONLY the experiment_ids exactly as given.
- Do NOT emit self-edges or duplicate edges.
- Do NOT force every experiment to have a parent. Top-of-chain experiments
  (initial screening) often have none.
- Prefer fewer, well-justified edges over many speculative ones.
- The paper might describe two independent threads (e.g. one product line, one
  scale-up effort); both should appear as separate chains, not artificially merged.
""").strip()


def _format_experiment(e: ExperimentRun) -> str:
    sc = e.strain_construct
    g = e.goal
    o = e.outcome
    parts = [
        f"[{e.experiment_id}] {e.title or '(no title)'}",
    ]
    if e.paper_section:
        parts.append(f"  section: {e.paper_section}")
    if g.summary:
        parts.append(f"  goal: {g.summary}")
    if g.varied_parameters:
        parts.append(f"  varied: {'; '.join(g.varied_parameters)[:200]}")
    if g.fixed_parameters:
        parts.append(f"  fixed: {'; '.join(g.fixed_parameters)[:200]}")
    construct_bits: list[str] = []
    if sc.host_strain:
        construct_bits.append(f"host={sc.host_strain}")
    if sc.parent_strain:
        construct_bits.append(f"parent_strain={sc.parent_strain}")
    if sc.expression_vector:
        construct_bits.append(f"vector={sc.expression_vector}")
    if sc.signal_peptide:
        construct_bits.append(f"sp={sc.signal_peptide}")
    if sc.copy_number:
        construct_bits.append(f"copies={sc.copy_number}")
    if sc.target_products:
        construct_bits.append(f"product={'/'.join(sc.target_products)[:80]}")
    if construct_bits:
        parts.append("  construct: " + ", ".join(construct_bits))
    outcome_bits: list[str] = []
    if o.max_yield:
        outcome_bits.append(f"yield={o.max_yield}")
    if o.max_wet_cell_weight:
        outcome_bits.append(f"WCW={o.max_wet_cell_weight}")
    if outcome_bits:
        parts.append("  outcome: " + ", ".join(outcome_bits))
    return "\n".join(parts)


class LineageExtractor:
    """Single-call LLM extractor that produces lineage edges for one paper."""

    def __init__(
        self,
        domain: DomainContext,
        model: str = "gemini-2.5-pro",
        request_timeout_ms: int = 600_000,
    ) -> None:
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=request_timeout_ms),
        )
        self.model = model
        self.domain = domain
        self._system = _LINEAGE_SYSTEM.format(field_summary=domain.field_summary)

    def extract(self, paper_exps: PaperExperiments) -> list[ExperimentLineageEdge]:
        if len(paper_exps.experiments) < 2:
            return []

        block = "\n\n".join(_format_experiment(e) for e in paper_exps.experiments)
        valid_ids = {e.experiment_id for e in paper_exps.experiments}

        prompt = _LINEAGE_PROMPT.format(
            source_file=paper_exps.source_file,
            n=len(paper_exps.experiments),
            experiments_block=block,
        )

        response = None
        last_err: Exception | None = None
        for attempt in range(1, 5):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=self._system,
                        max_output_tokens=8192,
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )
                break
            except Exception as e:
                last_err = e
                msg = str(e)
                transient = ("503" in msg or "UNAVAILABLE" in msg
                             or "429" in msg or "RESOURCE_EXHAUSTED" in msg
                             or "500" in msg or "DEADLINE_EXCEEDED" in msg)
                if not transient or attempt == 4:
                    print(
                        f"  [warn] lineage API call failed for {paper_exps.source_file}: "
                        f"{type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    return []
                wait = 5 * (2 ** (attempt - 1))
                print(
                    f"  [retry {attempt}] transient API error, sleeping {wait}s",
                    file=sys.stderr,
                )
                time.sleep(wait)

        if response is None:
            print(f"  [warn] lineage extraction got no response: {last_err}", file=sys.stderr)
            return []

        raw = (response.text or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as e:
            print(
                f"  [warn] lineage JSON parse failed for {paper_exps.source_file}: {e}",
                file=sys.stderr,
            )
            print(f"         raw[:400]={raw[:400]!r}", file=sys.stderr)
            return []

        edges_raw = data.get("edges") or []
        edges: list[ExperimentLineageEdge] = []
        seen: set[tuple[str, str]] = set()
        for raw_edge in edges_raw:
            try:
                edge = ExperimentLineageEdge(**raw_edge)
            except Exception as e:
                print(f"  [warn] dropped invalid edge {raw_edge}: {e}", file=sys.stderr)
                continue
            if edge.from_id == edge.to_id:
                continue
            if edge.from_id not in valid_ids or edge.to_id not in valid_ids:
                print(
                    f"  [warn] edge references unknown id, skipped: {edge.from_id} → {edge.to_id}",
                    file=sys.stderr,
                )
                continue
            key = (edge.from_id, edge.to_id)
            if key in seen:
                continue
            seen.add(key)
            edges.append(edge)

        return edges
