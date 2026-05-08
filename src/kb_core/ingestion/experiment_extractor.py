"""
Experiment extraction pipeline.

Reads a paper's full text (cached via read_pdf_text) and the paper's
already-extracted figure list, then asks Gemini to enumerate all distinct
fermentation/expression experiments described in the paper, returning each
as a complete ExperimentRun (goal + construct + setup + phases + outcome
+ linked figures).

Single LLM call per paper — full-paper context is needed to identify
discrete experiments and tie them to figures.
"""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path

from pydantic import BaseModel

from ..config import DomainContext
from ..llm import get_llm
from ..schema_engine import PaperExperiments
from .pdf_text import read_pdf_text


_EXPERIMENT_SYSTEM = textwrap.dedent("""
You are an expert in {field_summary}. Your task
is to read a research paper and enumerate every distinct experiment it describes,
capturing each as a complete parameter snapshot (工艺单).

Each experiment is the unit at which an engineer would reproduce the work — it
includes the construct / setup (scale, medium, inoculum), the parameters of each
phase, the reported outcome (max yield, biomass, time-to-peak), and a back-reference
to any figures whose curves quantify this experiment.

═══════════════════════════════════════════════════════════════════
GRANULARITY RULE — what counts as ONE experiment
═══════════════════════════════════════════════════════════════════
ONE experiment = ONE fermentation run with a unique parameter snapshot.
Do NOT collapse multiple runs into a single "experimental stage".

A 60-100 page thesis-style paper typically yields 8-15 experiments:
  - each signal peptide variant tested  → one experiment per variant
  - each copy number tested             → one experiment per copy number
  - each induction temperature/pH tested→ one experiment per setpoint
  - each scale-up step (shake-flask → bioreactor → fed-batch)  → one each
  - each strain rebuild with new genetic mod  → one experiment

ANTI-PATTERN — do NOT do any of these:
  ✗ Merging "induction time optimization + purification + characterization"
    into one experiment. Time optimization is its OWN experiment(s);
    purification is operations on the product, not a fermentation run.
  ✗ Using "experimental stage" or "chapter" as the experiment unit.
    A single thesis chapter often contains 3-5 distinct fermentation runs.
  ✗ Lumping a parameter sweep (e.g. 5 different methanol concentrations)
    into a single experiment. Each setpoint is a separate run.

If you find yourself emitting fewer than 5 experiments for a 50+ page
thesis-style paper, re-read the methods/results sections and split.
For genuinely short papers (e.g. ≤ 20 pages) or pure reviews, fewer
or zero experiments is fine — record that in extraction_notes.

═══════════════════════════════════════════════════════════════════
LANGUAGE RULE — write in the paper's language
═══════════════════════════════════════════════════════════════════
Papers in this project are in CHINESE. ALL free-text field values
(title, description, goal.summary, notes, etc.) MUST be written in
Chinese. Do NOT translate the paper's Chinese content into English.

Keep technical names exactly as printed:
  - strain IDs (GS115, X-33, SMD1168)
  - vector / promoter / gene symbols (pPIC9K, AOX1, COL3A1, hlCOLIII)
  - units (g/L, °C, vvm, h)
  - chemical abbreviations (PTM1, BSM, BMGY)

Examples of correct free-text style:
  ✅ title: "10 拷贝菌株摇瓶诱导优化"
  ✅ description: "在 BMMY 培养基中比较不同甲醇浓度对 hlCOLIII 表达量的影响"
  ✅ goal.summary: "确定最佳诱导温度"
  ❌ title: "Methanol Concentration Optimization for hlCOLIII Expression"
  ❌ description: "This experiment compared..."

Return ONLY valid JSON. Be quantitative. If a value is not stated,
use null or [].
""").strip()


_EXPERIMENT_PROMPT = textwrap.dedent("""
Source paper: {source_file}

The paper has the following figures already extracted (use these figure_ids when
linking each experiment to the figures whose curves quantify it):

{figures_listing}

Read the paper's full text below and emit a JSON object with this schema:

{{
  "extraction_notes": "any caveats, ambiguities, or gaps you noticed",
  "experiments": [
    {{
      "experiment_id": "stable paper-local id, e.g. '<author-prefix>-exp-01' (use 2-letter prefix)",
      "title": "short label (≤ 12 words)",
      "description": "1-3 sentences plain-text summary of what was done",
      "paper_section": "section/chapter name if discernible",

      "goal": {{
        "summary": "one-sentence purpose",
        "fixed_parameters": ["parameter=value", ...],
        "varied_parameters": ["parameter: value list or range", ...],
        "observation_targets": ["measured output 1", ...]
      }},

      "construct": {{
        "host_strain": "e.g. GS115 or null",
        "parent_strain": "if derived from another strain",
        "expression_vector": "e.g. pPIC9K",
        "promoters": ["AOX1"] or ["AOX1", "DAS2"] for dual,
        "signal_peptide": "e.g. MFα",
        "tag": "e.g. 6×His",
        "selection": "e.g. G418 high-copy screening",
        "target_products": ["product 1", "product 2 if co-expressed"],
        "product_variants": ["detailed description matching each target_product"],
        "copy_number": "e.g. '10 copies'"
      }},

      "setup": {{
        "scale": "e.g. '5 L bioreactor', '250 mL shake flask'",
        "initial_medium": "e.g. BSM, BMGY",
        "inoculum_percent": "e.g. '4%'",
        "inoculum_od600": "e.g. 'OD600 = 1.0'",
        "seed_culture_notes": "free text"
      }},

      "phases": [
        {{
          "phase_name": "glycerol_batch | glycerol_fed_batch | starvation | methanol_induction | shake_flask | seed_culture | harvest | other",
          "duration_hours": "e.g. '24h' or '24-30h'",
          "temperature_celsius": "e.g. '30' or '25-28'",
          "ph": "e.g. '5.0' or '6.0±0.1'",
          "agitation_rpm": "e.g. '600' or '1000'",
          "aeration_vvm": "e.g. '1' or '2-3'",
          "dissolved_oxygen_percent": "e.g. '>20%' or 'DO-stat 40%'",
          "feeding_strategy": "e.g. '50% glycerol + 1.2% PTM1, 18 mL/h/L' or 'DO-stat methanol'",
          "notes": "anything else"
        }}
      ],

      "outcome": {{
        "max_yield": "e.g. '1.05 g/L' or '10.3 g·L⁻¹'",
        "time_to_max_yield_hours": "e.g. '66h' or '120h'",
        "max_wet_cell_weight": "e.g. '270 g/L'",
        "max_dry_cell_weight": "if reported",
        "max_od600": "if reported",
        "productivity": "if reported, e.g. mg/L/h",
        "purity": "if reported",
        "activity": "biological activity result if measured",
        "other_results": ["any other numerical findings"]
      }},

      "linked_figure_ids": ["Fig-pXX-Y", ...],
      "purification_method": "e.g. 'Ni-NTA → SEC → lyophilization' or null",
      "analytical_methods": ["SDS-PAGE", "BCA", "qPCR", ...]
    }}
  ]
}}

CRITICAL extraction guidance:
- ONLY include experiments performed by the authors of THIS paper. Do NOT include
  experiments described in the literature review, industry case studies cited from
  third-party sources, or references to other groups' work. If a yield, strategy,
  or outcome is attributed to "某厂家", "某公司", "literature reports",
  "Smith et al.", a numbered reference like "[37]", a review of prior work, or a
  general industry survey, it is a CITATION, not an experiment of this paper, and
  must be EXCLUDED.
- An ExperimentRun is a FERMENTATION / EXPRESSION run only. Do NOT create separate
  ExperimentRuns for: downstream purification protocol comparisons (e.g. ammonium
  sulfate saturation screens), bioactivity / cell-based assays (HUVEC proliferation,
  migration, cell viability), animal toxicity tests (oral, percutaneous, etc.),
  structural characterization assays (CD spectroscopy, mass spec, SEM), or
  formulation studies. These belong inside a fermentation run's
  `purification_method` / `analytical_methods` / `outcome.activity` fields, NOT as
  standalone runs. The varied_parameters list of an ExperimentRun must describe
  fermentation conditions only — never purification or assay parameters.
- One experiment per distinct (varied parameter) combination. If the paper compares
  10 strains under the same conditions, that's ONE experiment with varied_parameters
  = ['菌株: 1-10']. If the paper then takes the best strain to 5 L bioreactor, that's
  a SECOND experiment. If the paper varies methanol induction duration (0/24/48/72h)
  in the SAME run, that's a single ExperimentRun with that variation listed.
- For each experiment, look for figures whose caption or surrounding text matches the
  experimental conditions and add their figure_id to linked_figure_ids.
- For phases: list them in temporal order. Even if the paper doesn't separate phases
  explicitly, infer reasonable phase boundaries (e.g. glycerol batch → induction).
- Be quantitative. Prefer numerical values over descriptions. Use "null" for missing
  values, not made-up defaults.
- If the experiment is shake-flask only (no phase breakdown), use a single phase with
  phase_name = "shake_flask".

=== PAPER FULL TEXT ===
{paper_text}
""").strip()


class ExperimentExtractor:
    """Single-call LLM extractor that produces ExperimentRun objects from a paper."""

    def __init__(
        self,
        domain: DomainContext,
        experiment_run_cls: type[BaseModel],
        model: str = "gemini-2.5-pro",
        cache_dir: Path | None = None,
        max_text_chars: int = 200_000,
        request_timeout_ms: int = 600_000,
    ) -> None:
        self.llm = get_llm(model)
        self.model = model
        self.cache_dir = cache_dir
        self.max_text_chars = max_text_chars
        self.domain = domain
        self.experiment_run_cls = experiment_run_cls
        self._system = _EXPERIMENT_SYSTEM.format(field_summary=domain.field_summary)

    def extract_from_pdf(
        self,
        pdf_path: Path,
        figures: list[dict] | None = None,
    ) -> PaperExperiments:
        """Extract all experiments from a paper.

        ``figures`` is a list of figure dicts (typically from StructuredStore.load_figures).
        Only id / type / caption / variables are sent to the LLM to keep prompt size bounded.
        """
        text = read_pdf_text(pdf_path, cache_dir=self.cache_dir)
        if len(text) > self.max_text_chars:
            text = text[: self.max_text_chars]

        figures_listing = self._format_figures(figures or [])

        prompt = _EXPERIMENT_PROMPT.format(
            source_file=pdf_path.name,
            figures_listing=figures_listing or "(no figures extracted yet)",
            paper_text=text,
        )

        last_err: Exception | None = None
        data = None
        for attempt in range(1, 5):  # up to 4 attempts with backoff
            try:
                data = self.llm.chat_json(
                    prompt, system=self._system,
                    temperature=0.1, max_tokens=32768,
                )
                break
            except Exception as e:
                last_err = e
                msg = str(e)
                # Retry only on transient overload / rate-limit / 5xx
                transient = ("503" in msg or "UNAVAILABLE" in msg
                             or "429" in msg or "RESOURCE_EXHAUSTED" in msg
                             or "500" in msg or "DEADLINE_EXCEEDED" in msg)
                if not transient or attempt == 4:
                    print(
                        f"  [warn] experiment extraction API call failed for {pdf_path.name}: "
                        f"{type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    return PaperExperiments(
                        source_file=pdf_path.name,
                        extraction_notes=f"API failure after {attempt} attempts: {e}",
                    )
                wait = 5 * (2 ** (attempt - 1))  # 5, 10, 20s
                print(
                    f"  [retry {attempt}] transient API error, sleeping {wait}s: {type(e).__name__}",
                    file=sys.stderr,
                )
                time.sleep(wait)

        if data is None:
            return PaperExperiments(
                source_file=pdf_path.name,
                extraction_notes=f"API failure: {last_err}",
            )

        return self._build_paper_experiments(pdf_path.name, data)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _format_figures(figures: list[dict]) -> str:
        if not figures:
            return ""
        lines = []
        for fig in figures:
            fid = fig.get("figure_id", "?")
            page = fig.get("page_number", "?")
            cap = (fig.get("caption") or "")[:120]
            panels = fig.get("panels") or []
            ftype = panels[0].get("figure_type", "?") if panels else "?"
            # Combine axis labels across panels (most figures have one
            # panel; this still works for multi-panel A/B figures).
            axes_bits: list[str] = []
            for p in panels:
                x = (p.get("x_axis") or {}).get("label", "")
                y = (p.get("y_axis") or {}).get("label", "")
                if x or y:
                    axes_bits.append(f"{x} → {y}")
            axes = f" | {' / '.join(axes_bits)}" if axes_bits else ""
            lines.append(f"- {fid} (p{page}, {ftype}){axes}: {cap}")
        return "\n".join(lines)

    def _build_paper_experiments(self, source_file: str, data: dict) -> PaperExperiments:
        """Validate each experiment via the dynamic ExperimentRun class."""
        exps_raw = data.get("experiments", []) or []
        experiments: list = []
        # Some legacy LLM outputs use 'construct' instead of 'strain_construct'.
        for i, raw in enumerate(exps_raw, 1):
            raw = dict(raw)
            if "construct" in raw and "strain_construct" not in raw:
                raw["strain_construct"] = raw.pop("construct")
            raw.setdefault("experiment_id", f"exp-{i:02d}")
            raw.setdefault("title", raw.get("description", "")[:60] or f"Experiment {i}")
            raw.setdefault("sources", [source_file])
            try:
                experiments.append(self.experiment_run_cls(**raw))
            except Exception as e:
                print(
                    f"  [warn] failed to build ExperimentRun from raw entry: {e}",
                    file=sys.stderr,
                )
                continue

        return PaperExperiments(
            source_file=source_file,
            experiments=experiments,
            extraction_notes=data.get("extraction_notes"),
        )
