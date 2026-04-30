"""LLM-based structured knowledge extraction from paper chunks using Gemini."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

from google import genai
from google.genai import types

from ..schema import ExtractionResult, KnowledgeChunk
from .pdf_processor import PDFProcessor


_EXTRACTION_SYSTEM = textwrap.dedent("""
You are an expert in Pichia pastoris (Komagataella phaffii) / 毕赤酵母 molecular biology
and bioprocess engineering. Extract structured knowledge from research paper text.

The papers may be written in Chinese, English, or both. Extract ALL relevant entities
regardless of language. Be precise and faithful to the source text.
Return ONLY valid JSON — no explanation, no markdown fences.
If a field value is not mentioned, omit it or use null.
""").strip()

_EXTRACTION_PROMPT_TEMPLATE = textwrap.dedent("""
Extract structured Pichia pastoris (毕赤酵母) knowledge from the text below.
Return a JSON object. Only include keys that have data; omit empty arrays.

=== SCHEMA ===

"strains": list of host strains (菌株). Look for: GS115, X-33, X33, CBS7435, SMD1168,
  KM71, SMD1165, or any text like "毕赤酵母菌株", "宿主菌株", "出发菌株".
  {{
    "name": "strain name",
    "genotype": "genetic markers, e.g. his4, Mut+, Muts",
    "methanol_utilization": "Mut+ | Muts | Mut-",
    "protease_deficiency": "e.g. pep4, prb1 or null",
    "notes": "any relevant detail"
  }}

"promoters": transcriptional promoters (启动子). Look for: AOX1、AOX2、GAP、DAS1、
  DAS2、FLD1、PEX8 or terms like "启动子", "promoter", "P_AOX1", "PAOX1".
  {{
    "name": "promoter name",
    "expression_type": "constitutive | inducible | hybrid",
    "inducer": "methanol / formaldehyde / null",
    "strength": "relative strength description",
    "notes": "key characteristics"
  }}

"vectors": expression vectors / plasmids (表达载体/质粒). Look for: pPICZα, pGAPZ,
  pPIC9K, pAO815, pHIL-D2, or terms like "表达载体", "重组质粒", "重组载体".
  {{
    "name": "vector name",
    "promoter": "promoter used",
    "secretion_signal": "alpha-MF | PHO1 | OST1 | none | other",
    "selection_marker": "HIS4 | Zeocin | G418 | Blasticidin | other",
    "integration_site": "AOX1 | AOX2 | HIS4 | GAP | other",
    "notes": "key features"
  }}

"media": culture media (培养基). Look for: BMMY、BMGY、YPD、MD、MM、FM22、BSM、
  or terms like "培养基", "发酵培养基", "诱导培养基".
  {{
    "name": "medium name",
    "composition": {{"component": "amount/concentration"}},
    "carbon_source": "glycerol / methanol / glucose etc.",
    "nitrogen_source": "yeast extract / ammonium sulfate etc.",
    "ph_range": "e.g. 5.0–6.0",
    "purpose": "growth phase / induction phase / seed culture",
    "notes": ""
  }}

"fermentation_conditions": process parameters for each phase (发酵工艺条件).
  Look for temperature (温度), pH, DO (溶氧), agitation (转速/搅拌), feed strategy (流加策略),
  methanol induction (甲醇诱导), glycerol batch (甘油批培养), induction time (诱导时间).
  {{
    "phase": "e.g. glycerol batch / methanol induction",
    "mode": "batch | fed-batch | continuous | methanol_induction",
    "temperature_celsius": "value or range",
    "ph": "set-point or range",
    "dissolved_oxygen_percent": "e.g. >20%",
    "agitation_rpm": "value or range",
    "feeding_strategy": "description of carbon feed",
    "duration_hours": "duration if mentioned",
    "notes": "other relevant details"
  }}

"glycosylation_patterns": glycosylation info (糖基化). Look for: N-糖基化、O-糖基化、
  高甘露糖、甘露糖基化、glycosylation, mannose, OCH1 knockout.
  {{
    "glycosylation_type": "N-linked | O-linked | none",
    "typical_glycan_structure": "e.g. high-mannose Man8-14",
    "impact_on_activity": "effect on protein activity",
    "engineering_strategies": ["e.g. OCH1 knockout", "humanization"],
    "notes": ""
  }}

"target_products": recombinant proteins or metabolites being produced (目标产物).
  Look for: 胶原蛋白、collagen、重组蛋白、目的蛋白、产物, or specific protein names.
  {{
    "name": "product name",
    "type": "enzyme | collagen | antibody fragment | VLP | metabolite | other",
    "gene_source": "origin organism",
    "codon_optimized": true | false | null,
    "molecular_weight_kda": number or null,
    "expected_yield": "e.g. 500 mg/L",
    "desired_modifications": ["hydroxylation", "signal peptide cleavage", etc.],
    "activity_assay": "how activity is measured",
    "notes": ""
  }}

"process_parameters": quantitative process parameters (过程参数). Look for: qP (比生产速率)、
  qMeOH (甲醇消耗速率)、μ (比生长速率)、OD600 (光密度)、DCW (干细胞重)、titer (效价).
  {{
    "parameter_name": "full name",
    "symbol": "e.g. qP, μ, DCW",
    "unit": "unit",
    "typical_range": "value range",
    "optimal_value": "best known value",
    "effect_on_expression": "how this affects yield/quality",
    "notes": ""
  }}

"analytical_methods": characterization methods (分析方法). Look for: SDS-PAGE、
  Western blot、ELISA、HPLC、LC-MS、circular dichroism (CD)、胶原蛋白特异性检测.
  {{
    "name": "method name",
    "purpose": "what it measures",
    "sample_type": "culture supernatant / cell lysate / purified protein",
    "key_conditions": {{"param": "value"}},
    "interpretation_notes": "how to read results",
    "notes": ""
  }}

=== SOURCE INFO ===
Section: {section}
File: {source_file}

=== TEXT TO EXTRACT FROM ===
{text}
""").strip()


class KnowledgeExtractor:
    """Uses Gemini to extract structured entities from paper chunks."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        cache_dir: Path | None = None,
        request_timeout_ms: int = 120_000,
    ) -> None:
        # Without an explicit timeout the SDK can hang indefinitely on a
        # stalled connection — we lost a 70-min ingest run to this on
        # 2026-04-29. 120s is generous for a single chunk extraction.
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=request_timeout_ms),
        )
        self.model = model
        self.pdf_processor = PDFProcessor(cache_dir=cache_dir)

    def extract_from_pdf(self, pdf_path: Path, source_ref: str | None = None) -> ExtractionResult:
        chunks = self.pdf_processor.process(pdf_path)
        return self.extract_from_chunks(chunks, str(pdf_path.name), source_ref)

    def extract_from_chunks(
        self,
        chunks: list[KnowledgeChunk],
        source_file: str,
        source_ref: str | None = None,
    ) -> ExtractionResult:
        merged = ExtractionResult(source_file=source_file, source_ref=source_ref)
        total = len(chunks)
        print(f"  {total} chunks — extracting...")

        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}/{total}", end="\r")
            partial = self._extract_chunk(chunk)
            if partial:
                self._merge(merged, partial, chunk.source_file, chunk.chunk_id)

        print(f"  Done — {source_file}" + " " * 30)
        return merged

    # ── private ──────────────────────────────────────────────────────────────

    def _extract_chunk(self, chunk: KnowledgeChunk) -> dict | None:
        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(
            section=chunk.section or "unknown",
            source_file=chunk.source_file,
            text=chunk.content,
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_EXTRACTION_SYSTEM,
                    max_output_tokens=8192,
                    temperature=0.1,
                ),
            )
        except Exception as e:
            print(
                f"  [warn] API call failed for chunk={chunk.chunk_id}: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return None
        raw = (response.text or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(
                f"  [warn] JSON parse failed for chunk={chunk.chunk_id} "
                f"section={chunk.section!r}: {e}",
                file=sys.stderr,
            )
            print(f"         raw[:300]={raw[:300]!r}", file=sys.stderr)
            return None

    def _merge(
        self,
        target: ExtractionResult,
        data: dict,
        source_file: str,
        chunk_id: str,
    ) -> None:
        ref = [source_file]
        chunk_ref = [chunk_id]
        field_to_class = {
            "strains": "Strain",
            "promoters": "Promoter",
            "vectors": "ExpressionVector",
            "media": "CultureMedium",
            "fermentation_conditions": "FermentationConditionFact",
            "glycosylation_patterns": "GlycosylationPattern",
            "target_products": "TargetProduct",
            "process_parameters": "ProcessParameter",
            "analytical_methods": "AnalyticalMethod",
        }
        from .. import schema as schema_module
        for field_name, class_name in field_to_class.items():
            items = data.get(field_name, [])
            if not items:
                continue
            cls = getattr(schema_module, class_name)
            existing_list: list = getattr(target, field_name)
            for item_data in items:
                # The LLM occasionally emits ``None`` for list-typed fields
                # (e.g. ``desired_modifications: null``) and produces
                # placeholder entries with ``name: null`` that aren't really
                # entities. Drop None values so pydantic defaults kick in,
                # then skip silently if the identifying ``name`` is missing.
                item_data = {k: v for k, v in item_data.items() if v is not None}
                if "name" in cls.model_fields and not item_data.get("name"):
                    continue
                item_data.setdefault("sources", ref)
                item_data.setdefault("chunk_ids", chunk_ref)
                try:
                    existing_list.append(cls(**item_data))
                except Exception as e:
                    print(
                        f"  [warn] {class_name} validation failed: "
                        f"{type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    print(f"         data={item_data!r}", file=sys.stderr)
