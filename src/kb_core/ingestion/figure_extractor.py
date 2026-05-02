"""
Figure extraction pipeline.

Step 1 (pymupdf): extract figure images from PDF pages, save as PNG.
Step 2 (pdfplumber): extract captions and surrounding text for each figure.
Step 3 (Gemini vision): read the image + text context, return structured data.
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

import pdfplumber

from pydantic import BaseModel

from ..llm import get_llm
from .pdf_text import read_pdf_text


_FIGURE_SYSTEM = textwrap.dedent("""
You are an expert in {field_summary} and experimental data analysis.
Your task is to extract ALL quantitative information from a research figure or table,
structured so that an engineer can select process parameters from it.

Papers may be in Chinese or English. Extract everything regardless of language.
Return ONLY valid JSON matching the schema provided. Be as quantitative as possible.
If exact values cannot be read, give your best estimate and note the uncertainty.
""").strip()

_FIGURE_PROMPT = textwrap.dedent("""
Analyze this figure/table from a research paper and extract structured data.

CONTEXT:
- Source paper: {source_file}
- Page: {page_number}
- Figure ID: {figure_id}
- Caption: {caption}
- Surrounding text (experimental description and conclusions):
{surrounding_text}

Return a JSON object with this exact schema:

{{
  "figure_type": "line_curve|bar_chart|sds_page|table|scatter|heatmap|schematic|microscopy|other",

  "independent_variables": [
    {{
      "name": "variable name in Chinese or English",
      "unit": "unit string or null",
      "values": [list of discrete tested values, or null if continuous],
      "range_min": number or null,
      "range_max": number or null,
      "range_description": "e.g. '0-144h' or null"
    }}
  ],

  "dependent_variables": [
    {{
      "name": "variable name",
      "unit": "unit string or null"
    }}
  ],

  "fixed_conditions": {{
    "key": "value for any parameter held constant across this experiment"
  }},

  "data_points": [
    {{
      "conditions": {{"independent_var_name": value, ...}},
      "values": {{"dependent_var_name": numeric_value, ...}},
      "note": "optional note, e.g. 'estimated from graph' or null"
    }}
  ],

  "notable_points": [
    {{
      "condition_description": "e.g. 'copy_number=10, time=144h'",
      "value_description": "e.g. '0.70 g/L'",
      "point_type": "optimal|inflection|plateau_start|minimum|other",
      "note": "why this point matters"
    }}
  ],

  "observed_trend": "One sentence describing the shape/direction of the relationship",

  "interpolation_range": {{
    "description": "safe parameter range for industrial use based on this data",
    "parameters": {{"param_name": {{"min": value, "max": value, "recommended": value}}}}
  }},

  "author_conclusion": "What the authors concluded from this figure",

  "industrial_note": "How an engineer should use this data for parameter selection"
}}

IMPORTANT for data_points:
- For bar charts: one data_point per bar, conditions = {{x_label: value}}
- For line curves: extract all visible data points along each curve
- For time-series with multiple lines: include curve_label in conditions
- For SDS-PAGE: describe band pattern as data_points with lane labels and band MW
- For tables: each row is one data_point
- Estimate numeric values from visual inspection when exact numbers are not labeled
""").strip()


_CAPTION_PATTERNS = [
    re.compile(r'(图\s*\d+[-‐–—]\d+[^。\n]{0,200})', re.MULTILINE),
    re.compile(r'(Fig\.?\s*\d+[-‐–—]\d+[^\n]{0,200})', re.IGNORECASE | re.MULTILINE),
    re.compile(r'(Figure\s*\d+[^\n]{0,200})', re.IGNORECASE | re.MULTILINE),
    re.compile(r'(表\s*\d+[-‐–—]\d+[^。\n]{0,200})', re.MULTILINE),
    re.compile(r'(Table\s*\d+[^\n]{0,200})', re.IGNORECASE | re.MULTILINE),
]


def _find_captions(text: str) -> list[tuple[int, str]]:
    """Return list of (char_position, caption_text) found in text."""
    found = []
    for pat in _CAPTION_PATTERNS:
        for m in pat.finditer(text):
            found.append((m.start(), m.group(0).strip()))
    found.sort(key=lambda x: x[0])
    return found


def _surrounding_text(full_text: str, caption_pos: int, window: int = 1500) -> str:
    start = max(0, caption_pos - window)
    end = min(len(full_text), caption_pos + window)
    return full_text[start:end].strip()


class FigureExtractor:
    """Extracts figures from PDFs and uses Gemini vision to parse their data."""

    def __init__(
        self,
        figures_dir: Path,
        domain,  # DomainContext, kept untyped to avoid circular import
        figure_data_cls: type[BaseModel],
        model: str = "gemini-2.5-flash",
        dpi: int = 150,
        cache_dir: Path | None = None,
    ) -> None:
        self.figures_dir = figures_dir
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.dpi = dpi
        self.cache_dir = cache_dir
        self.llm = get_llm(model)
        self.domain = domain
        self.figure_data_cls = figure_data_cls
        self._system = _FIGURE_SYSTEM.format(field_summary=domain.field_summary)

    # ── public ───────────────────────────────────────────────────────────────

    def extract_from_pdf(self, pdf_path: Path) -> list[BaseModel]:
        """Extract all figures from a PDF and return structured FigureData list."""
        import fitz  # pymupdf

        stem = pdf_path.stem
        full_text = self._extract_full_text(pdf_path)
        captions = _find_captions(full_text)

        results: list[BaseModel] = []
        doc = fitz.open(str(pdf_path))

        fig_index = 0
        for page_num, page in enumerate(doc, start=1):
            images = page.get_images(full=True)
            if not images:
                continue

            page_text = self._page_text(pdf_path, page_num)

            for img_index, img_info in enumerate(images):
                xref = img_info[0]
                pix = fitz.Pixmap(doc, xref)

                # Skip small images likely to be decorations, icons, or separators
                # Require at least 150px in both dimensions
                if pix.width < 150 or pix.height < 150:
                    continue

                # Convert any non-RGB/RGBA colorspace to RGB for PNG saving
                if pix.colorspace and pix.colorspace not in (fitz.csRGB, fitz.csGRAY):
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                # Strip alpha if present (PNG with alpha is fine, but some
                # downstream tools don't support it — keep alpha as-is)

                fig_index += 1
                fig_id = f"Fig-p{page_num}-{img_index+1}"
                img_filename = f"{stem}__{fig_id}.png"
                img_path = self.figures_dir / img_filename
                pix.save(str(img_path))

                # Find nearest caption
                caption, ctx = self._find_context(
                    full_text, captions, page_text, fig_index
                )

                print(f"    [{fig_id}] {img_path.name} ({pix.width}×{pix.height})")

                # Vision extraction
                fd = self._extract_figure_data(
                    img_path=img_path,
                    source_file=pdf_path.name,
                    figure_id=fig_id,
                    page_number=page_num,
                    caption=caption,
                    surrounding_text=ctx,
                )
                results.append(fd)

        doc.close()
        return results

    # ── private ──────────────────────────────────────────────────────────────

    def _extract_full_text(self, pdf_path: Path) -> str:
        return read_pdf_text(pdf_path, cache_dir=self.cache_dir)

    def _page_text(self, pdf_path: Path, page_num: int) -> str:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num <= len(pdf.pages):
                t = pdf.pages[page_num - 1].extract_text()
                return t or ""
        return ""

    def _find_context(
        self,
        full_text: str,
        captions: list[tuple[int, str]],
        page_text: str,
        fig_index: int,
    ) -> tuple[str | None, str]:
        # Try to pick the fig_index-th caption
        if fig_index - 1 < len(captions):
            pos, cap = captions[fig_index - 1]
            ctx = _surrounding_text(full_text, pos)
            return cap, ctx

        # Fallback: use page text
        page_captions = _find_captions(page_text)
        if page_captions:
            pos, cap = page_captions[0]
            ctx = _surrounding_text(page_text, pos, window=800)
            return cap, ctx

        return None, page_text[:1500]

    # ── refinement using experiment context ──────────────────────────────────

    def refine_with_experiment_hints(
        self,
        img_path: Path,
        source_file: str,
        figure_id: str,
        page_number: int,
        caption: str | None,
        surrounding_text: str,
        expected_x_categories: list[str] | None = None,
        expected_y_metrics: list[str] | None = None,
        experiment_summary: str | None = None,
    ) -> BaseModel:
        """Re-extract a figure with prior knowledge of expected axis categories.

        Useful for bar charts where adjacent x-axis labels were merged in the
        first pass (e.g. "OST1Δ57-70" was actually "OST1-pre-α-pro" + "Δ57-70").
        Pass the experiment's varied_parameters values as `expected_x_categories`
        and the model is forced to align bars to that exact list.
        """
        return self._extract_figure_data(
            img_path=img_path,
            source_file=source_file,
            figure_id=figure_id,
            page_number=page_number,
            caption=caption,
            surrounding_text=surrounding_text,
            expected_x_categories=expected_x_categories or None,
            expected_y_metrics=expected_y_metrics or None,
            experiment_summary=experiment_summary,
        )

    def _extract_figure_data(
        self,
        img_path: Path,
        source_file: str,
        figure_id: str,
        page_number: int,
        caption: str | None,
        surrounding_text: str,
        expected_x_categories: list[str] | None = None,
        expected_y_metrics: list[str] | None = None,
        experiment_summary: str | None = None,
    ) -> BaseModel:
        prompt = _FIGURE_PROMPT.format(
            source_file=source_file,
            page_number=page_number,
            figure_id=figure_id,
            caption=caption or "(no caption found)",
            surrounding_text=surrounding_text[:2000],
        )

        # Prepend experiment-derived hints when available
        hint_parts: list[str] = []
        if experiment_summary:
            hint_parts.append(f"LINKED EXPERIMENT GOAL: {experiment_summary}")
        if expected_x_categories:
            n_expected = len(expected_x_categories)
            cats_numbered = "\n".join(
                f"  [{i}] {c}" for i, c in enumerate(expected_x_categories)
            )
            hint_parts.append(
                f"=== AUTHORITATIVE X-AXIS CATEGORIES (FROM PAPER BODY — DO NOT READ FROM IMAGE) ===\n"
                f"There are EXACTLY {n_expected} categories, in this LEFT-TO-RIGHT order:\n"
                f"{cats_numbered}"
            )
            hint_parts.append(
                f"=== HOW TO EXTRACT DATA POINTS (STRICT PROCEDURE) ===\n"
                f"DO NOT attempt to read the x-axis text labels in the image. They are unreliable\n"
                f"because CJK + Greek characters run together without spaces (e.g. you may visually\n"
                f"see 'OST1Δ57-70' as one token when the actual labels are 'OST1-pre-α-pro' AND\n"
                f"'Δ57-70' as TWO separate categories whose tick marks happen to be adjacent).\n"
                f"\n"
                f"Instead, follow this procedure:\n"
                f"  1. Look ONLY at the data area of the chart (above the x-axis line, below the\n"
                f"     plot frame top). Count the distinct vertical BAR GROUPS.\n"
                f"     - For dual-axis charts (e.g. yield + OD600), one 'group' = one x-position\n"
                f"       that contains a small cluster of 2 bars (one per metric). Count the\n"
                f"       CLUSTERS, not the individual bars.\n"
                f"  2. The expected count is N = {n_expected}. If your visual count differs, look\n"
                f"     more carefully — adjacent groups can be visually close. Trust the EXPECTED\n"
                f"     count over your initial visual count, unless you are extremely confident.\n"
                f"  3. Assign the categories to groups by INDEX:\n"
                f"     - Leftmost group → category [0]\n"
                f"     - 2nd from left  → category [1]\n"
                f"     - ...\n"
                f"     - Rightmost group → category [{n_expected - 1}]\n"
                f"  4. For each group, read the bar HEIGHTS against the y-axis to get values.\n"
                f"  5. Emit EXACTLY {n_expected} data_points, one per group, in the order [0]..[{n_expected - 1}].\n"
                f"     Use the EXACT category strings provided above (with original Greek/CJK)."
            )
        if expected_y_metrics:
            metrics = ", ".join(expected_y_metrics)
            hint_parts.append(
                f"EXPECTED Y-AXIS METRICS (figure may be dual-axis; each x-category should yield "
                f"one data_point with values for each metric): {metrics}"
            )
        if hint_parts:
            prompt = (
                "=== STRONG PRIOR FROM PAPER BODY (override what you read on the axes if conflicting) ===\n"
                + "\n\n".join(hint_parts)
                + "\n\n=== EXTRACTION TASK ===\n"
                + prompt
            )

        try:
            data = self.llm.chat_vision_json(
                prompt, img_path,
                system=self._system,
                temperature=0.1, max_tokens=8192,
            )
        except NotImplementedError:
            print(
                f"      [error] model {self.model} ({self.llm.provider}) does not "
                f"support vision input. Pick a vision model (gemini-*, qwen-vl-*, "
                f"doubao-*-vision-*, gpt-4o, ...).",
                file=sys.stderr,
            )
            data = {}
        except Exception as e:
            print(f"      [warn] vision extraction failed for {figure_id}: {e}", file=sys.stderr)
            data = {}

        # Map unknown figure types to "other" so a vendor-emitted novel label
        # (e.g. "mass_spectrometry", "gel_image") doesn't crash validation.
        _FIG_TYPES = {"line_curve", "bar_chart", "sds_page", "table",
                      "scatter", "heatmap", "schematic", "microscopy", "other"}
        ft = data.get("figure_type", "other")
        if ft not in _FIG_TYPES:
            ft = "other"

        return self.figure_data_cls(
            source_file=source_file,
            figure_id=figure_id,
            figure_type=ft,
            image_path=str(img_path.relative_to(img_path.parent.parent)),
            page_number=page_number,
            caption=caption,
            surrounding_text=surrounding_text[:2000],
            section=None,
            independent_variables=[
                v if isinstance(v, dict) else v
                for v in data.get("independent_variables", [])
            ],
            dependent_variables=[
                v if isinstance(v, dict) else v
                for v in data.get("dependent_variables", [])
            ],
            fixed_conditions=data.get("fixed_conditions", {}),
            data_points=[
                {"conditions": dp.get("conditions", {}),
                 "values": dp.get("values", {}),
                 "note": dp.get("note")}
                for dp in data.get("data_points", [])
            ],
            notable_points=[
                {"condition_description": np_.get("condition_description", ""),
                 "value_description": np_.get("value_description", ""),
                 "point_type": np_.get("point_type", "other"),
                 "note": np_.get("note")}
                for np_ in data.get("notable_points", [])
            ],
            observed_trend=data.get("observed_trend"),
            interpolation_range=data.get("interpolation_range"),
            author_conclusion=data.get("author_conclusion"),
            industrial_note=data.get("industrial_note"),
        )
