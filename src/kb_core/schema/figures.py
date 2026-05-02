"""Schema for figure and table data extracted from papers."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ExperimentalVariable(BaseModel):
    name: str
    unit: str | None = None
    values: list[Any] | None = None       # discrete tested values
    range_min: float | None = None
    range_max: float | None = None
    range_description: str | None = None  # e.g. "0-144h"


class DataPoint(BaseModel):
    """One measurement: maps condition labels to values."""
    conditions: dict[str, Any]   # e.g. {"copy_number": 10, "time_h": 144}
    values: dict[str, Any]       # e.g. {"yield_gL": 0.70, "OD600": 85}
    note: str | None = None


class NotablePoint(BaseModel):
    condition_description: str   # e.g. "copy=10, t=144h"
    value_description: str       # e.g. "0.70 g/L"
    point_type: str              # "optimal" | "inflection" | "plateau_start" | "minimum" | "other"
    note: str | None = None


class FigureData(BaseModel):
    """Structured data extracted from one figure or table."""

    # Identity
    source_file: str
    figure_id: str                          # e.g. "Fig.3-7" or "Table 2-1"
    figure_type: Literal[
        "line_curve", "bar_chart", "sds_page", "table",
        "scatter", "heatmap", "schematic", "microscopy", "other"
    ]
    image_path: str                         # relative path to saved PNG
    page_number: int

    # Context from surrounding text
    caption: str | None = None
    surrounding_text: str | None = None     # paragraph(s) describing this figure
    section: str | None = None             # e.g. "Results", "发酵工艺优化"

    # Experimental design
    independent_variables: list[ExperimentalVariable] = Field(default_factory=list)
    dependent_variables: list[ExperimentalVariable] = Field(default_factory=list)
    fixed_conditions: dict[str, Any] = Field(default_factory=dict)

    # Extracted data
    data_points: list[DataPoint] = Field(default_factory=list)
    notable_points: list[NotablePoint] = Field(default_factory=list)

    # Interpretation
    observed_trend: str | None = None
    interpolation_range: dict[str, Any] | None = None  # safe range for parameter selection
    author_conclusion: str | None = None
    industrial_note: str | None = None     # relevance for process parameter selection
