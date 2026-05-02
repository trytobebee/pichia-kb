"""Generic ExtractionResult — replaces the per-domain hardcoded version.

Stores entities as a dict keyed by extraction_key (e.g. "strains"), so the
shape is determined by the project's knowledge.json rather than by code.

Each entity in the lists is a plain dict (already validated by the
dynamic Pydantic class at extraction time, then serialized for storage).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# Legacy top-level keys produced before 3b. Migrated into `entities` on load.
_LEGACY_TOP_LEVEL_KEYS = {
    "strains", "promoters", "vectors", "media",
    "fermentation_conditions", "glycosylation_patterns",
    "target_products", "process_parameters", "analytical_methods",
}


class ExtractionResult(BaseModel):
    """Structured entities extracted from a single document."""

    source_file: str
    source_ref: str | None = None
    extraction_notes: str = ""
    entities: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionResult":
        """Construct from a JSON dict, migrating legacy layouts in place."""
        data = dict(data)  # don't mutate caller's dict
        if "entities" not in data:
            data["entities"] = {}
        for k in list(data.keys()):
            if k in _LEGACY_TOP_LEVEL_KEYS:
                data["entities"].setdefault(k, data.pop(k))
        return cls(**data)
