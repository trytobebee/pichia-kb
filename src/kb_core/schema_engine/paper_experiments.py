"""PaperExperiments — generic container for one paper's extracted experiments.

Holds whichever Pydantic instances the project's experiments schema
produces (dynamically-built classes). Consumers access fields via
attribute notation — the dynamic class makes that work transparently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class PaperExperiments(BaseModel):
    """All experiments + lineage edges extracted from one paper."""

    source_file: str
    extraction_notes: str | None = None
    experiments: list[Any] = Field(default_factory=list)
    lineage: list[Any] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def to_disk(self, path: Path) -> None:
        """Serialize to JSON file. Pydantic walks nested BaseModel instances."""
        data = {
            "source_file": self.source_file,
            "extraction_notes": self.extraction_notes,
            "experiments": [
                _to_dump(e) for e in self.experiments
            ],
            "lineage": [
                _to_dump(l) for l in self.lineage
            ],
        }
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_disk(
        cls,
        path: Path,
        experiment_cls: type[BaseModel] | None = None,
        lineage_cls: type[BaseModel] | None = None,
    ) -> "PaperExperiments":
        """Load from JSON file, optionally validating items through the
        provided dynamic Pydantic classes. Without classes, items remain
        as plain dicts (still attribute-accessible via SimpleNamespace? No —
        plain dicts; consumers should pass classes for typed access)."""
        data = json.loads(path.read_text(encoding="utf-8"))
        experiments_raw = data.get("experiments", []) or []
        lineage_raw = data.get("lineage", []) or []

        if experiment_cls is not None:
            experiments = [experiment_cls(**e) for e in experiments_raw]
        else:
            experiments = experiments_raw

        if lineage_cls is not None:
            lineage = [lineage_cls(**l) for l in lineage_raw]
        else:
            lineage = lineage_raw

        return cls(
            source_file=data["source_file"],
            extraction_notes=data.get("extraction_notes"),
            experiments=experiments,
            lineage=lineage,
        )


def _to_dump(item: Any) -> dict[str, Any]:
    """Serialize a Pydantic instance OR a plain dict uniformly."""
    if isinstance(item, BaseModel):
        return item.model_dump(exclude_none=True)
    if isinstance(item, dict):
        return item
    raise TypeError(f"Unexpected experiment item type: {type(item).__name__}")
