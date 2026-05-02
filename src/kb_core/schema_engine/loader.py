"""Disk loader: read a project's schema/<system>.json files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .dynamic import build_models
from .meta import SchemaFile


SystemName = Literal["knowledge", "experiments", "data"]
_SYSTEMS: tuple[SystemName, ...] = ("knowledge", "experiments", "data")


class ProjectSchemas(BaseModel):
    """All three systems for one project, both raw spec and built classes."""

    knowledge_spec: SchemaFile | None = None
    experiments_spec: SchemaFile | None = None
    data_spec: SchemaFile | None = None

    knowledge_models: dict[str, type[BaseModel]] = {}
    experiments_models: dict[str, type[BaseModel]] = {}
    data_models: dict[str, type[BaseModel]] = {}

    model_config = {"arbitrary_types_allowed": True}

    def all_models(self) -> dict[str, type[BaseModel]]:
        merged: dict[str, type[BaseModel]] = {}
        merged.update(self.knowledge_models)
        merged.update(self.experiments_models)
        merged.update(self.data_models)
        return merged


def _load_one(path: Path, system: SystemName) -> SchemaFile | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.setdefault("system", system)
    return SchemaFile(**raw)


def load_project_schemas(project_dir: Path) -> ProjectSchemas:
    """Load and build all three system schemas for one project.

    Missing files are tolerated (the project may not yet have all three).
    """
    schema_dir = project_dir / "schema"
    out = ProjectSchemas()

    for system in _SYSTEMS:
        spec = _load_one(schema_dir / f"{system}.json", system)
        if spec is None:
            continue
        models = build_models(spec)
        if system == "knowledge":
            out.knowledge_spec = spec
            out.knowledge_models = models
        elif system == "experiments":
            out.experiments_spec = spec
            out.experiments_models = models
        elif system == "data":
            out.data_spec = spec
            out.data_models = models
    return out
