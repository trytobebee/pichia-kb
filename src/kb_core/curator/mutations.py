"""Schema mutation tools — every change writes an audit entry.

These are plain Python functions exposed to the curator agent (phase 4c)
as tools. Tests in tests/test_curator_mutations.py.

Layout assumption: schema files live at
``<project_dir>/schema/{knowledge,experiments,data}.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from ..schema_engine import append_audit_entry


SystemName = Literal["knowledge", "experiments", "data"]


# ── Internals ────────────────────────────────────────────────────────────────

def _schema_path(project_dir: Path, system: SystemName) -> Path:
    return project_dir / "schema" / f"{system}.json"


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Schema file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _find_entity_type(spec: dict, entity_type: str) -> dict | None:
    for et in spec.get("entity_types", []):
        if et.get("name") == entity_type:
            return et
    return None


def _audit(
    project_dir: Path,
    actor: str,
    action: str,
    system: SystemName,
    target: dict,
    rationale: str | None,
    triggered_by: str | None,
) -> None:
    append_audit_entry(
        project_dir,
        {
            "actor": actor,  # "user" | "curator_agent" | "system"
            "action": action,  # e.g. "add_field", "rename_field"
            "system": system,
            "target": target,
            "rationale": rationale,
            "triggered_by": triggered_by,  # e.g. paper filename
        },
    )


# ── Public mutation API ──────────────────────────────────────────────────────

def add_entity_type(
    project_dir: Path,
    system: SystemName,
    name: str,
    description: str = "",
    inherits: list[str] | None = None,
    extraction_key: str | None = None,
    fields: list[dict] | None = None,
    *,
    actor: str = "curator_agent",
    rationale: str | None = None,
    triggered_by: str | None = None,
) -> None:
    """Append a new EntityTypeDefinition to a schema file."""
    path = _schema_path(project_dir, system)
    spec = _read(path)
    if _find_entity_type(spec, name) is not None:
        raise ValueError(f"Entity type '{name}' already exists in {system}.json")
    entry: dict[str, Any] = {"name": name, "description": description}
    if extraction_key:
        entry["extraction_key"] = extraction_key
    if inherits:
        entry["inherits"] = inherits
    entry["fields"] = fields or []
    spec.setdefault("entity_types", []).append(entry)
    _write(path, spec)
    _audit(
        project_dir, actor, "add_entity_type", system,
        {"name": name},
        rationale, triggered_by,
    )


def add_field(
    project_dir: Path,
    system: SystemName,
    entity_type: str,
    field: dict,
    *,
    actor: str = "curator_agent",
    rationale: str | None = None,
    triggered_by: str | None = None,
) -> None:
    """Append a new field to an existing entity type."""
    if "name" not in field or "type" not in field:
        raise ValueError("field dict must include 'name' and 'type'")
    path = _schema_path(project_dir, system)
    spec = _read(path)
    et = _find_entity_type(spec, entity_type)
    if et is None:
        raise ValueError(f"Entity type '{entity_type}' not found in {system}.json")
    fields = et.setdefault("fields", [])
    if any(f.get("name") == field["name"] for f in fields):
        raise ValueError(
            f"Field '{field['name']}' already exists on {entity_type}"
        )
    fields.append(field)
    _write(path, spec)
    _audit(
        project_dir, actor, "add_field", system,
        {"entity_type": entity_type, "field": field["name"]},
        rationale, triggered_by,
    )


def remove_field(
    project_dir: Path,
    system: SystemName,
    entity_type: str,
    field_name: str,
    *,
    actor: str = "curator_agent",
    rationale: str | None = None,
    triggered_by: str | None = None,
) -> None:
    """Remove a field from an entity type. Raises if not found."""
    path = _schema_path(project_dir, system)
    spec = _read(path)
    et = _find_entity_type(spec, entity_type)
    if et is None:
        raise ValueError(f"Entity type '{entity_type}' not found in {system}.json")
    before = len(et.get("fields", []))
    et["fields"] = [f for f in et.get("fields", []) if f.get("name") != field_name]
    if len(et["fields"]) == before:
        raise ValueError(f"Field '{field_name}' not found on {entity_type}")
    _write(path, spec)
    _audit(
        project_dir, actor, "remove_field", system,
        {"entity_type": entity_type, "field": field_name},
        rationale, triggered_by,
    )


def rename_field(
    project_dir: Path,
    system: SystemName,
    entity_type: str,
    old_name: str,
    new_name: str,
    *,
    actor: str = "curator_agent",
    rationale: str | None = None,
    triggered_by: str | None = None,
) -> None:
    """Rename a field on an entity type (semantically a label-only change)."""
    path = _schema_path(project_dir, system)
    spec = _read(path)
    et = _find_entity_type(spec, entity_type)
    if et is None:
        raise ValueError(f"Entity type '{entity_type}' not found in {system}.json")
    fields = et.get("fields", [])
    for f in fields:
        if f.get("name") == old_name:
            if any(other.get("name") == new_name for other in fields if other is not f):
                raise ValueError(f"Field '{new_name}' already exists on {entity_type}")
            f["name"] = new_name
            _write(path, spec)
            _audit(
                project_dir, actor, "rename_field", system,
                {"entity_type": entity_type, "from": old_name, "to": new_name},
                rationale, triggered_by,
            )
            return
    raise ValueError(f"Field '{old_name}' not found on {entity_type}")


def record_rejection(
    project_dir: Path,
    system: SystemName,
    proposal: dict,
    rationale: str,
    *,
    actor: str = "user",
    triggered_by: str | None = None,
) -> None:
    """Log a rejected proposal so the curator agent doesn't re-propose it.

    Doesn't modify the schema file — just audit log.
    """
    _audit(
        project_dir, actor, "reject_proposal", system,
        {"proposal": proposal},
        rationale, triggered_by,
    )
