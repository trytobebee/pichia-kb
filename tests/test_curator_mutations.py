"""Tests for curator schema mutation tools (phase 4a)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kb_core.curator import (
    add_entity_type,
    add_field,
    record_rejection,
    remove_field,
    rename_field,
)


def _make_project(tmp: Path) -> Path:
    schema_dir = tmp / "schema"
    schema_dir.mkdir(parents=True)
    (schema_dir / "knowledge.json").write_text(
        json.dumps({
            "system": "knowledge",
            "version": 1,
            "entity_types": [
                {
                    "name": "Strain",
                    "inherits": ["provenance"],
                    "fields": [
                        {"name": "name", "type": "str", "required": True},
                        {"name": "genotype", "type": "str"},
                    ],
                }
            ],
        }),
        encoding="utf-8",
    )
    return tmp


def _read_schema(project_dir: Path, system: str = "knowledge") -> dict:
    return json.loads((project_dir / "schema" / f"{system}.json").read_text())


def _read_audit(project_dir: Path) -> list[dict]:
    p = project_dir / "schema_audit.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_add_field(tmp_path: Path):
    proj = _make_project(tmp_path)
    add_field(
        proj, "knowledge", "Strain",
        {"name": "copy_number", "type": "str", "description": "e.g. '10-copy'"},
        rationale="seen in 3 new papers",
    )
    spec = _read_schema(proj)
    fields = spec["entity_types"][0]["fields"]
    names = [f["name"] for f in fields]
    assert "copy_number" in names
    audit = _read_audit(proj)
    assert audit[0]["action"] == "add_field"
    assert audit[0]["target"]["field"] == "copy_number"
    assert audit[0]["rationale"] == "seen in 3 new papers"


def test_add_field_duplicate_raises(tmp_path: Path):
    proj = _make_project(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        add_field(proj, "knowledge", "Strain",
                  {"name": "name", "type": "str"})


def test_add_field_unknown_entity_raises(tmp_path: Path):
    proj = _make_project(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        add_field(proj, "knowledge", "Nonexistent",
                  {"name": "x", "type": "str"})


def test_remove_field(tmp_path: Path):
    proj = _make_project(tmp_path)
    remove_field(proj, "knowledge", "Strain", "genotype",
                 rationale="never used in 7 papers")
    fields = _read_schema(proj)["entity_types"][0]["fields"]
    assert all(f["name"] != "genotype" for f in fields)
    audit = _read_audit(proj)
    assert audit[0]["action"] == "remove_field"


def test_rename_field(tmp_path: Path):
    proj = _make_project(tmp_path)
    rename_field(proj, "knowledge", "Strain", "genotype", "marker_genes",
                 rationale="clearer term")
    fields = _read_schema(proj)["entity_types"][0]["fields"]
    names = [f["name"] for f in fields]
    assert "marker_genes" in names and "genotype" not in names
    audit = _read_audit(proj)
    assert audit[0]["action"] == "rename_field"
    assert audit[0]["target"]["from"] == "genotype"
    assert audit[0]["target"]["to"] == "marker_genes"


def test_rename_field_collision_raises(tmp_path: Path):
    proj = _make_project(tmp_path)
    add_field(proj, "knowledge", "Strain", {"name": "marker_genes", "type": "str"})
    with pytest.raises(ValueError, match="already exists"):
        rename_field(proj, "knowledge", "Strain", "genotype", "marker_genes")


def test_add_entity_type(tmp_path: Path):
    proj = _make_project(tmp_path)
    add_entity_type(
        proj, "knowledge", "Promoter",
        description="Transcriptional promoter.",
        inherits=["provenance"],
        fields=[{"name": "name", "type": "str", "required": True}],
        rationale="appears across all papers",
    )
    spec = _read_schema(proj)
    names = [et["name"] for et in spec["entity_types"]]
    assert "Promoter" in names
    audit = _read_audit(proj)
    assert audit[0]["action"] == "add_entity_type"


def test_add_entity_type_duplicate_raises(tmp_path: Path):
    proj = _make_project(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        add_entity_type(proj, "knowledge", "Strain")


def test_record_rejection_logs_only(tmp_path: Path):
    proj = _make_project(tmp_path)
    record_rejection(
        proj, "knowledge",
        proposal={"action": "add_field", "field": "useless_thing"},
        rationale="not relevant to this domain",
    )
    # Schema unchanged
    fields = _read_schema(proj)["entity_types"][0]["fields"]
    assert len(fields) == 2
    # Audit entry present
    audit = _read_audit(proj)
    assert audit[0]["action"] == "reject_proposal"
    assert audit[0]["target"]["proposal"]["field"] == "useless_thing"


def test_audit_entries_carry_actor_and_timestamp(tmp_path: Path):
    proj = _make_project(tmp_path)
    add_field(proj, "knowledge", "Strain", {"name": "x", "type": "str"},
              actor="user", rationale="test")
    audit = _read_audit(proj)
    assert audit[0]["actor"] == "user"
    assert "timestamp_iso" in audit[0]
