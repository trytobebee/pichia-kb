"""Tests for curator inspection tools (phase 4b)."""

from __future__ import annotations

import json
from pathlib import Path

from kb_core.curator import (
    compute_field_completeness,
    find_papers_with_field,
    query_schema_provenance,
    add_field,
    rename_field,
)


def _make_project_with_extractions(tmp: Path) -> Path:
    """Project with a simple knowledge schema + 2 paper extractions."""
    schema_dir = tmp / "schema"
    structured_dir = tmp / "structured"
    schema_dir.mkdir(parents=True)
    structured_dir.mkdir(parents=True)

    (schema_dir / "knowledge.json").write_text(json.dumps({
        "system": "knowledge",
        "entity_types": [
            {"name": "Strain", "fields": [
                {"name": "name", "type": "str", "required": True},
                {"name": "genotype", "type": "str"},
            ]}
        ],
    }), encoding="utf-8")

    (structured_dir / "paper1.pdf.json").write_text(json.dumps({
        "source_file": "paper1.pdf",
        "entities": {
            "strains": [
                {"name": "GS115", "genotype": "his4"},
                {"name": "X-33"},  # genotype missing
            ]
        }
    }), encoding="utf-8")

    (structured_dir / "paper2.pdf.json").write_text(json.dumps({
        "source_file": "paper2.pdf",
        "entities": {
            "strains": [
                {"name": "SMD1168", "genotype": "pep4"},
            ]
        }
    }), encoding="utf-8")

    return tmp


def test_find_papers_with_field_knowledge(tmp_path: Path):
    proj = _make_project_with_extractions(tmp_path)
    hits = find_papers_with_field(proj, "knowledge", "Strain", "genotype")
    # paper1's GS115 (his4) + paper2's SMD1168 (pep4); X-33 has no genotype
    assert len(hits) == 2
    papers = {h["paper"] for h in hits}
    assert papers == {"paper1.pdf", "paper2.pdf"}


def test_find_papers_with_field_returns_empty_for_unknown(tmp_path: Path):
    proj = _make_project_with_extractions(tmp_path)
    hits = find_papers_with_field(proj, "knowledge", "Strain", "made_up_field")
    assert hits == []


def test_compute_field_completeness(tmp_path: Path):
    proj = _make_project_with_extractions(tmp_path)
    rows = compute_field_completeness(
        proj, "knowledge", "Strain", ["name", "genotype"]
    )
    by_field = {r["field"]: r for r in rows}
    # 3 strains total across both papers
    assert by_field["name"]["total"] == 3
    assert by_field["name"]["filled"] == 3
    assert by_field["name"]["fill_rate"] == 1.0
    assert by_field["genotype"]["total"] == 3
    assert by_field["genotype"]["filled"] == 2
    assert abs(by_field["genotype"]["fill_rate"] - 2/3) < 1e-9


def test_query_schema_provenance(tmp_path: Path):
    proj = _make_project_with_extractions(tmp_path)
    add_field(proj, "knowledge", "Strain",
              {"name": "copy_number", "type": "str"},
              actor="curator_agent",
              rationale="seen in 2 papers")
    rename_field(proj, "knowledge", "Strain", "genotype", "marker_genes",
                 actor="user", rationale="clearer")

    # No filter — both entries
    all_entries = query_schema_provenance(proj)
    assert len(all_entries) == 2

    # Filter by field name (catches both add and rename's from/to)
    field_history = query_schema_provenance(proj, field_name="copy_number")
    assert len(field_history) == 1
    assert field_history[0]["action"] == "add_field"

    rename_history = query_schema_provenance(proj, field_name="genotype")
    assert len(rename_history) == 1
    assert rename_history[0]["action"] == "rename_field"

    # Filter by actor
    user_actions = query_schema_provenance(proj, actor="user")
    assert len(user_actions) == 1
    assert user_actions[0]["action"] == "rename_field"


def test_find_papers_experiments_supports_dotted_path(tmp_path: Path):
    structured = tmp_path / "structured"
    structured.mkdir(parents=True)
    (structured / "p1.experiments.json").write_text(json.dumps({
        "source_file": "p1.pdf",
        "experiments": [
            {"experiment_id": "p1-exp-01", "outcome": {"max_yield": "10.3 g/L"}},
            {"experiment_id": "p1-exp-02", "outcome": {"max_yield": None}},
        ]
    }), encoding="utf-8")
    hits = find_papers_with_field(tmp_path, "experiments", "ExperimentRun", "outcome.max_yield")
    assert len(hits) == 1
    assert hits[0]["entity_id"] == "p1-exp-01"
    assert hits[0]["value"] == "10.3 g/L"
