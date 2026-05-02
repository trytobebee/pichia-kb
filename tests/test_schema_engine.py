"""Tests for the dynamic schema engine.

Each test exercises a distinct meta-schema feature: scalars, enums,
lists, references, optional vs required, provenance inheritance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kb_core.schema_engine import (
    EntityTypeDefinition,
    FieldSpec,
    SchemaFile,
    build_class,
    build_models,
    load_project_schemas,
)


def test_scalar_fields_required_and_optional():
    et = EntityTypeDefinition(
        name="Strain",
        fields=[
            FieldSpec(name="name", type="str", required=True),
            FieldSpec(name="genotype", type="str", required=False),
            FieldSpec(name="copy_number", type="int", required=False),
        ],
    )
    Cls = build_class(et)
    obj = Cls(name="GS115")
    assert obj.name == "GS115"
    assert obj.genotype is None
    # required field missing → ValidationError
    with pytest.raises(ValidationError):
        Cls()


def test_enum_field_accepts_only_listed_values():
    et = EntityTypeDefinition(
        name="Promoter",
        fields=[
            FieldSpec(name="name", type="str", required=True),
            FieldSpec(
                name="expression_type",
                type="enum",
                enum_values=["constitutive", "inducible", "hybrid"],
            ),
        ],
    )
    Cls = build_class(et)
    Cls(name="AOX1", expression_type="inducible")
    with pytest.raises(ValidationError):
        Cls(name="AOX1", expression_type="random_value")


def test_list_field_with_string_items():
    et = EntityTypeDefinition(
        name="Vector",
        fields=[
            FieldSpec(name="name", type="str", required=True),
            FieldSpec(name="features", type="list", item_type="str"),
        ],
    )
    Cls = build_class(et)
    obj = Cls(name="pPIC9K", features=["AOX1", "His4"])
    assert obj.features == ["AOX1", "His4"]


def test_ref_field_stored_as_string():
    et = EntityTypeDefinition(
        name="ExperimentRun",
        fields=[
            FieldSpec(name="experiment_id", type="str", required=True),
            FieldSpec(name="host_strain_ref", type="ref:knowledge.Strain"),
        ],
    )
    Cls = build_class(et)
    obj = Cls(experiment_id="WJ-01", host_strain_ref="GS115")
    assert obj.host_strain_ref == "GS115"


def test_provenance_inheritance_adds_sources_and_chunk_ids():
    et = EntityTypeDefinition(
        name="Strain",
        inherits=["provenance"],
        fields=[FieldSpec(name="name", type="str", required=True)],
    )
    Cls = build_class(et)
    obj = Cls(name="GS115", sources=["paper.pdf"], chunk_ids=["abc123"])
    assert obj.sources == ["paper.pdf"]
    assert obj.chunk_ids == ["abc123"]
    assert obj.extraction_confidence is None


def test_unknown_inherits_target_raises():
    et = EntityTypeDefinition(
        name="X",
        inherits=["nonexistent_base"],
        fields=[FieldSpec(name="name", type="str", required=True)],
    )
    with pytest.raises(ValueError, match="Unknown inherits target"):
        build_class(et)


def test_build_models_returns_class_per_entity_type():
    sf = SchemaFile(
        system="knowledge",
        entity_types=[
            EntityTypeDefinition(
                name="Strain",
                fields=[FieldSpec(name="name", type="str", required=True)],
            ),
            EntityTypeDefinition(
                name="Promoter",
                fields=[FieldSpec(name="name", type="str", required=True)],
            ),
        ],
    )
    models = build_models(sf)
    assert set(models) == {"Strain", "Promoter"}
    assert models["Strain"](name="GS115").name == "GS115"


def test_load_project_schemas_with_all_three_files(tmp_path: Path):
    schema_dir = tmp_path / "schema"
    schema_dir.mkdir()

    knowledge = {
        "system": "knowledge",
        "entity_types": [
            {
                "name": "Strain",
                "inherits": ["provenance"],
                "fields": [{"name": "name", "type": "str", "required": True}],
            }
        ],
    }
    experiments = {
        "system": "experiments",
        "entity_types": [
            {
                "name": "ExperimentRun",
                "fields": [
                    {"name": "experiment_id", "type": "str", "required": True},
                    {"name": "host", "type": "ref:knowledge.Strain"},
                ],
            }
        ],
    }
    (schema_dir / "knowledge.json").write_text(json.dumps(knowledge), encoding="utf-8")
    (schema_dir / "experiments.json").write_text(json.dumps(experiments), encoding="utf-8")

    ps = load_project_schemas(tmp_path)
    assert "Strain" in ps.knowledge_models
    assert "ExperimentRun" in ps.experiments_models
    assert ps.data_spec is None  # missing file tolerated

    Strain = ps.knowledge_models["Strain"]
    Run = ps.experiments_models["ExperimentRun"]
    s = Strain(name="GS115", sources=["x.pdf"])
    r = Run(experiment_id="E1", host="GS115")
    assert s.sources == ["x.pdf"]
    assert r.host == "GS115"


def test_load_project_schemas_missing_dir_returns_empty():
    ps = load_project_schemas(Path("/nonexistent"))
    assert ps.knowledge_spec is None
    assert ps.knowledge_models == {}
