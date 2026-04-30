"""Basic schema and KB smoke tests (no API calls)."""

from pathlib import Path
import tempfile

from pichia_kb.schema import (
    Strain, Promoter, ExpressionVector, FermentationConditionFact,
    TargetProduct, GlycosylationPattern, KnowledgeChunk, ExtractionResult,
)
from pichia_kb.knowledge_base import KnowledgeBase


def test_strain_schema():
    s = Strain(name="GS115", genotype="his4", methanol_utilization="Mut+")
    assert s.name == "GS115"
    assert s.methanol_utilization == "Mut+"


def test_strain_schema_minimal():
    """A bare-name strain mention should validate (Layer 3 keeps partial facts)."""
    s = Strain(name="GS115")
    assert s.name == "GS115"
    assert s.genotype is None
    assert s.methanol_utilization is None


def test_strain_provenance_fields():
    s = Strain(
        name="GS115-10hlCOLIII",
        sources=["paper.pdf"],
        chunk_ids=["abc123"],
        raw_mention="GS115 transformant carrying 10 copies",
        extraction_confidence="explicit",
    )
    assert s.chunk_ids == ["abc123"]
    assert s.extraction_confidence == "explicit"


def test_fermentation_condition():
    fc = FermentationConditionFact(
        phase="methanol induction",
        mode="methanol_induction",
        temperature_celsius="28",
        ph="5.0",
        dissolved_oxygen_percent=">20",
    )
    assert fc.mode == "methanol_induction"


def test_fermentation_condition_partial():
    """A condition mention with only temperature should still validate."""
    fc = FermentationConditionFact(temperature_celsius="28")
    assert fc.phase is None
    assert fc.mode is None


def test_expression_vector_partial():
    """ExpressionVector with only a name was the largest source of dropped data."""
    v = ExpressionVector(name="pPICZα")
    assert v.promoter is None
    assert v.selection_marker is None
    assert v.integration_site is None


def test_extraction_result_serialization():
    result = ExtractionResult(
        source_file="test.pdf",
        strains=[Strain(name="X-33", genotype="wild-type", methanol_utilization="Mut+")],
    )
    json_str = result.model_dump_json()
    assert "X-33" in json_str


def test_kb_ingest_and_search():
    with tempfile.TemporaryDirectory() as tmp:
        kb = KnowledgeBase(data_dir=Path(tmp))
        chunk = KnowledgeChunk(
            chunk_id="test001",
            source_file="dummy.pdf",
            section="Methods",
            content=(
                "Pichia pastoris GS115 was transformed with pPICZα vector. "
                "Methanol induction was performed at 28°C, pH 5.0, with >20% DO. "
                "The AOX1 promoter drove expression of the recombinant protein."
            ),
            keywords=["pichia", "methanol", "AOX1"],
        )
        added = kb.ingest_chunks([chunk])
        assert added == 1

        hits = kb.semantic_search("methanol induction temperature", n=1)
        assert len(hits) == 1
        assert "GS115" in hits[0]["content"] or "methanol" in hits[0]["content"].lower()


def test_kb_structured_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        kb = KnowledgeBase(data_dir=Path(tmp))
        result = ExtractionResult(
            source_file="paper1.pdf",
            strains=[
                Strain(name="CBS7435", genotype="Mut+", methanol_utilization="Mut+")
            ],
        )
        kb.ingest_extraction(result)
        entities = kb.get_entities("strains")
        assert len(entities) == 1
        assert entities[0]["name"] == "CBS7435"
