"""KB smoke tests (no API calls).

Schema-level entity validation moved to test_schema_engine.py once the
hardcoded Pydantic classes were replaced by per-project JSON schemas.
This file keeps the KB-level integration tests: vector store ingest and
structured store round-trip.
"""

from pathlib import Path
import tempfile

from kb_core.knowledge_base import KnowledgeBase
from kb_core.schema_engine import ExtractionResult, KnowledgeChunk


def test_kb_ingest_and_search():
    """Ingest one chunk and confirm semantic search returns it."""
    with tempfile.TemporaryDirectory() as tmp:
        kb = KnowledgeBase(data_dir=Path(tmp))
        chunk = KnowledgeChunk(
            chunk_id="abc",
            source_file="test_paper.pdf",
            section="Materials and Methods",
            content="GS115 was cultured in BMGY medium with methanol induction at 30°C.",
        )
        added = kb.ingest_chunks([chunk])
        assert added == 1

        hits = kb.semantic_search("methanol induction temperature", n=1)
        assert len(hits) == 1
        assert "GS115" in hits[0]["content"] or "methanol" in hits[0]["content"].lower()


def test_kb_structured_roundtrip():
    """Save an ExtractionResult through KB and read entities back."""
    with tempfile.TemporaryDirectory() as tmp:
        kb = KnowledgeBase(data_dir=Path(tmp))
        result = ExtractionResult(
            source_file="paper1.pdf",
            entities={
                "strains": [
                    {
                        "name": "CBS7435",
                        "genotype": "Mut+",
                        "methanol_utilization": "Mut+",
                        "sources": ["paper1.pdf"],
                    }
                ]
            },
        )
        kb.ingest_extraction(result)
        entities = kb.get_entities("strains")
        assert len(entities) == 1
        assert entities[0]["name"] == "CBS7435"
        assert entities[0]["_source_doc"] == "paper1.pdf"


def test_extraction_result_legacy_migration():
    """Loading a pre-3b layout (entity lists at top level) should work."""
    legacy = {
        "source_file": "old.pdf",
        "strains": [{"name": "X-33"}],
        "promoters": [{"name": "AOX1"}],
    }
    result = ExtractionResult.from_dict(legacy)
    assert result.entities["strains"][0]["name"] == "X-33"
    assert result.entities["promoters"][0]["name"] == "AOX1"
