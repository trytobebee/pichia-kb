"""EntityProvenance — traceability fields available to any entity type.

This is the one piece of "schema" that lives in framework code. Any
project entity type can declare `inherits: ["provenance"]` in its
JSON schema definition and pick up these fields automatically.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EntityProvenance(BaseModel):
    """Provenance fields for traceability of LLM-extracted entities."""

    sources: list[str] = Field(default_factory=list, description="Source filenames")
    chunk_ids: list[str] = Field(default_factory=list, description="Chunk IDs the entity was found in")
    raw_mention: str | None = Field(default=None, description="Verbatim mention from source")
    extraction_confidence: float | None = Field(
        default=None, description="LLM-reported confidence 0..1"
    )

    model_config = {"extra": "allow"}


# Registry of named base classes that an entity type can inherit from.
INHERITABLE_BASES: dict[str, type[BaseModel]] = {
    "provenance": EntityProvenance,
}
