"""EntityProvenance — traceability fields available to any entity type.

This is the one piece of "schema" that lives in framework code. Any
project entity type can declare `inherits: ["provenance"]` in its
JSON schema definition and pick up these fields automatically.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EntityProvenance(BaseModel):
    """Provenance fields for traceability of LLM-extracted entities."""

    # LLMs frequently emit numeric values for fields modeled as strings
    # (``temperature_celsius: 25`` instead of ``"25"``). Coerce silently
    # rather than reject — losing the fact is worse than losing the type.
    model_config = ConfigDict(coerce_numbers_to_str=True, extra="allow")

    sources: list[str] = Field(default_factory=list, description="Source filenames")
    chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs the entity was found in",
    )
    raw_mention: str | None = Field(
        default=None, description="Verbatim mention from source"
    )
    extraction_confidence: str | None = Field(
        default=None,
        description="'explicit' (stated outright) | 'inferred' (derived).",
    )


# Registry of named base classes that an entity type can inherit from.
INHERITABLE_BASES: dict[str, type[BaseModel]] = {
    "provenance": EntityProvenance,
}
