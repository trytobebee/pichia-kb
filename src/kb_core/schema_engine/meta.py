"""Meta-schema: Pydantic models that describe what a project schema looks like.

A project's `schema/<system>.json` (knowledge / experiments / data) is parsed
into one `SchemaFile`. Each `EntityTypeDefinition` inside it is what the
dynamic builder turns into a real Pydantic class at runtime.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# Primitive types the framework understands. Anything else is a reference
# (e.g. "ref:knowledge.Strain") or invalid.
ScalarType = Literal["str", "int", "float", "bool"]
ContainerType = Literal["list", "dict"]
SpecialType = Literal["enum"]


class FieldSpec(BaseModel):
    """Describes one field of an entity type."""

    name: str
    # type is one of:
    #   - scalar:  "str" | "int" | "float" | "bool"
    #   - container: "list" | "dict"  (use item_type for list element type)
    #   - "enum"   (use enum_values to enumerate allowed strings)
    #   - "ref:<system>.<EntityType>"  — reference by canonical name (stored as str)
    type: str
    required: bool = False
    default: Any = None
    description: str | None = None
    examples: list[Any] = Field(default_factory=list)

    # type-specific options
    enum_values: list[str] | None = Field(
        default=None, description="When type='enum'; allowed string values"
    )
    item_type: str | None = Field(
        default=None,
        description="When type='list'; element type. Recursively one of the type strings.",
    )

    def model_post_init(self, _ctx: Any) -> None:
        if self.type == "enum" and not self.enum_values:
            raise ValueError(
                f"FieldSpec '{self.name}' has type='enum' but no enum_values"
            )
        if self.type == "list" and self.item_type is None:
            # Default to list[str] when unspecified; loose but pragmatic
            self.item_type = "str"


class EntityTypeDefinition(BaseModel):
    """One entity type in a project schema."""

    name: str  # PascalCase, used as the dynamic Pydantic class name
    description: str | None = None
    extraction_key: str | None = Field(
        default=None,
        description="The JSON key the LLM emits these entities under, e.g. 'strains' for Strain. Defaults to lowercase(name)+'s'.",
    )
    inherits: list[str] = Field(
        default_factory=list,
        description="Names from INHERITABLE_BASES, e.g. ['provenance']",
    )
    fields: list[FieldSpec] = Field(default_factory=list)

    def resolved_extraction_key(self) -> str:
        return self.extraction_key or (self.name.lower() + "s")


class SchemaFile(BaseModel):
    """Top-level container for one schema file (knowledge / experiments / data)."""

    system: Literal["knowledge", "experiments", "data"]
    version: int = 1
    entity_types: list[EntityTypeDefinition] = Field(default_factory=list)
