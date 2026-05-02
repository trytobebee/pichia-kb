"""Dynamic Pydantic class generation from a parsed SchemaFile.

`build_models(schema_file)` walks each EntityTypeDefinition and uses
pydantic.create_model() to construct an actual Pydantic class. The
result is a mapping {entity_type.name: Class} the rest of the
framework consumes (extractors, structured store, validation, etc.).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model

from .meta import EntityTypeDefinition, FieldSpec, SchemaFile
from .provenance import INHERITABLE_BASES


def _python_type_for(field: FieldSpec) -> Any:
    """Map a FieldSpec.type string to the Python typing annotation."""
    t = field.type

    if t == "str":
        return str
    if t == "int":
        return int
    if t == "float":
        return float
    if t == "bool":
        return bool
    if t == "dict":
        return dict[str, Any]
    if t == "enum":
        # Build a Literal[...] from enum_values. typing.Literal accepts a
        # tuple via __getitem__ and unpacks it into Literal['a','b',...].
        if not field.enum_values:
            return str
        return Literal[tuple(field.enum_values)]  # type: ignore[valid-type]
    if t == "list":
        # Recursively map the item type — but item_type is a string with
        # the same vocabulary; build a fake FieldSpec to recurse cleanly.
        item_field = FieldSpec(name=field.name + "_item", type=field.item_type or "str")
        item_py = _python_type_for(item_field)
        return list[item_py]  # type: ignore[valid-type]
    if t.startswith("ref:"):
        # References are stored as plain string identifiers (the entity's
        # canonical name). The relationship lookup happens at query time,
        # not validation time.
        return str

    # Unknown type → fall back to Any so we don't crash on partial schemas.
    return Any


def _resolve_bases(inherits: list[str]) -> tuple[type[BaseModel], ...]:
    bases: list[type[BaseModel]] = []
    for name in inherits:
        if name not in INHERITABLE_BASES:
            raise ValueError(
                f"Unknown inherits target: '{name}'. "
                f"Available: {sorted(INHERITABLE_BASES)}"
            )
        bases.append(INHERITABLE_BASES[name])
    return tuple(bases) if bases else (BaseModel,)


def build_class(entity_type: EntityTypeDefinition) -> type[BaseModel]:
    """Construct a single Pydantic class from one EntityTypeDefinition."""
    field_defs: dict[str, tuple[Any, Any]] = {}

    for f in entity_type.fields:
        py_type = _python_type_for(f)
        if not f.required:
            py_type = Optional[py_type]
            default = f.default
        else:
            default = ...  # required sentinel for create_model

        field_info = Field(default=default, description=f.description)
        field_defs[f.name] = (py_type, field_info)

    bases = _resolve_bases(entity_type.inherits)
    model = create_model(entity_type.name, __base__=bases, **field_defs)  # type: ignore[arg-type,call-overload]
    if entity_type.description:
        model.__doc__ = entity_type.description
    return model


def build_models(schema_file: SchemaFile) -> dict[str, type[BaseModel]]:
    """Build all Pydantic classes for one schema file. Returns {name: class}."""
    out: dict[str, type[BaseModel]] = {}
    for et in schema_file.entity_types:
        out[et.name] = build_class(et)
    return out
