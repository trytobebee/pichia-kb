"""schema_engine — meta-schema + dynamic Pydantic generation.

Public API:

    from kb_core.schema_engine import (
        FieldSpec, EntityTypeDefinition, SchemaFile,
        build_class, build_models,
        load_project_schemas,
        EntityProvenance, INHERITABLE_BASES,
    )

A project's domain schema lives in JSON files at
`data/projects/<slug>/schema/{knowledge,experiments,data}.json`.
"""

from .audit import append_audit_entry
from .chunk import KnowledgeChunk
from .dialectical import (
    ConflictPoint,
    ConsensusPoint,
    DialecticalReview,
    EvidenceStrength,
    PaperPosition,
    TopicSynthesis,
)
from .dynamic import build_class, build_models
from .extraction_result import ExtractionResult
from .loader import ProjectSchemas, load_project_schemas
from .meta import EntityTypeDefinition, FieldSpec, SchemaFile
from .paper_experiments import PaperExperiments
from .provenance import INHERITABLE_BASES, EntityProvenance

__all__ = [
    "FieldSpec",
    "EntityTypeDefinition",
    "SchemaFile",
    "build_class",
    "build_models",
    "ProjectSchemas",
    "load_project_schemas",
    "EntityProvenance",
    "INHERITABLE_BASES",
    "ExtractionResult",
    "PaperExperiments",
    "KnowledgeChunk",
    "DialecticalReview",
    "TopicSynthesis",
    "ConsensusPoint",
    "ConflictPoint",
    "PaperPosition",
    "EvidenceStrength",
    "append_audit_entry",
]
