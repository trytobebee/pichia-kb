"""schema curator — tools for managing project schemas via dialog with an LLM agent.

Phase 4a (this module): plain-Python mutation tools (add/remove/rename
field, add entity type) that write audit entries.
Phase 4b: inspection tools (find_papers_with_field, completeness, ...).
Phase 4c: the agent runtime (Gemini + tool calling).
Phase 4d: web chat panel.
"""

from .inspection import (
    compute_field_completeness,
    find_papers_with_field,
    query_schema_provenance,
)
from .mutations import (
    add_entity_type,
    add_field,
    record_rejection,
    remove_field,
    rename_field,
)

__all__ = [
    "add_entity_type",
    "add_field",
    "remove_field",
    "rename_field",
    "record_rejection",
    "find_papers_with_field",
    "compute_field_completeness",
    "query_schema_provenance",
]
