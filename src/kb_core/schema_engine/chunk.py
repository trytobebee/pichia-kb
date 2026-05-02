"""KnowledgeChunk — a retrievable text fragment from a paper.

Lives in the framework (not in the project schema) because it's how
documents flow through ingestion regardless of domain.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class KnowledgeChunk(BaseModel):
    """A retrievable piece of knowledge from a paper or structured source."""

    chunk_id: str
    source_file: str = Field(description="PDF filename or source identifier")
    source_ref: Optional[str] = Field(default=None)
    section: Optional[str] = Field(default=None)
    content: str = Field(description="The raw or lightly cleaned text chunk")
    entity_types: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
