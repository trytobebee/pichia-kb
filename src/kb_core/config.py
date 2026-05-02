"""Per-project configuration loaded from data/projects/<slug>/config.yaml.

This is the place where domain-specific text lives. Framework code reads
fields from `ProjectConfig` and substitutes them into prompt templates,
so renaming the framework to a new domain only requires editing yaml.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DomainContext(BaseModel):
    """Domain-specific text injected into LLM prompts."""

    # One short noun phrase, e.g. "Pichia pastoris molecular biology and bioprocess engineering"
    expert_field: str

    # One short phrase describing what the papers are about,
    # e.g. "Pichia pastoris collagen expression"
    paper_topic: str

    # One short noun phrase describing the field at large, e.g.
    # "Pichia pastoris bioprocess engineering and fermentation control"
    field_summary: str

    # Free-form description shown in the QA system prompt to anchor expertise.
    qa_role_description: str = ""

    # Per-entity-type hints injected into the entity extractor prompt.
    # Keys are entity type names (strains, promoters, etc.). Values are
    # short descriptions and/or example tokens.
    entity_hints: dict[str, str] = Field(default_factory=dict)

    # Cross-paper entity types and their human-readable descriptions
    # used by the cross-normalizer when calling the LLM clustering step.
    cross_entity_descriptions: dict[str, str] = Field(default_factory=dict)


class ProjectConfig(BaseModel):
    """All per-project metadata. Loaded from `<project_dir>/config.yaml`."""

    slug: str
    name: str
    description: str = ""

    # Keywords used by the PDF processor to detect domain pages.
    keywords: list[str] = Field(default_factory=list)

    domain: DomainContext


_DEFAULT_DOMAIN = DomainContext(
    expert_field="the relevant scientific or technical field",
    paper_topic="the topic covered by the source papers",
    field_summary="the relevant scientific or technical field",
    qa_role_description=(
        "You are a research assistant. Answer questions strictly based "
        "on the retrieved context. If context is missing, say so clearly."
    ),
)


def load_project_config(project_dir: Path) -> ProjectConfig:
    """Load `config.yaml` from a project directory.

    If the file is missing we return a permissive default — the framework
    is still usable, just generic-sounding. Logging of the missing file
    is the caller's job.
    """
    cfg_path = project_dir / "config.yaml"
    if not cfg_path.is_file():
        return ProjectConfig(
            slug=project_dir.name,
            name=project_dir.name,
            domain=_DEFAULT_DOMAIN,
        )
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw.setdefault("slug", project_dir.name)
    raw.setdefault("name", project_dir.name)
    raw.setdefault("domain", _DEFAULT_DOMAIN.model_dump())
    return ProjectConfig(**raw)
