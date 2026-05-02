"""Stage runner — invoke one extraction stage on one paper.

Wraps the existing extractor classes with consistent task-log writes
and status detection. Used by the web "📄 论文管理" page; the CLI keeps
calling extractors directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ..config import load_project_config
from ..ingestion import (
    ExperimentExtractor,
    FigureExtractor,
    KnowledgeExtractor,
    PDFProcessor,
)
from ..knowledge_base import KnowledgeBase
from ..schema_engine import load_project_schemas
from .tasks import StageName, task


# ── Status detection ─────────────────────────────────────────────────────────

def stage_outputs_present(project_dir: Path, paper_filename: str, stage: StageName) -> bool:
    """Cheap filesystem check for whether a stage's outputs exist for this paper."""
    structured = project_dir / "structured"
    figures_dir = structured / "figures"
    stem = paper_filename.removesuffix(".pdf")
    safe = paper_filename.replace("/", "_").replace(" ", "_")

    if stage == "chunks":
        # No per-paper file; only check that the cache exists (proxy for ingest)
        cache = project_dir / "cache" / f"{stem}.txt"
        return cache.is_file()
    if stage == "entities":
        return (structured / f"{safe}.json").is_file()
    if stage == "figures":
        if not figures_dir.is_dir():
            return False
        # Look for at least one figure JSON tagged with this paper's stem
        return any(figures_dir.glob(f"{stem}*.json"))
    if stage == "experiments":
        return (structured / f"{safe.removesuffix('.pdf')}.experiments.json").is_file()
    return False


def status_matrix(project_dir: Path, paper_filenames: list[str]) -> dict[str, dict[str, bool]]:
    """Return {paper: {stage: bool}} for the given paper list."""
    return {
        p: {s: stage_outputs_present(project_dir, p, s)  # type: ignore[arg-type]
            for s in ("chunks", "entities", "figures", "experiments")}
        for p in paper_filenames
    }


# ── Stage runners ────────────────────────────────────────────────────────────

def run_chunks_and_entities(
    project_dir: Path,
    paper_path: Path,
    *,
    model: str = "gemini-2.5-flash",
) -> int:
    """Layer 2 + Layer 3: chunk + embed + entity extraction. Returns chunk count."""
    cfg = load_project_config(project_dir)
    schemas = load_project_schemas(project_dir)
    if schemas.knowledge_spec is None:
        raise RuntimeError("Project has no schema/knowledge.json")

    kb = KnowledgeBase(data_dir=project_dir)
    cache_dir = project_dir / "cache"

    with task(project_dir, paper_path.name, "chunks") as ctx:
        processor = PDFProcessor(cache_dir=cache_dir, keywords=cfg.keywords)
        chunks = processor.process(paper_path)
        kb.ingest_chunks(chunks)
        ctx["notes"] = f"{len(chunks)} chunks"
        ctx["chunks"] = chunks  # pass through to next stage

    with task(project_dir, paper_path.name, "entities") as ctx:
        extractor = KnowledgeExtractor(
            domain=cfg.domain,
            knowledge_spec=schemas.knowledge_spec,
            knowledge_models=schemas.knowledge_models,
            model=model,
            cache_dir=cache_dir,
            keywords=cfg.keywords,
        )
        result = extractor.extract_from_chunks(chunks, paper_path.name)
        kb.ingest_extraction(result)
        ctx["llm_calls"] = len(chunks)
        ctx["notes"] = f"{sum(len(v) for v in result.entities.values())} entities"

    return len(chunks)


def run_figures(
    project_dir: Path,
    paper_path: Path,
    *,
    model: str = "gemini-2.5-flash",
) -> int:
    """Layer 5: figure extraction. Returns number of figures saved."""
    cfg = load_project_config(project_dir)
    schemas = load_project_schemas(project_dir)
    fig_cls = schemas.data_models.get("FigureData")
    if fig_cls is None:
        raise RuntimeError("Project has no FigureData in schema/data.json")

    kb = KnowledgeBase(data_dir=project_dir)
    figures_dir = project_dir / "figures"

    with task(project_dir, paper_path.name, "figures") as ctx:
        extractor = FigureExtractor(
            figures_dir=figures_dir,
            domain=cfg.domain,
            figure_data_cls=fig_cls,
            model=model,
            cache_dir=project_dir / "cache",
        )
        figures = extractor.extract_from_pdf(paper_path)
        n = kb.structured_store.save_figure_data(figures)
        ctx["llm_calls"] = len(figures)
        ctx["notes"] = f"{n} figures saved"

    return n


def run_experiments(
    project_dir: Path,
    paper_path: Path,
    *,
    model: str = "gemini-2.5-pro",
) -> int:
    """Layer 6: experiment extraction. Returns number of experiments."""
    cfg = load_project_config(project_dir)
    schemas = load_project_schemas(project_dir)
    exp_cls = schemas.experiments_models.get("ExperimentRun")
    if exp_cls is None:
        raise RuntimeError("Project has no ExperimentRun in schema/experiments.json")

    kb = KnowledgeBase(data_dir=project_dir)

    with task(project_dir, paper_path.name, "experiments") as ctx:
        extractor = ExperimentExtractor(
            domain=cfg.domain,
            experiment_run_cls=exp_cls,
            model=model,
            cache_dir=project_dir / "cache",
        )
        # Pass already-extracted figures so the LLM can link them.
        figures = kb.structured_store.load_figures(source_file=paper_path.name)
        paper_exps = extractor.extract_from_pdf(paper_path, figures=figures)
        kb.structured_store.save_experiments(paper_exps)
        ctx["llm_calls"] = 1
        ctx["notes"] = f"{len(paper_exps.experiments)} experiments"

    return len(paper_exps.experiments)


# ── Convenience: run a stage by name ─────────────────────────────────────────

_STAGE_FNS = {
    "chunks": run_chunks_and_entities,  # chunks alone is rarely useful; combine
    "entities": run_chunks_and_entities,
    "figures": run_figures,
    "experiments": run_experiments,
}


def run_stage(
    project_dir: Path,
    paper_path: Path,
    stage: Literal["entities", "figures", "experiments"],
    *,
    model: str | None = None,
) -> int:
    """Dispatch a stage by name. 'entities' transparently runs chunks first."""
    fn = _STAGE_FNS.get(stage)
    if fn is None:
        raise ValueError(f"Unknown stage: {stage}")
    if model is None:
        return fn(project_dir, paper_path)
    return fn(project_dir, paper_path, model=model)
