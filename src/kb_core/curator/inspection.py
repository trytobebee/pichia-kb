"""Schema inspection tools — read-only over project data + audit log.

Used by the curator agent (and humans) to evaluate proposals before
making changes. None of these mutate state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from ..schema_engine import ExtractionResult


SystemName = Literal["knowledge", "experiments", "data"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _structured(project_dir: Path) -> Path:
    return project_dir / "structured"


def _is_filled(value: Any) -> bool:
    """A field is 'filled' if not None, not empty string, not empty container."""
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, dict, tuple, set)) and len(value) == 0:
        return False
    return True


def _load_extraction_files(project_dir: Path) -> list[tuple[str, dict]]:
    """Return list of (paper_filename, raw_data) for every extraction JSON.

    Skips the cross-paper aggregate files (process_knowledge,
    domain_knowledge, dialectical_review).
    """
    excluded = {"process_knowledge.json", "dialectical_review.json", "domain_knowledge.json"}
    out: list[tuple[str, dict]] = []
    for f in sorted(_structured(project_dir).glob("*.pdf.json")):
        if f.name in excluded:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out.append((data.get("source_file", f.name), data))
        except Exception:
            continue
    return out


def _load_experiment_files(project_dir: Path) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for f in sorted(_structured(project_dir).glob("*.experiments.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out.append((data.get("source_file", f.name), data))
        except Exception:
            continue
    return out


def _load_figure_files(project_dir: Path) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    fig_dir = _structured(project_dir) / "figures"
    if not fig_dir.is_dir():
        return out
    for f in sorted(fig_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out.append((data.get("source_file", f.name), data))
        except Exception:
            continue
    return out


# ── Public inspection API ────────────────────────────────────────────────────

def find_papers_with_field(
    project_dir: Path,
    system: SystemName,
    entity_type: str,
    field_name: str,
    extraction_key: str | None = None,
) -> list[dict]:
    """Find every recorded value for a field across all papers.

    For ``system='knowledge'``: ``extraction_key`` (e.g. 'strains') points
    to the entity bucket inside ExtractionResult.entities; if omitted we
    fall back to the lowercase plural of entity_type.

    For ``system='experiments'``: looks inside each ExperimentRun, supporting
    nested fields by dot-path (e.g. 'outcome.max_yield').

    For ``system='data'``: looks at FigureData top-level fields.

    Returns: list of {paper, value, entity_id (best-effort)}.
    """
    extraction_key = extraction_key or (entity_type.lower() + "s")
    results: list[dict] = []

    if system == "knowledge":
        for paper, raw in _load_extraction_files(project_dir):
            er = ExtractionResult.from_dict(raw)
            for i, ent in enumerate(er.entities.get(extraction_key, [])):
                if field_name in ent and _is_filled(ent.get(field_name)):
                    results.append({
                        "paper": paper,
                        "value": ent[field_name],
                        "entity_id": ent.get("name") or f"#{i}",
                    })

    elif system == "experiments":
        # Support dot-path for nested fields.
        path = field_name.split(".")
        for paper, raw in _load_experiment_files(project_dir):
            for i, exp in enumerate(raw.get("experiments", [])):
                cur: Any = exp
                for key in path:
                    if not isinstance(cur, dict):
                        cur = None
                        break
                    cur = cur.get(key)
                if _is_filled(cur):
                    results.append({
                        "paper": paper,
                        "value": cur,
                        "entity_id": exp.get("experiment_id") or f"#{i}",
                    })

    elif system == "data":
        for paper, fig in _load_figure_files(project_dir):
            if field_name in fig and _is_filled(fig.get(field_name)):
                results.append({
                    "paper": paper,
                    "value": fig[field_name],
                    "entity_id": fig.get("figure_id") or "",
                })

    return results


def compute_field_completeness(
    project_dir: Path,
    system: SystemName,
    entity_type: str,
    field_names: list[str],
    extraction_key: str | None = None,
) -> list[dict]:
    """For each field, return total/filled counts across all papers.

    Output: list of {field, total, filled, fill_rate}. fill_rate ∈ [0,1].
    """
    extraction_key = extraction_key or (entity_type.lower() + "s")
    counts: dict[str, dict[str, int]] = {f: {"total": 0, "filled": 0} for f in field_names}

    if system == "knowledge":
        for _, raw in _load_extraction_files(project_dir):
            er = ExtractionResult.from_dict(raw)
            for ent in er.entities.get(extraction_key, []):
                for f in field_names:
                    counts[f]["total"] += 1
                    if _is_filled(ent.get(f)):
                        counts[f]["filled"] += 1

    elif system == "experiments":
        for _, raw in _load_experiment_files(project_dir):
            for exp in raw.get("experiments", []):
                for f in field_names:
                    counts[f]["total"] += 1
                    cur: Any = exp
                    for key in f.split("."):
                        if not isinstance(cur, dict):
                            cur = None
                            break
                        cur = cur.get(key)
                    if _is_filled(cur):
                        counts[f]["filled"] += 1

    elif system == "data":
        for _, fig in _load_figure_files(project_dir):
            for f in field_names:
                counts[f]["total"] += 1
                if _is_filled(fig.get(f)):
                    counts[f]["filled"] += 1

    return [
        {
            "field": f,
            "total": c["total"],
            "filled": c["filled"],
            "fill_rate": (c["filled"] / c["total"]) if c["total"] else 0.0,
        }
        for f, c in counts.items()
    ]


def query_schema_provenance(
    project_dir: Path,
    *,
    entity_type: str | None = None,
    field_name: str | None = None,
    system: SystemName | None = None,
    actor: str | None = None,
) -> list[dict]:
    """Read schema_audit.jsonl, optionally filtering.

    Returns chronological list of audit entries (oldest first).
    """
    audit_path = project_dir / "schema_audit.jsonl"
    if not audit_path.is_file():
        return []
    out: list[dict] = []
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if system and entry.get("system") != system:
            continue
        if actor and entry.get("actor") != actor:
            continue
        target = entry.get("target") or {}
        if entity_type and target.get("entity_type") != entity_type and target.get("name") != entity_type:
            continue
        if field_name:
            # Match field, or rename's from/to
            if (
                target.get("field") != field_name
                and target.get("from") != field_name
                and target.get("to") != field_name
            ):
                continue
        out.append(entry)
    return out
