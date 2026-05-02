"""One-time migration: legacy flat FigureData → panels-based shape.

Old layout (data.json schema v1):
    {
      "figure_type": "...",
      "independent_variables": [...],
      "dependent_variables": [...],
      "data_points": [...],
      "notable_points": [...],
      "observed_trend": "...",
      ...
    }

New layout (data.json schema v2):
    {
      "panels": [{
        "panel_label": "",
        "figure_type": "...",
        "x_axis": {label, unit, range_description},
        "y_axis": {label, unit, range_description},
        "y_axis_secondary": null,
        "data_points": [...],
        "notable_points": [...],
        "observed_trend": "...",
        ...
      }],
      ... (figure-level fields untouched)
    }

The migration creates ONE panel per existing figure (since old format had
no concept of panels). Multi-panel figures will need re-extraction with
the new prompt to capture both panels — that's a separate manual step.

Usage:
    python scripts/migrate_figures_to_panels.py --project pichia-collagen
    python scripts/migrate_figures_to_panels.py --project pichia-collagen --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _axis_from_var(var: dict | None) -> dict | None:
    """Convert a legacy {name, unit, range_description, ...} dict to new
    {label, unit, range_description}. Returns None if the input is empty."""
    if not var or not isinstance(var, dict):
        return None
    label = var.get("name") or var.get("label") or ""
    if not label:
        return None
    return {
        "label": label,
        "unit": var.get("unit"),
        "range_description": var.get("range_description"),
    }


def migrate_one(data: dict) -> dict:
    """Return a new dict in the panels-based layout.

    If `data` already has 'panels', returned unchanged.
    """
    if "panels" in data:
        return data

    indep = data.get("independent_variables") or []
    dep = data.get("dependent_variables") or []

    panel = {
        "panel_label": "",
        "figure_type": data.get("figure_type") or "other",
        "x_axis": _axis_from_var(indep[0]) if indep else None,
        "y_axis": _axis_from_var(dep[0]) if dep else None,
        "y_axis_secondary": _axis_from_var(dep[1]) if len(dep) > 1 else None,
        "data_points": data.get("data_points") or [],
        "notable_points": data.get("notable_points") or [],
        "fitted_equation": None,
        "r_squared": None,
        "observed_trend": data.get("observed_trend"),
    }

    new = {
        "source_file": data.get("source_file"),
        "figure_id": data.get("figure_id"),
        "image_path": data.get("image_path"),
        "page_number": data.get("page_number"),
        "caption": data.get("caption"),
        "surrounding_text": data.get("surrounding_text"),
        "section": data.get("section"),
        "fixed_conditions": data.get("fixed_conditions") or {},
        "panels": [panel],
        "interpolation_range": data.get("interpolation_range"),
        "author_conclusion": data.get("author_conclusion"),
        "industrial_note": data.get("industrial_note"),
    }
    # Drop None top-level keys for cleanliness
    return {k: v for k, v in new.items() if v is not None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Project slug")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    fig_dir = repo / "data" / "projects" / args.project / "structured" / "figures"
    if not fig_dir.is_dir():
        print(f"No figures directory: {fig_dir}")
        return

    files = sorted(fig_dir.glob("*.json"))
    migrated = 0
    skipped = 0
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if "panels" in data:
            skipped += 1
            continue
        new = migrate_one(data)
        if not args.dry_run:
            f.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
        migrated += 1

    verb = "Would migrate" if args.dry_run else "Migrated"
    print(f"{verb} {migrated} files; {skipped} already in panels format. Total: {len(files)}.")


if __name__ == "__main__":
    main()
