"""Generate summary report of all extracted experiments per paper.

Run: .venv/bin/python3 scripts/experiment_summary.py
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path


DATA = Path(__file__).resolve().parent.parent / "data" / "projects" / "pichia-collagen" / "structured"
FIG_DIR = DATA / "figures"


def short_paper(name: str) -> str:
    """Return author-name suffix from paper filename."""
    return name.split("_")[-1].replace(".pdf", "").replace(".experiments.json", "")


def field_filled(d: dict, *keys) -> int:
    """Count how many of these dotted-paths have non-empty values."""
    n = 0
    for k in keys:
        cur = d
        for part in k.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                cur = None
                break
        if cur not in (None, "", [], {}):
            n += 1
    return n


def figure_data_points(figure_id: str, source_file: str) -> int:
    """Look up data_points count for a given figure_id in the source paper."""
    safe = source_file.replace("/", "_").replace(" ", "_")
    fpath = FIG_DIR / f"{safe}__{figure_id}.json"
    if not fpath.exists():
        return 0
    try:
        d = json.loads(fpath.read_text(encoding="utf-8"))
        return len(d.get("data_points") or [])
    except Exception:
        return 0


def main() -> None:
    paper_files = sorted(DATA.glob("*.experiments.json"))
    print("=" * 100)
    print(f"{'Paper':<22} {'#Exp':<6} {'#Phases':<10} {'Figs':<6} {'#DataPts':<10} {'#Yield':<8} {'#Phase-detail'}")
    print("=" * 100)

    grand_exp = 0
    grand_dp = 0
    grand_phases = 0
    paper_details: list[tuple[str, dict]] = []

    for f in paper_files:
        d = json.loads(f.read_text(encoding="utf-8"))
        paper = short_paper(d["source_file"])
        exps = d.get("experiments", [])
        n_exp = len(exps)
        total_phases = sum(len(e.get("phases", [])) for e in exps)
        all_figs = []
        for e in exps:
            all_figs.extend(e.get("linked_figure_ids", []))
        total_dp = sum(figure_data_points(fid, d["source_file"]) for fid in all_figs)
        n_yield = sum(1 for e in exps if e.get("outcome", {}).get("max_yield"))
        # phase fill: how many phases have ≥3 fields filled
        phase_detail = 0
        for e in exps:
            for p in e.get("phases", []):
                filled = sum(
                    1 for k in (
                        "duration_hours", "temperature_celsius", "ph",
                        "agitation_rpm", "aeration_vvm", "dissolved_oxygen_percent",
                        "feeding_strategy",
                    )
                    if p.get(k)
                )
                if filled >= 3:
                    phase_detail += 1

        grand_exp += n_exp
        grand_dp += total_dp
        grand_phases += total_phases
        paper_details.append((paper, {"exps": exps, "src": d["source_file"]}))

        print(f"{paper:<22} {n_exp:<6} {total_phases:<10} {len(all_figs):<6} {total_dp:<10} {n_yield:<8} {phase_detail}")

    print("=" * 100)
    print(f"{'TOTAL':<22} {grand_exp:<6} {grand_phases:<10} {'':<6} {grand_dp:<10}")

    # Per-experiment detail per paper
    print("\n\n" + "=" * 100)
    print("Per-experiment detail")
    print("=" * 100)
    for paper, info in paper_details:
        print(f"\n### {paper}  ({len(info['exps'])} experiments)\n")
        for e in info["exps"]:
            sc = e.get("strain_construct", {})
            setup = e.get("setup", {})
            outcome = e.get("outcome", {})
            phases = e.get("phases", [])
            figs = e.get("linked_figure_ids", [])
            dp_count = sum(figure_data_points(fid, info["src"]) for fid in figs)

            host = sc.get("host_strain", "?")
            vector = sc.get("expression_vector", "?")
            promoters = ",".join(sc.get("promoters", [])) or "?"
            scale = setup.get("scale", "?")
            yield_v = outcome.get("max_yield", "?")
            wcw = outcome.get("max_wet_cell_weight", "?")
            t_max = outcome.get("time_to_max_yield_hours", "?")

            print(f"  [{e.get('experiment_id')}] {e.get('title','')[:80]}")
            print(f"    构建: {host} / {vector} / {promoters}")
            print(f"    规模: {scale}")
            print(f"    阶段({len(phases)}): {' → '.join(p['phase_name'] for p in phases) or '—'}")
            print(f"    产量: {yield_v}  WCW: {wcw}  时间: {t_max}")
            print(f"    曲线: {len(figs)} 张图, 共 {dp_count} 个 data_points")
            varied = e.get("goal", {}).get("varied_parameters", [])
            if varied:
                print(f"    研究: {'; '.join(varied)[:140]}")


if __name__ == "__main__":
    main()
