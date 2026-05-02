"""Pretty-print one extracted experiment with its linked figures.

Usage:
    .venv/bin/python3 scripts/show_experiment.py                  # auto-pick the richest experiment
    .venv/bin/python3 scripts/show_experiment.py --paper 王婧
    .venv/bin/python3 scripts/show_experiment.py --paper 王婧 --exp wj-exp-06
    .venv/bin/python3 scripts/show_experiment.py --list           # list all paper-experiment ids
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

DATA = Path(__file__).resolve().parent.parent / "data" / "projects" / "pichia-collagen" / "structured"
FIG_JSON_DIR = DATA / "figures"

console = Console()


def short_paper(name: str) -> str:
    return name.split("_")[-1].replace(".pdf", "").replace(".experiments.json", "")


def load_papers() -> list[Path]:
    return sorted(DATA.glob("*.experiments.json"))


def figure_json(source_file: str, fid: str) -> dict | None:
    safe = source_file.replace("/", "_").replace(" ", "_")
    p = FIG_JSON_DIR / f"{safe}__{fid}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def richness(exp: dict) -> int:
    """Heuristic: prefer experiments with most phases + linked figs + filled outcome."""
    return (
        len(exp.get("phases", [])) * 3
        + len(exp.get("linked_figure_ids", [])) * 2
        + sum(1 for v in (exp.get("outcome") or {}).values() if v)
    )


def find_target(paper_filter: str | None, exp_filter: str | None) -> tuple[dict, dict] | None:
    candidates: list[tuple[dict, dict]] = []
    for path in load_papers():
        data = json.loads(path.read_text(encoding="utf-8"))
        if paper_filter and paper_filter not in data["source_file"]:
            continue
        for e in data.get("experiments", []):
            if exp_filter and exp_filter not in e.get("experiment_id", ""):
                continue
            candidates.append((data, e))
    if not candidates:
        return None
    candidates.sort(key=lambda pe: -richness(pe[1]))
    return candidates[0]


def list_all() -> None:
    table = Table(title="所有实验", box=box.SIMPLE_HEAVY)
    table.add_column("paper")
    table.add_column("exp_id")
    table.add_column("title", overflow="fold")
    table.add_column("phases", justify="right")
    table.add_column("figs", justify="right")
    table.add_column("yield")
    for path in load_papers():
        data = json.loads(path.read_text(encoding="utf-8"))
        paper = short_paper(data["source_file"])
        for e in data.get("experiments", []):
            table.add_row(
                paper,
                e.get("experiment_id", ""),
                (e.get("title") or "")[:60],
                str(len(e.get("phases", []))),
                str(len(e.get("linked_figure_ids", []))),
                (e.get("outcome") or {}).get("max_yield") or "—",
            )
    console.print(table)


def render_kv(title: str, items: dict, skip: tuple = ()) -> None:
    rows = [(k, v) for k, v in items.items() if v not in (None, "", [], {}) and k not in skip]
    if not rows:
        return
    table = Table(title=title, box=box.MINIMAL, show_header=False, title_style="bold cyan")
    table.add_column("k", style="bold")
    table.add_column("v", overflow="fold")
    for k, v in rows:
        if isinstance(v, list):
            v = "、".join(str(x) for x in v)
        table.add_row(k, str(v))
    console.print(table)


def render_phases(phases: list[dict]) -> None:
    if not phases:
        return
    table = Table(title=f"📊 阶段参数 ({len(phases)} 阶段)", box=box.SIMPLE_HEAD, title_style="bold cyan")
    cols = ["phase_name", "duration_hours", "temperature_celsius", "ph",
            "agitation_rpm", "aeration_vvm", "dissolved_oxygen_percent",
            "feeding_strategy", "notes"]
    for c in cols:
        table.add_column(c, overflow="fold")
    for p in phases:
        table.add_row(*[str(p.get(c) or "—") for c in cols])
    console.print(table)


def render_figure(source_file: str, fid: str) -> None:
    fd = figure_json(source_file, fid)
    if not fd:
        console.print(f"  [yellow]· {fid} (figure JSON missing)[/yellow]")
        return
    n_dp = len(fd.get("data_points") or [])
    n_np = len(fd.get("notable_points") or [])
    page = fd.get("page_number")
    cap = (fd.get("caption") or "").strip()[:120]
    console.print(f"  [bold]· {fid}[/bold]  (p{page}, {fd.get('figure_type')}, {n_dp} 数据点 / {n_np} 关键点)")
    if cap:
        console.print(f"    [dim]{cap}[/dim]")
    for np_ in (fd.get("notable_points") or [])[:4]:
        desc = np_.get("condition_description", "")
        val = np_.get("value_description", "")
        pt = np_.get("point_type", "")
        console.print(f"    [green]→[/green] ({pt}) {desc} → {val}")
    if fd.get("author_conclusion"):
        console.print(f"    [italic]结论:[/italic] {fd['author_conclusion'][:160]}")


def show(paper: dict, exp: dict) -> None:
    console.rule(f"[bold cyan]{exp['title']}")
    console.print(f"[dim]{paper['source_file']} · {exp['experiment_id']}[/dim]")
    if exp.get("description"):
        console.print(Panel(exp["description"], title="描述", style="dim"))

    render_kv("🎯 目标", exp.get("goal") or {})
    render_kv("🧬 菌株/载体构建", exp.get("strain_construct") or {})
    render_kv("⚗️ 发酵设置", exp.get("setup") or {})
    render_phases(exp.get("phases") or [])
    render_kv("🎁 实验结果", exp.get("outcome") or {})
    render_kv(
        "🔬 分析与纯化",
        {
            "purification_method": exp.get("purification_method"),
            "analytical_methods": exp.get("analytical_methods"),
        },
    )

    figs = exp.get("linked_figure_ids") or []
    if figs:
        console.print(f"\n[bold cyan]🖼️ 链接的图表 ({len(figs)} 张)[/bold cyan]")
        for fid in figs:
            render_figure(paper["source_file"], fid)
    else:
        console.print("\n[dim]无链接图表(可能是摇瓶筛选实验,数据以表格给出)[/dim]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", help="paper name substring (e.g. 王婧)")
    ap.add_argument("--exp", help="experiment_id substring (e.g. wj-exp-06)")
    ap.add_argument("--list", action="store_true", help="list all experiments and exit")
    args = ap.parse_args()

    if args.list:
        list_all()
        return

    found = find_target(args.paper, args.exp)
    if not found:
        console.print("[red]No matching experiment found.[/red]")
        return
    paper, exp = found
    show(paper, exp)


if __name__ == "__main__":
    main()
