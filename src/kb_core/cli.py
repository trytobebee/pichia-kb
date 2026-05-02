"""Command-line interface for the kb-core knowledge construction framework."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import typer
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from .knowledge_base import KnowledgeBase
from .ingestion import (PDFProcessor, KnowledgeExtractor, ProcessKnowledgeSynthesizer,
                        DialecticalReviewer, FigureExtractor, DomainKnowledgeSynthesizer,
                        normalize_all, build_registry, save_registry,
                        ExperimentExtractor, LineageExtractor)
from .qa import PichiaAssistant

app = typer.Typer(
    name="kb",
    help="kb-core — AI-assisted knowledge construction framework: ingest papers, query a project's KB",
    no_args_is_help=True,
)
console = Console()

# Project layout: data/projects/<slug>/{papers,db,cache,figures,structured,...}
_PROJECTS_ROOT = Path(__file__).parent.parent.parent / "data" / "projects"


def _resolve_project(slug: str) -> Path:
    """Return the project directory for a slug. Errors out if it doesn't exist."""
    proj = _PROJECTS_ROOT / slug
    if not proj.is_dir():
        existing = sorted(p.name for p in _PROJECTS_ROOT.iterdir() if p.is_dir()) if _PROJECTS_ROOT.is_dir() else []
        msg = f"Project '{slug}' not found at {proj}."
        if existing:
            msg += f" Existing projects: {', '.join(existing)}"
        else:
            msg += " No projects exist yet."
        raise typer.BadParameter(msg)
    return proj


def _get_kb(project_dir: Path) -> KnowledgeBase:
    return KnowledgeBase(data_dir=project_dir)


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    pdf: Path = typer.Argument(..., help="Path to PDF paper (or directory of PDFs)"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    source_ref: str = typer.Option("", help="DOI or citation for this paper"),
    model: str = typer.Option("gemini-2.5-flash", help="Claude model for extraction"),
):
    """Ingest a PDF paper (or all PDFs in a directory) into the knowledge base."""
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    cache_dir = data_dir / "cache"
    extractor = KnowledgeExtractor(model=model, cache_dir=cache_dir)
    processor = PDFProcessor(cache_dir=cache_dir)

    pdfs: list[Path] = []
    if pdf.is_dir():
        pdfs = sorted(pdf.glob("*.pdf"))
    elif pdf.suffix.lower() == ".pdf":
        pdfs = [pdf]
    else:
        console.print(f"[red]Not a PDF or directory: {pdf}[/red]")
        raise typer.Exit(1)

    if not pdfs:
        console.print("[yellow]No PDF files found.[/yellow]")
        raise typer.Exit(0)

    for p in pdfs:
        console.print(Panel(f"Ingesting: [bold]{p.name}[/bold]", style="blue"))
        ref = source_ref or p.stem

        # Vector store: add raw chunks
        chunks = processor.process(p)
        added = kb.ingest_chunks(chunks)
        console.print(f"  [green]Chunks added to vector store:[/green] {added} / {len(chunks)}")

        # Structured store: LLM extraction
        result = extractor.extract_from_chunks(chunks, str(p.name), ref or None)
        kb.ingest_extraction(result)
        console.print(f"  [green]Structured entities saved.[/green]")

    console.print("\n[bold green]Ingestion complete.[/bold green]")
    _print_summary(kb)


@app.command()
def add(
    pdf: Path = typer.Argument(..., help="PDF file or directory to add"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    model: str = typer.Option("gemini-2.5-flash", help="Gemini model"),
):
    """Add a new paper: ingest + synthesize in one step.

    Run 'review' separately after adding several papers to update
    the cross-paper dialectical synthesis.
    """
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    cache_dir = data_dir / "cache"
    extractor = KnowledgeExtractor(model=model, cache_dir=cache_dir)
    processor = PDFProcessor(cache_dir=cache_dir)
    synth = ProcessKnowledgeSynthesizer(model=model, cache_dir=cache_dir)

    pdfs: list[Path] = sorted(pdf.glob("*.pdf")) if pdf.is_dir() else [pdf]
    if not pdfs:
        console.print("[yellow]No PDF files found.[/yellow]")
        raise typer.Exit(0)

    for p in pdfs:
        console.print(Panel(f"Adding: [bold]{p.name}[/bold]", style="blue"))

        # Step 1: ingest
        chunks = processor.process(p)
        added = kb.ingest_chunks(chunks)
        console.print(f"  Chunks → vector store: {added}/{len(chunks)}")
        result = extractor.extract_from_chunks(chunks, p.name, p.stem)
        kb.ingest_extraction(result)
        console.print(f"  Entities extracted.")

        # Step 2: synthesize
        data = synth.synthesize_from_pdf(p)
        kb.ingest_process_knowledge(p.name, data)
        n_p = len(data.get("control_principles", []))
        n_s = len(data.get("process_stages", []))
        console.print(f"  Control principles: {n_p}  Process stages: {n_s}")

    console.print("\n[bold green]Done.[/bold green] "
                  "Run [cyan]kb review[/cyan] when ready to update cross-paper synthesis.")
    _print_summary(kb)


@app.command()
def status(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
):
    """Show knowledge base statistics."""
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    _print_summary(kb)


@app.command()
def normalize(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change without writing"),
):
    """Layer 3.5: deduplicate and merge entities within each paper's extraction JSON."""
    data_dir = _resolve_project(project)
    structured_dir = data_dir / "structured"
    verb = "Would remove" if dry_run else "Removed"

    all_stats = normalize_all(structured_dir, dry_run=dry_run)

    table = Table(
        title=f"Normalization {'(dry-run) ' if dry_run else ''}results",
        show_lines=True,
    )
    table.add_column("Paper", overflow="fold", max_width=35)
    table.add_column("strains", justify="right")
    table.add_column("promoters", justify="right")
    table.add_column("vectors", justify="right")
    table.add_column("media", justify="right")
    table.add_column("products", justify="right")
    table.add_column("params", justify="right")
    table.add_column("methods", justify="right")
    table.add_column("total", justify="right", style="bold green")

    totals: dict[str, int] = {}
    for fname, stats in all_stats.items():
        total = sum(stats.values())
        totals_sum = totals.get('total', 0) + total
        totals['total'] = totals_sum
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v

        short = fname.replace('.pdf.json', '')[-35:]
        table.add_row(
            short,
            str(stats.get('strains', 0)),
            str(stats.get('promoters', 0)),
            str(stats.get('vectors', 0)),
            str(stats.get('media', 0)),
            str(stats.get('target_products', 0)),
            str(stats.get('process_parameters', 0)),
            str(stats.get('analytical_methods', 0)),
            str(total),
        )

    table.add_row(
        "[bold]TOTAL[/bold]",
        str(totals.get('strains', 0)),
        str(totals.get('promoters', 0)),
        str(totals.get('vectors', 0)),
        str(totals.get('media', 0)),
        str(totals.get('target_products', 0)),
        str(totals.get('process_parameters', 0)),
        str(totals.get('analytical_methods', 0)),
        str(totals.get('total', 0)),
    )
    console.print(table)

    if dry_run:
        console.print("\n[yellow]Dry-run: no files changed.[/yellow]")
    else:
        console.print(f"\n[bold green]Done.[/bold green] {verb} {totals.get('total', 0)} duplicate records across all papers.")


@app.command("build-registry")
def build_registry_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    model: str = typer.Option("gemini-2.5-flash", help="Gemini model for synonym clustering"),
):
    """Layer 3.5: build cross-paper entity registry (rule-based + LLM synonym clustering)."""
    data_dir = _resolve_project(project)
    structured_dir = data_dir / "structured"
    console.print(Panel(
        "Building cross-paper entity registry.\n"
        f"LLM synonym clustering: target_products, analytical_methods, promoters\n"
        f"Model: [bold]{model}[/bold]",
        style="magenta",
        title="Entity Registry",
    ))

    registry = build_registry(structured_dir=structured_dir, model=model)
    out_path = save_registry(registry, structured_dir)

    table = Table(title="Registry summary", show_lines=True)
    table.add_column("Entity type", style="cyan")
    table.add_column("Canonical entities", justify="right")
    table.add_column("Cross-paper (2+ papers)", justify="right", style="green")
    table.add_column("With aliases", justify="right", style="yellow")

    for etype, entries in registry.items():
        if etype == "synthesis_date" or not isinstance(entries, list):
            continue
        cross = sum(1 for e in entries if len(e.get("papers", [])) > 1)
        aliased = sum(1 for e in entries if e.get("aliases"))
        table.add_row(etype, str(len(entries)), str(cross), str(aliased))

    console.print(table)
    console.print(f"\n[bold green]Saved:[/bold green] {out_path}")


@app.command()
def entities(
    entity_type: str = typer.Argument(
        ...,
        help=(
            "Entity type: strains | promoters | vectors | media | "
            "fermentation_conditions | glycosylation_patterns | "
            "target_products | process_parameters | analytical_methods"
        ),
    ),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
):
    """List extracted entities of a given type."""
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    items = kb.get_entities(entity_type)
    if not items:
        console.print(f"[yellow]No {entity_type} found in knowledge base.[/yellow]")
        return

    table = Table(title=f"{entity_type} ({len(items)} total)", show_lines=True)
    # Dynamic columns from first item keys
    sample = {k: v for k, v in items[0].items() if k != "_source_doc"}
    for col in list(sample.keys())[:6]:  # cap columns for readability
        table.add_column(col, overflow="fold", max_width=40)
    table.add_column("source", style="dim")

    for item in items:
        row = []
        for col in list(sample.keys())[:6]:
            val = item.get(col, "")
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val[:3])
            elif isinstance(val, dict):
                val = str(val)[:60]
            row.append(str(val)[:80] if val else "—")
        row.append(item.get("_source_doc", ""))
        table.add_row(*row)

    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Semantic search query"),
    n: int = typer.Option(5, help="Number of results"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
):
    """Semantic search in the vector store (show raw retrieved chunks)."""
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    hits = kb.semantic_search(query, n=n)
    if not hits:
        console.print("[yellow]No results.[/yellow]")
        return
    for i, h in enumerate(hits, 1):
        console.print(
            Panel(
                h["content"],
                title=f"[{i}] {h['source_file']} | {h['section']} | relevance={h['relevance']}",
                style="cyan",
            )
        )


@app.command()
def chat(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    model: str = typer.Option("gemini-2.5-flash", help="Claude model for Q&A"),
    n_chunks: int = typer.Option(6, help="Number of context chunks to retrieve"),
):
    """Interactive Q&A session with the Pichia assistant."""
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    assistant = PichiaAssistant(kb=kb, model=model, n_chunks=n_chunks)

    console.print(Panel(
        "[bold]PichiaGPT[/bold] — Pichia pastoris experimental assistant\n"
        "Type your question. Commands: [yellow]/reset[/yellow] (clear history), "
        "[yellow]/quit[/yellow] (exit).",
        style="green",
    ))
    kb_summary = kb.summary()
    console.print(
        f"Knowledge base: {kb_summary['vector_chunks']} chunks | "
        f"{kb_summary['documents']} documents | "
        f"{kb_summary['strains']} strains | "
        f"{kb_summary['target_products']} products\n"
    )

    while True:
        try:
            question = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            break

        if question.strip() == "/quit":
            break
        if question.strip() == "/reset":
            assistant.reset_history()
            console.print("[yellow]Conversation history cleared.[/yellow]")
            continue
        if not question.strip():
            continue

        console.print("[bold green]Assistant[/bold green]:")
        assistant.ask(question, stream=True)
        console.print()

    console.print("[dim]Session ended.[/dim]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    model: str = typer.Option("gemini-2.5-flash", help="Gemini model"),
    n_chunks: int = typer.Option(6, help="Context chunks"),
):
    """Ask a single question (non-interactive)."""
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    assistant = PichiaAssistant(kb=kb, model=model, n_chunks=n_chunks)
    console.print("[bold green]Answer:[/bold green]")
    assistant.ask(question, stream=True)


@app.command()
def synthesize(
    pdf: Path = typer.Argument(..., help="PDF file or directory to synthesize process knowledge from"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    model: str = typer.Option("gemini-2.5-flash", help="Gemini model"),
):
    """Extract fermentation control principles and protocols from papers."""
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    synth = ProcessKnowledgeSynthesizer(model=model, cache_dir=data_dir / "cache")

    pdfs: list[Path] = sorted(pdf.glob("*.pdf")) if pdf.is_dir() else [pdf]
    if not pdfs:
        console.print("[yellow]No PDFs found.[/yellow]")
        raise typer.Exit(0)

    for p in pdfs:
        console.print(Panel(f"Synthesizing: [bold]{p.name}[/bold]", style="magenta"))
        data = synth.synthesize_from_pdf(p)
        kb.ingest_process_knowledge(p.name, data)
        n_principles = len(data.get("control_principles", []))
        n_stages = len(data.get("process_stages", []))
        n_ts = len(data.get("troubleshooting", []))
        n_qf = len(data.get("product_quality_factors", []))
        console.print(
            f"  [green]control_principles={n_principles}  process_stages={n_stages}  "
            f"troubleshooting={n_ts}  quality_factors={n_qf}[/green]"
        )

    console.print("\n[bold green]Synthesis complete.[/bold green]")
    pk = kb.structured_store.process_knowledge_summary()
    table = Table(title="Process Knowledge Summary")
    table.add_column("Category", style="magenta")
    table.add_column("Count", justify="right")
    for k, v in pk.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)


@app.command()
def review(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    model: str = typer.Option("gemini-2.5-pro", help="Model for dialectical review (use pro for best quality)"),
):
    """Cross-paper dialectical review: find consensus, contradictions, and uncertainties."""
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)

    pk_raw = kb.structured_store._load_pk_raw()
    if not pk_raw:
        console.print("[red]No process knowledge found. Run 'synthesize' first.[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"Performing dialectical review across [bold]{len(pk_raw)}[/bold] papers "
        f"using [bold]{model}[/bold].\n"
        "This compares findings across papers to identify consensus and contradictions.",
        style="magenta",
        title="Dialectical Review",
    ))

    reviewer = DialecticalReviewer(model=model)
    dr = reviewer.review(pk_raw)
    kb.ingest_dialectical_review(dr)

    console.print(f"\n[bold green]Review complete.[/bold green] "
                  f"{len(dr.topic_syntheses)} topics analysed.\n")

    # Print overview
    console.print(Panel(dr.overall_summary, title="Overall Summary", style="green"))

    console.print("\n[bold]Highest-Confidence Findings (safe to rely on):[/bold]")
    for f in dr.highest_confidence_findings:
        console.print(f"  [green]✓[/green] {f}")

    console.print("\n[bold]Most Uncertain Areas (test carefully):[/bold]")
    for u in dr.most_uncertain_areas:
        console.print(f"  [yellow]⚠[/yellow]  {u}")

    console.print("\n[bold]Topics covered:[/bold]")
    for ts in dr.topic_syntheses:
        n_consensus = len(ts.consensus_points)
        n_conflict = len(ts.conflict_points)
        conf_color = {"high": "green", "medium": "yellow", "low": "red",
                      "conflicting": "red", "uncertain": "yellow"}.get(ts.overall_confidence, "white")
        console.print(
            f"  [{conf_color}]{ts.topic_area}[/{conf_color}] "
            f"[dim]confidence={ts.overall_confidence} | "
            f"{n_consensus} consensus, {n_conflict} conflicts[/dim]"
        )


@app.command()
def show_review(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    topic: str = typer.Option("", help="Filter by topic keyword"),
):
    """Display the stored dialectical review."""
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    dr = kb.get_dialectical_review()
    if not dr:
        console.print("[yellow]No dialectical review found. Run 'review' first.[/yellow]")
        return

    console.print(Panel(dr.overall_summary, title=f"Dialectical Review ({dr.review_date})", style="green"))

    topics = dr.topic_syntheses
    if topic:
        topics = [t for t in topics if topic.lower() in t.topic_area.lower()]

    def e(s: str) -> str:
        return rich_escape(str(s))

    for ts in topics:
        conf_val = str(ts.overall_confidence)
        conf_color = {"high": "green", "medium": "yellow", "low": "red",
                      "conflicting": "red", "uncertain": "yellow"}.get(conf_val, "white")

        header = f"[bold]{e(ts.topic_area)}[/bold]  [dim]confidence=[{conf_color}]{conf_val}[/{conf_color}][/dim]"
        lines = [header, "", e(ts.summary), ""]

        if ts.consensus_points:
            lines.append("[bold green]CONSENSUS:[/bold green]")
            for cp in ts.consensus_points:
                val = f" → [cyan]{e(cp.recommended_value)}[/cyan]" if cp.recommended_value else ""
                lines.append(f"  [green]✓[/green] [{e(str(cp.evidence_strength))}] {e(cp.consensus_claim)}{val}")
                lines.append(f"    {e(cp.practical_implication)}")
                for sp in cp.supporting_papers:
                    ctx = f" ({e(sp.experimental_context)})" if sp.experimental_context else ""
                    lines.append(f"    · {e(sp.paper)}{ctx}: {e(sp.claim)}")
                lines.append("")

        if ts.conflict_points:
            lines.append("[bold red]CONFLICTS:[/bold red]")
            for cfp in ts.conflict_points:
                lines.append(f"  [red]✗[/red] risk={e(cfp.risk_level)} | {e(cfp.topic)}")
                lines.append(f"    Why: {e(cfp.divergence_explanation)}")
                for pos in cfp.positions:
                    val_str = f" = {e(pos.value)}" if pos.value else ""
                    lines.append(f"    · {e(pos.paper)}{val_str}: {e(pos.claim)}")
                lines.append(f"    → {e(cfp.recommended_approach)}")
                if cfp.open_questions:
                    lines.append(f"    Open: {e('; '.join(cfp.open_questions))}")
                lines.append("")

        if ts.key_uncertainties:
            lines.append("[bold yellow]Uncertainties:[/bold yellow]")
            for u in ts.key_uncertainties:
                lines.append(f"  [yellow]?[/yellow] {e(u)}")

        lines.append(f"\n[bold]Recommendation:[/bold] {e(ts.actionable_recommendation)}")

        console.print(Panel("\n".join(lines), style="magenta"))


@app.command()
def principles(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    category: str = typer.Option(
        "control_principles",
        help="control_principles | process_stages | troubleshooting | product_quality_factors | fermentation_protocols"
    ),
):
    """List synthesized process control knowledge."""
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    items = kb.get_process_knowledge(category)
    if not items:
        console.print(f"[yellow]No {category} found. Run 'synthesize' first.[/yellow]")
        return

    for i, item in enumerate(items, 1):
        src = item.pop("_source_doc", "")
        lines = [f"[bold]{item.get('title', item.get('problem', item.get('stage', f'Item {i}')))}[/bold]"]
        for k, v in item.items():
            if not v:
                continue
            if isinstance(v, list):
                v = "\n  ".join(f"• {x}" for x in v)
                lines.append(f"[cyan]{k}[/cyan]:\n  {v}")
            else:
                lines.append(f"[cyan]{k}[/cyan]: {v}")
        if src:
            lines.append(f"[dim]source: {src}[/dim]")
        console.print(Panel("\n".join(lines), style="magenta"))


@app.command("extract-figures")
def extract_figures(
    pdf: Path = typer.Argument(..., help="PDF file or directory of PDFs"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    model: str = typer.Option("gemini-2.5-flash", help="Gemini model for vision extraction"),
):
    """Extract figures from PDFs: save images + structured data (data points, trends, conclusions)."""
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    figures_dir = data_dir / "figures"
    extractor = FigureExtractor(
        figures_dir=figures_dir, model=model, cache_dir=data_dir / "cache"
    )

    pdfs: list[Path] = sorted(pdf.glob("*.pdf")) if pdf.is_dir() else [pdf]
    if not pdfs:
        console.print("[yellow]No PDF files found.[/yellow]")
        raise typer.Exit(0)

    total_figs = 0
    for p in pdfs:
        console.print(Panel(f"Extracting figures: [bold]{p.name}[/bold]", style="blue"))
        figs = extractor.extract_from_pdf(p)
        saved = kb.structured_store.save_figure_data(figs)
        total_figs += saved

        by_type: dict[str, int] = {}
        for fd in figs:
            t = fd.figure_type
            by_type[t] = by_type.get(t, 0) + 1
        type_str = "  ".join(f"{t}={n}" for t, n in sorted(by_type.items()))
        console.print(f"  [green]Saved {saved} figures.[/green]  {type_str}")

    console.print(f"\n[bold green]Done.[/bold green] {total_figs} figures extracted total.")
    summary = kb.structured_store.figure_summary()
    table = Table(title="Figure Store Summary")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right")
    for t, n in sorted(summary.get("by_type", {}).items()):
        table.add_row(t, str(n))
    table.add_row("[bold]TOTAL[/bold]", str(summary.get("total_figures", 0)))
    console.print(table)


def _parse_varied_values(varied_str: str) -> tuple[str, list[str]]:
    """Parse 'axis_name: [v1, v2]' or 'axis: v1/v2/v3' → (axis, [v1, v2, ...])."""
    import ast as _ast

    if not varied_str or ":" not in varied_str:
        return varied_str.strip() if varied_str else "", []
    axis, rhs = varied_str.split(":", 1)
    axis = axis.strip()
    rhs = rhs.strip()
    try:
        v = _ast.literal_eval(rhs)
        if isinstance(v, (list, tuple)):
            return axis, [str(x) for x in v]
        if isinstance(v, str):
            return axis, [v]
    except Exception:
        pass
    rhs_clean = rhs.strip("[]()")
    for sep in ["/", ",", "、", ";", "；"]:
        if sep in rhs_clean:
            return axis, [x.strip().strip("'\"") for x in rhs_clean.split(sep) if x.strip()]
    return axis, [rhs_clean] if rhs_clean else []


@app.command("refine-figures")
def refine_figures_cmd(
    paper: str = typer.Option("", help="Filter by paper filename (partial match)"),
    exp: str = typer.Option("", help="Filter by experiment_id (partial match)"),
    fig: str = typer.Option("", help="Filter by figure_id (partial match)"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    model: str = typer.Option("gemini-2.5-flash", help="Gemini model. Flash+dynamic thinking is enough for label alignment; use Pro if values are off."),
):
    """Re-extract experiment-linked figures using the experiment's varied_parameters as a label prior.

    Useful when the original extraction merged adjacent x-axis labels (CJK + Greek mix) or
    miscounted bar groups. Combines visual bar-counting with text-derived expected categories.
    """
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    extractor = FigureExtractor(
        figures_dir=data_dir / "figures", model=model, cache_dir=data_dir / "cache"
    )

    all_papers = kb.structured_store.load_all_experiments()
    targets = [p for p in all_papers if not paper or paper in p.source_file]
    if not targets:
        console.print(f"[yellow]No paper matches '{paper}'.[/yellow]")
        raise typer.Exit(0)

    refined = 0
    skipped = 0
    for paper_exps in targets:
        figs_lookup = {
            f["figure_id"]: f
            for f in kb.structured_store.load_figures(source_file=paper_exps.source_file)
        }
        for e in paper_exps.experiments:
            if exp and exp not in e.experiment_id:
                continue
            if not e.linked_figure_ids:
                continue

            # Build hint from this experiment's varied_parameters (use the FIRST axis only —
            # most figures correspond to one primary varied axis).
            x_categories: list[str] = []
            for v in (e.goal.varied_parameters or []):
                _, vals = _parse_varied_values(v)
                if vals:
                    x_categories = vals
                    break
            y_metrics = list(e.goal.observation_targets or [])
            exp_summary = e.goal.summary or e.title or ""

            for fid in e.linked_figure_ids:
                if fig and fig not in fid:
                    continue
                fdata = figs_lookup.get(fid)
                if not fdata:
                    skipped += 1
                    continue
                stem = paper_exps.source_file.replace(".pdf", "")
                img_path = data_dir / "figures" / f"{stem}__{fid}.png"
                if not img_path.exists():
                    console.print(f"  [yellow]skip {fid}: image missing[/yellow]")
                    skipped += 1
                    continue

                paper_short = paper_exps.source_file.split("_")[-1].replace(".pdf", "")
                console.print(
                    f"  refining {paper_short} / {e.experiment_id} / {fid}  "
                    f"(N_expected={len(x_categories)})"
                )
                if x_categories:
                    console.print(f"    {rich_escape('expected x-cats: ' + ', '.join(x_categories))}")
                refined_data = extractor.refine_with_experiment_hints(
                    img_path=img_path,
                    source_file=paper_exps.source_file,
                    figure_id=fid,
                    page_number=fdata.get("page_number") or 0,
                    caption=fdata.get("caption"),
                    surrounding_text=fdata.get("surrounding_text") or "",
                    expected_x_categories=x_categories or None,
                    expected_y_metrics=y_metrics or None,
                    experiment_summary=exp_summary,
                )
                # Safety: do not overwrite existing non-empty data with an empty
                # refinement (LLM hiccup, JSON truncation, etc.)
                old_dp = len(fdata.get("data_points") or [])
                old_np = len(fdata.get("notable_points") or [])
                new_dp = len(refined_data.data_points)
                new_np = len(refined_data.notable_points)
                if new_dp == 0 and new_np == 0 and (old_dp > 0 or old_np > 0):
                    console.print(
                        f"    [yellow]refusing to overwrite ({old_dp} dp / {old_np} np) "
                        f"with empty refinement[/yellow]"
                    )
                    skipped += 1
                    continue
                kb.structured_store.save_figure_data([refined_data])
                console.print(
                    f"    [green]saved {new_dp} data_points / {new_np} notable[/green]"
                )
                refined += 1

    console.print(
        f"\n[bold green]Done.[/bold green] refined {refined} figures, skipped {skipped}."
    )


@app.command("extract-experiments")
def extract_experiments(
    pdf: Path = typer.Argument(..., help="PDF file or directory of PDFs"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    model: str = typer.Option("gemini-2.5-pro", help="Gemini model"),
):
    """Extract structured experiment runs (parameter snapshots + outcome + figure links) from papers."""
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    extractor = ExperimentExtractor(model=model, cache_dir=data_dir / "cache")

    pdfs: list[Path] = sorted(pdf.glob("*.pdf")) if pdf.is_dir() else [pdf]
    if not pdfs:
        console.print("[yellow]No PDF files found.[/yellow]")
        raise typer.Exit(0)

    total_exps = 0
    for p in pdfs:
        console.print(Panel(f"Extracting experiments: [bold]{p.name}[/bold]", style="blue"))
        figures = kb.structured_store.load_figures(source_file=p.name)
        paper_exps = extractor.extract_from_pdf(p, figures=figures)
        kb.structured_store.save_experiments(paper_exps)
        n = len(paper_exps.experiments)
        total_exps += n
        console.print(
            f"  [green]Saved {n} experiments[/green] ({len(figures)} figures available for linking)"
        )
        if paper_exps.extraction_notes:
            console.print(f"  [dim]notes: {paper_exps.extraction_notes[:200]}[/dim]")

    summary = kb.structured_store.experiment_summary()
    console.print(
        f"\n[bold green]Done.[/bold green] {total_exps} experiments extracted in this run. "
        f"Total in store: {summary['total_experiments']} across {summary['papers_with_experiments']} papers."
    )


@app.command("extract-lineage")
def extract_lineage(
    source: str = typer.Option("", help="Filter by paper filename (partial match). Empty = all."),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    model: str = typer.Option("gemini-2.5-pro", help="Gemini model"),
):
    """Extract intra-paper experiment lineage (parent → child edges) and persist on PaperExperiments."""
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    extractor = LineageExtractor(model=model)

    all_papers = kb.structured_store.load_all_experiments()
    if not all_papers:
        console.print("[yellow]No experiments found. Run extract-experiments first.[/yellow]")
        raise typer.Exit(0)

    targets = [p for p in all_papers if not source or source in p.source_file]
    if not targets:
        console.print(f"[yellow]No paper matches filter '{source}'.[/yellow]")
        raise typer.Exit(0)

    total_edges = 0
    for paper_exps in targets:
        console.print(Panel(f"Lineage: [bold]{paper_exps.source_file}[/bold]", style="blue"))
        if len(paper_exps.experiments) < 2:
            console.print("  [dim]<2 experiments, skipping[/dim]")
            continue
        edges = extractor.extract(paper_exps)
        paper_exps.lineage = edges
        kb.structured_store.save_experiments(paper_exps)
        total_edges += len(edges)
        console.print(f"  [green]Saved {len(edges)} edges[/green]")
        for e in edges:
            console.print(
                f"    {rich_escape('[' + e.from_id + ']')} → "
                f"{rich_escape('[' + e.to_id + ']')}  "
                f"([cyan]{e.relation}[/cyan])  {rich_escape(e.summary[:80])}"
            )

    console.print(f"\n[bold green]Done.[/bold green] {total_edges} lineage edges across {len(targets)} papers.")


@app.command()
def experiments(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    source: str = typer.Option("", help="Filter by paper filename (partial match)"),
    show_phases: bool = typer.Option(False, "--phases", help="Show phase parameter detail"),
):
    """Browse extracted experiment runs."""
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    all_papers = kb.structured_store.load_all_experiments()
    if source:
        all_papers = [p for p in all_papers if source.lower() in p.source_file.lower()]
    if not all_papers:
        console.print("[yellow]No experiments found.[/yellow]")
        return

    for paper in all_papers:
        console.print(Panel(f"[bold]{paper.source_file}[/bold] — {len(paper.experiments)} experiments", style="cyan"))
        for exp in paper.experiments:
            lines = [
                f"[bold cyan]{exp.experiment_id}[/bold cyan] {exp.title}",
                f"  Goal: {exp.goal.summary}",
            ]
            if exp.goal.varied_parameters:
                lines.append(f"  Varied: {', '.join(exp.goal.varied_parameters)}")
            if exp.goal.observation_targets:
                lines.append(f"  Observed: {', '.join(exp.goal.observation_targets)}")
            if exp.strain_construct.host_strain or exp.strain_construct.expression_vector:
                lines.append(
                    f"  Construct: {exp.strain_construct.host_strain or '?'} / {exp.strain_construct.expression_vector or '?'} / {','.join(exp.strain_construct.promoters) or '?'}"
                )
            if exp.setup.scale:
                lines.append(f"  Scale: {exp.setup.scale}  Medium: {exp.setup.initial_medium or '?'}")
            if exp.outcome.max_yield:
                lines.append(
                    f"  Outcome: {exp.outcome.max_yield}"
                    + (f" @ {exp.outcome.time_to_max_yield_hours}" if exp.outcome.time_to_max_yield_hours else "")
                    + (f", WCW {exp.outcome.max_wet_cell_weight}" if exp.outcome.max_wet_cell_weight else "")
                )
            if exp.linked_figure_ids:
                lines.append(f"  Figures: {', '.join(exp.linked_figure_ids)}")
            if show_phases and exp.phases:
                for ph in exp.phases:
                    bits = [ph.phase_name]
                    for k in ("duration_hours", "temperature_celsius", "ph",
                             "agitation_rpm", "aeration_vvm", "dissolved_oxygen_percent"):
                        v = getattr(ph, k, None)
                        if v:
                            bits.append(f"{k.split('_')[0]}={v}")
                    if ph.feeding_strategy:
                        bits.append(f"feed={ph.feeding_strategy[:60]}")
                    lines.append(f"    · {' | '.join(bits)}")
            console.print("\n".join(lines))
            console.print()


@app.command()
def figures(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    source: str = typer.Option("", help="Filter by paper filename (partial match)"),
    fig_type: str = typer.Option("", help="Filter by figure type (e.g. line_curve, bar_chart)"),
    show_data: bool = typer.Option(False, "--data", help="Show extracted data points"),
):
    """Browse extracted figure data and their quantitative results."""
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    all_figs = kb.structured_store.load_figures()

    if source:
        all_figs = [f for f in all_figs if source.lower() in f.get("source_file", "").lower()]
    if fig_type:
        all_figs = [f for f in all_figs if f.get("figure_type", "") == fig_type]

    if not all_figs:
        console.print("[yellow]No figures found.[/yellow]")
        return

    def e(s) -> str:
        return rich_escape(str(s)) if s else ""

    for fd in all_figs:
        lines = [
            f"[bold]{e(fd.get('figure_id'))}[/bold]  "
            f"[dim]{e(fd.get('figure_type'))}  p.{fd.get('page_number')}[/dim]",
            f"[cyan]caption:[/cyan] {e(fd.get('caption', '—'))}",
        ]

        ivars = fd.get("independent_variables", [])
        dvars = fd.get("dependent_variables", [])
        if ivars:
            iv_str = ", ".join(f"{v.get('name')}({v.get('unit','')})" for v in ivars)
            lines.append(f"[cyan]IV:[/cyan] {e(iv_str)}")
        if dvars:
            dv_str = ", ".join(f"{v.get('name')}({v.get('unit','')})" for v in dvars)
            lines.append(f"[cyan]DV:[/cyan] {e(dv_str)}")

        if fd.get("fixed_conditions"):
            fc = "  ".join(f"{k}={v}" for k, v in fd["fixed_conditions"].items())
            lines.append(f"[cyan]fixed:[/cyan] {e(fc)}")

        if fd.get("observed_trend"):
            lines.append(f"[yellow]trend:[/yellow] {e(fd['observed_trend'])}")

        notable = fd.get("notable_points", [])
        for np_ in notable:
            lines.append(
                f"  [green]★[/green] {e(np_.get('condition_description'))} → "
                f"[bold]{e(np_.get('value_description'))}[/bold]  "
                f"[dim]({e(np_.get('point_type'))})[/dim]"
            )

        if fd.get("industrial_note"):
            lines.append(f"[magenta]industrial:[/magenta] {e(fd['industrial_note'])}")

        if show_data:
            dps = fd.get("data_points", [])
            if dps:
                lines.append(f"[dim]data_points ({len(dps)}):[/dim]")
                for dp in dps[:8]:
                    cond = " ".join(f"{k}={v}" for k, v in dp.get("conditions", {}).items())
                    vals = " ".join(f"{k}={v}" for k, v in dp.get("values", {}).items())
                    lines.append(f"  [dim]{e(cond)} → {e(vals)}[/dim]")
                if len(dps) > 8:
                    lines.append(f"  [dim]... +{len(dps)-8} more[/dim]")

        lines.append(f"[dim]image: {e(fd.get('image_path'))}[/dim]")
        console.print(Panel("\n".join(lines), style="blue"))


@app.command("domain-knowledge")
def domain_knowledge_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    model: str = typer.Option("gemini-2.5-flash", help="Gemini model"),
):
    """Synthesize cross-paper domain knowledge: proteins, substrates, yields, challenges, innovations."""
    data_dir = _resolve_project(project)
    papers_dir = data_dir / "papers"
    kb = _get_kb(data_dir)
    synth = DomainKnowledgeSynthesizer(model=model, cache_dir=data_dir / "cache")

    console.print(Panel(
        f"Synthesizing domain knowledge from [bold]{papers_dir}[/bold]",
        style="magenta",
    ))
    data = synth.synthesize(papers_dir)
    if not data:
        console.print("[red]Synthesis returned empty result.[/red]")
        raise typer.Exit(1)

    kb.structured_store.save_domain_knowledge(data)

    def e(s) -> str:
        return rich_escape(str(s)) if s else ""

    console.print(f"\n[bold green]Domain knowledge saved.[/bold green]  "
                  f"({data.get('synthesis_date', '')})\n")

    # Target proteins
    proteins = data.get("target_proteins", [])
    if proteins:
        t = Table(title=f"Target Proteins ({len(proteins)})", show_lines=True)
        t.add_column("Name", style="cyan", max_width=30)
        t.add_column("Type")
        t.add_column("Collagen")
        t.add_column("MW")
        t.add_column("Hydroxylation")
        for p in proteins:
            t.add_row(
                e(p.get("name")), e(p.get("protein_type")),
                e(p.get("collagen_type")), e(p.get("molecular_weight_kda")),
                "✓" if p.get("hydroxylation_required") else ("—" if p.get("hydroxylation_required") is None else "✗"),
            )
        console.print(t)

    # Yield benchmarks
    yields = data.get("yield_benchmarks", [])
    if yields:
        t = Table(title=f"Yield Benchmarks ({len(yields)})", show_lines=True)
        t.add_column("Protein", max_width=25)
        t.add_column("System", max_width=25)
        t.add_column("Yield", style="green", justify="right")
        t.add_column("Key Strategies", max_width=40)
        for y in yields:
            strats = "; ".join(y.get("key_strategies", [])[:2])
            t.add_row(e(y.get("protein")), e(y.get("expression_system")),
                      e(y.get("yield_value")), e(strats))
        console.print(t)

    # Technical challenges
    challenges = data.get("technical_challenges", [])
    if challenges:
        console.print(f"\n[bold]Technical Challenges ({len(challenges)}):[/bold]")
        for ch in challenges:
            status_color = {"solved": "green", "partially_solved": "yellow",
                            "unsolved": "red"}.get(ch.get("status", ""), "white")
            console.print(
                f"  [{status_color}]●[/{status_color}] [bold]{e(ch.get('challenge'))}[/bold]  "
                f"[dim]status={e(ch.get('status'))}[/dim]"
            )
            if ch.get("benefit_if_solved"):
                console.print(f"    → benefit: {e(ch['benefit_if_solved'])}")

    # Open questions
    oqs = data.get("key_open_questions", [])
    if oqs:
        console.print(f"\n[bold]Key Open Questions:[/bold]")
        for q in oqs:
            console.print(f"  [yellow]?[/yellow] {e(q)}")


@app.command("show-domain")
def show_domain(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
):
    """Display stored domain knowledge."""
    data_dir = _resolve_project(project)
    kb = _get_kb(data_dir)
    data = kb.structured_store.load_domain_knowledge()
    if not data:
        console.print("[yellow]No domain knowledge found. Run 'domain-knowledge' first.[/yellow]")
        return

    def e(s) -> str:
        return rich_escape(str(s)) if s else ""

    console.print(Panel(
        f"Domain Knowledge  [dim]{data.get('synthesis_date', '')}[/dim]\n"
        f"Papers: {len(data.get('papers_analyzed', []))}",
        style="magenta",
        title="Pichia × Collagen Domain Knowledge",
    ))

    for protein in data.get("target_proteins", []):
        lines = [
            f"[bold cyan]{e(protein.get('name'))}[/bold cyan]  "
            f"[dim]{e(protein.get('protein_type'))} | Col-{e(protein.get('collagen_type'))} | "
            f"{e(protein.get('molecular_weight_kda'))}[/dim]",
            f"  origin: {e(protein.get('sequence_origin'))}",
        ]
        if protein.get("key_features"):
            lines.append("  features: " + "; ".join(protein["key_features"]))
        console.print(Panel("\n".join(lines), style="cyan"))

    for ch in data.get("technical_challenges", []):
        status_color = {"solved": "green", "partially_solved": "yellow",
                        "unsolved": "red"}.get(ch.get("status", ""), "white")
        lines = [
            f"[bold]{e(ch.get('challenge'))}[/bold]  "
            f"[{status_color}]{e(ch.get('status'))}[/{status_color}]",
            f"  cause: {e(ch.get('root_cause'))}",
            f"  solutions: {e('; '.join(ch.get('solutions', [])))}",
        ]
        if ch.get("benefit_if_solved"):
            lines.append(f"  [green]benefit: {e(ch['benefit_if_solved'])}[/green]")
        console.print(Panel("\n".join(lines), style="yellow"))

    if data.get("field_maturity"):
        console.print(Panel(e(data["field_maturity"]), title="Field Maturity", style="green"))
    if data.get("industrialization_readiness"):
        console.print(Panel(e(data["industrialization_readiness"]),
                            title="Industrialization Readiness", style="blue"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_summary(kb: KnowledgeBase) -> None:
    s = kb.summary()
    table = Table(title="Knowledge Base Status")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    for k, v in s.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)


@app.command()
def serve(
    port: int = typer.Option(8502, help="Port to run on"),
):
    """Launch the web interface (Streamlit)."""
    import subprocess
    web_dir = Path(__file__).parent.parent.parent / "web" / "Home.py"
    subprocess.run(
        ["streamlit", "run", str(web_dir), "--server.port", str(port)],
        check=True,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
