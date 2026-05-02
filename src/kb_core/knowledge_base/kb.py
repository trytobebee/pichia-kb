"""High-level KnowledgeBase facade combining vector and structured stores."""

from __future__ import annotations

import re
from pathlib import Path

from .vector_store import VectorStore
from .structured_store import StructuredStore
from ..schema import KnowledgeChunk, ExtractionResult
from ..schema.dialectical import DialecticalReview


def _count_hits(searchable: str, q_words: set[str]) -> int:
    """Count how many distinct query words appear in a searchable string."""
    return sum(1 for w in q_words if w in searchable)


def _yield_to_numeric(yield_str: str | None) -> float:
    """Extract first numeric value from a yield string for sorting (e.g. '18.7 g/L' → 18.7)."""
    if not yield_str:
        return 0.0
    m = re.search(r"(\d+(?:\.\d+)?)", yield_str)
    return float(m.group(1)) if m else 0.0


class KnowledgeBase:
    """Unified interface for adding and retrieving Pichia pastoris knowledge."""

    def __init__(self, data_dir: Path) -> None:
        self.vector_store = VectorStore(db_path=data_dir / "db")
        self.structured_store = StructuredStore(store_path=data_dir / "structured")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_chunks(self, chunks: list[KnowledgeChunk]) -> int:
        return self.vector_store.add_chunks(chunks)

    def ingest_extraction(self, result: ExtractionResult) -> None:
        self.structured_store.save(result)

    def ingest_process_knowledge(self, source_file: str, data: dict) -> None:
        self.structured_store.save_process_knowledge(source_file, data)

    def ingest_dialectical_review(self, review: DialecticalReview) -> None:
        self.structured_store.save_dialectical_review(review)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def semantic_search(self, query: str, n: int = 6) -> list[dict]:
        return self.vector_store.query(query, n_results=n)

    def get_entities(self, entity_type: str) -> list[dict]:
        return self.structured_store.get_all_entities(entity_type)

    def get_process_knowledge(self, category: str) -> list[dict]:
        return self.structured_store.get_process_knowledge(category)

    def get_dialectical_review(self) -> DialecticalReview | None:
        return self.structured_store.load_dialectical_review()

    # ── Status ────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        s = self.structured_store.summary()
        s["vector_chunks"] = self.vector_store.count()
        pk = self.structured_store.process_knowledge_summary()
        s.update(pk)
        dr = self.structured_store.load_dialectical_review()
        s["dialectical_topics"] = len(dr.topic_syntheses) if dr else 0
        return s

    # ── RAG context ───────────────────────────────────────────────────────────

    def context_for_query(self, query: str, n_chunks: int = 6) -> str:
        """Build context string for RAG: paper chunks + control principles + dialectical review."""
        parts = []

        # 1. Semantic chunks from papers
        hits = self.semantic_search(query, n=n_chunks)
        if hits:
            chunk_parts = []
            for h in hits:
                src = h["source_file"]
                sec = h.get("section") or "unknown section"
                chunk_parts.append(f"[Source: {src} | {sec}]\n{h['content']}")
            parts.append("## Paper Excerpts\n\n" + "\n\n---\n\n".join(chunk_parts))

        # 2. Control principles (injected always — compact)
        principles = self.get_process_knowledge("control_principles")
        if principles:
            lines = []
            for p in principles[:15]:
                title = p.get("title", "")
                rec = p.get("recommendation", "")
                val = p.get("target_value", "")
                priority = p.get("priority", "")
                src = p.get("_source_doc", "")
                line = f"• [{priority.upper()}] {title}: {rec}"
                if val:
                    line += f" → {val}"
                if src:
                    line += f" [{src}]"
                lines.append(line)
            parts.append("## Per-paper Control Principles\n\n" + "\n".join(lines))

        # 3. Dialectical review: relevant topic syntheses + high-confidence findings
        dr = self.structured_store.load_dialectical_review()
        if dr:
            dr_parts = []

            # High-confidence findings (always inject)
            if dr.highest_confidence_findings:
                hcf = "\n".join(f"✓ {f}" for f in dr.highest_confidence_findings)
                dr_parts.append(f"### High-Confidence Cross-Paper Findings\n{hcf}")

            # Most uncertain areas (always inject as warnings)
            if dr.most_uncertain_areas:
                ua = "\n".join(f"⚠ {u}" for u in dr.most_uncertain_areas)
                dr_parts.append(f"### Uncertain Areas (handle with care)\n{ua}")

            # Topic syntheses matching query keywords
            q_lower = query.lower()
            for ts in dr.topic_syntheses:
                topic_lower = ts.topic_area.lower()
                # Simple relevance: topic words appear in query
                if any(word in q_lower for word in topic_lower.split()):
                    block = [f"### Topic: {ts.topic_area} [confidence={ts.overall_confidence}]"]
                    block.append(ts.summary)
                    for cp in ts.consensus_points:
                        val_str = f" ({cp.recommended_value})" if cp.recommended_value else ""
                        papers_str = ", ".join(
                            p.paper for p in cp.supporting_papers
                        )
                        block.append(
                            f"  CONSENSUS [{cp.evidence_strength}]{val_str}: {cp.consensus_claim} [{papers_str}]"
                        )
                    for cfp in ts.conflict_points:
                        block.append(
                            f"  CONFLICT [risk={cfp.risk_level}]: {cfp.topic} — {cfp.divergence_explanation}"
                        )
                        block.append(f"  → {cfp.recommended_approach}")
                    dr_parts.append("\n".join(block))

            if dr_parts:
                parts.append("## Dialectical Cross-Paper Review\n\n" + "\n\n".join(dr_parts))

        # 4. Quantitative figure data relevant to the query
        q_words = {w for w in query.lower().split() if len(w) > 1}
        # Add CJK 2-grams as fallback keywords (Chinese has no spaces)
        cjk_only = "".join(c for c in query if "一" <= c <= "鿿")
        for i in range(len(cjk_only) - 1):
            q_words.add(cjk_only[i:i + 2])
        _QUANT_TYPES = {"line_curve", "bar_chart", "scatter"}

        def _var_names(vars_list) -> list[str]:
            return [
                (v.get("name", "") if isinstance(v, dict) else str(v))
                for v in (vars_list or [])
            ]

        all_figs = self.structured_store.load_figures()
        scored_figs: list[tuple[int, dict]] = []
        for fig in all_figs:
            if fig.get("figure_type") not in _QUANT_TYPES:
                continue
            searchable = " ".join([
                fig.get("source_file", ""),
                fig.get("caption", ""),
                " ".join(_var_names(fig.get("independent_variables"))),
                " ".join(_var_names(fig.get("dependent_variables"))),
                fig.get("author_conclusion", ""),
                fig.get("observed_trend", ""),
            ]).lower()
            hits = _count_hits(searchable, q_words)
            if hits:
                scored_figs.append((hits, fig))
        if scored_figs:
            scored_figs.sort(key=lambda x: x[0], reverse=True)
            relevant_figs = [fig for _, fig in scored_figs]
            fig_lines = []
            for fig in relevant_figs[:8]:
                src = fig.get("source_file", "")
                fid = fig.get("figure_id", "")
                x = ", ".join(_var_names(fig.get("independent_variables")))
                y = ", ".join(_var_names(fig.get("dependent_variables")))
                notable_bits = []
                for p in (fig.get("notable_points") or [])[:3]:
                    cond = p.get("condition_description", "")
                    val = p.get("value_description", "")
                    note = p.get("note", "")
                    bit = f"{cond} → {val}" if cond or val else note
                    if note and (cond or val):
                        bit += f" ({note})"
                    if bit:
                        notable_bits.append(bit)
                notable = "; ".join(notable_bits)
                conclusion = fig.get("author_conclusion", "")[:150]
                line = f"[{src} | {fid}] {x} vs {y}"
                if notable:
                    line += f"\n  Notable: {notable}"
                if conclusion:
                    line += f"\n  Conclusion: {conclusion}"
                fig_lines.append(line)
            parts.append("## Relevant Figure Data\n\n" + "\n\n".join(fig_lines))

        # 5. Cross-paper entities from registry matching query
        import json as _json
        reg_file = self.structured_store._path / "entity_registry.json"
        if reg_file.exists():
            registry = _json.loads(reg_file.read_text(encoding="utf-8"))
            matched_entities: list[str] = []
            _FOCUS_TYPES = ("strains", "promoters", "target_products", "process_parameters")
            for etype in _FOCUS_TYPES:
                # Sort by papers desc so canonical (alias-rich) entries come first
                ents = sorted(
                    [e for e in registry.get(etype, []) if isinstance(e, dict)],
                    key=lambda e: len(e.get("papers", [])),
                    reverse=True,
                )
                # Track names already covered as aliases of earlier (bigger) entries
                covered: set[str] = set()
                for entity in ents:
                    papers = entity.get("papers", [])
                    if len(papers) < 2:
                        continue
                    name = entity.get("canonical_name", "")
                    if name.lower() in covered:
                        continue
                    aliases = entity.get("aliases", [])
                    searchable = " ".join([name] + aliases).lower()
                    if any(w in searchable for w in q_words):
                        alias_str = f" (aka {', '.join(aliases)})" if aliases else ""
                        matched_entities.append(
                            f"• [{etype}] {name}{alias_str} — appears in {len(papers)} papers"
                        )
                    covered.update(a.lower() for a in aliases)
            if matched_entities:
                parts.append(
                    "## Cross-Paper Entity Registry\n\n" + "\n".join(matched_entities[:20])
                )

        # 6. Experiment protocols matching the query (Layer 6 → QA injection)
        all_paper_exps = self.structured_store.load_all_experiments()
        if all_paper_exps:
            scored_exps: list[tuple[int, float, str, object]] = []  # (hits, yield_num, paper_short, ExperimentRun)
            for paper in all_paper_exps:
                paper_short = paper.source_file.split("_")[-1].replace(".pdf", "")
                for exp in paper.experiments:
                    g = exp.goal
                    sc = exp.strain_construct
                    setup = exp.setup
                    out = exp.outcome
                    searchable = " ".join(filter(None, [
                        paper.source_file,
                        paper_short,
                        exp.title or "",
                        exp.description or "",
                        exp.paper_section or "",
                        g.summary or "",
                        " ".join(g.fixed_parameters or []),
                        " ".join(g.varied_parameters or []),
                        " ".join(g.observation_targets or []),
                        sc.host_strain or "",
                        sc.expression_vector or "",
                        " ".join(sc.promoters or []),
                        sc.signal_peptide or "",
                        " ".join(sc.target_products or []),
                        " ".join(sc.product_variants or []),
                        sc.copy_number or "",
                        setup.scale or "",
                        setup.initial_medium or "",
                        " ".join((p.phase_name or "") for p in (exp.phases or [])),
                        " ".join((p.feeding_strategy or "") for p in (exp.phases or [])),
                        " ".join((p.notes or "") for p in (exp.phases or [])),
                        out.max_yield or "",
                        out.max_wet_cell_weight or "",
                        out.productivity or "",
                    ])).lower()
                    hits = _count_hits(searchable, q_words)
                    if hits:
                        scored_exps.append(
                            (hits, _yield_to_numeric(out.max_yield), paper_short, exp)
                        )

            if scored_exps:
                # Rank by keyword hits desc, then by yield desc as tiebreaker
                scored_exps.sort(key=lambda x: (x[0], x[1]), reverse=True)
                exp_blocks = []
                for _, _, paper_short, exp in scored_exps[:10]:
                    g = exp.goal
                    sc = exp.strain_construct
                    setup = exp.setup
                    out = exp.outcome

                    block = [f"[{paper_short} | {exp.experiment_id}] {exp.title}"]

                    if g.summary:
                        block.append(f"  Goal: {g.summary}")
                    if g.varied_parameters:
                        block.append(f"  Varied: {'; '.join(g.varied_parameters)}")

                    construct_bits = list(filter(None, [
                        sc.host_strain,
                        sc.expression_vector,
                        ("/".join(sc.promoters) if sc.promoters else None),
                        sc.signal_peptide,
                        ("/".join(sc.target_products) if sc.target_products else None),
                        (f"copy={sc.copy_number}" if sc.copy_number else None),
                    ]))
                    if construct_bits:
                        block.append(f"  Construct: {' + '.join(construct_bits)}")

                    setup_bits = list(filter(None, [setup.scale, setup.initial_medium]))
                    if setup_bits:
                        block.append(f"  Setup: {', '.join(setup_bits)}")

                    if exp.phases:
                        phase_strs = []
                        for p in exp.phases:
                            tag = p.phase_name
                            if p.feeding_strategy:
                                fs = p.feeding_strategy
                                tag += f"({fs[:60]})" if len(fs) > 60 else f"({fs})"
                            phase_strs.append(tag)
                        block.append(f"  Phases: {' → '.join(phase_strs)}")

                    outcome_bits = list(filter(None, [
                        (f"yield={out.max_yield}" if out.max_yield else None),
                        (f"WCW={out.max_wet_cell_weight}" if out.max_wet_cell_weight else None),
                        (f"OD600={out.max_od600}" if out.max_od600 else None),
                        (f"t_peak={out.time_to_max_yield_hours}h" if out.time_to_max_yield_hours else None),
                        (f"productivity={out.productivity}" if out.productivity else None),
                        (f"purity={out.purity}" if out.purity else None),
                    ]))
                    if outcome_bits:
                        block.append(f"  Outcome: {'; '.join(outcome_bits)}")

                    if exp.linked_figure_ids:
                        block.append(f"  Linked figs: {', '.join(exp.linked_figure_ids[:5])}")

                    exp_blocks.append("\n".join(block))

                parts.append("## Experiment Protocols\n\n" + "\n\n".join(exp_blocks))

        # 7. Domain knowledge overview (yield benchmarks, conditions, open questions)
        dk = self.structured_store.load_domain_knowledge()
        if dk:
            dk_parts = []

            benchmarks = dk.get("yield_benchmarks", [])
            if benchmarks:
                sorted_bm = sorted(
                    benchmarks,
                    key=lambda b: _yield_to_numeric(b.get("yield_value", "")),
                    reverse=True,
                )
                lines = [f"• {b.get('yield_value', '?')} — {b.get('protein', '')} ({b.get('expression_system', '')})" for b in sorted_bm]
                dk_parts.append("**Yield Benchmarks (sorted desc):**\n" + "\n".join(lines))

            conditions = dk.get("fermentation_conditions", [])
            if conditions:
                lines = [f"• {c.get('parameter', '')}: {c.get('typical_range', '')} (optimal: {c.get('optimal_value', '')})" for c in conditions]
                dk_parts.append("**Key Fermentation Conditions:**\n" + "\n".join(lines))

            questions = dk.get("key_open_questions", [])
            if questions:
                dk_parts.append("**Key Open Questions:**\n" + "\n".join(f"? {q}" for q in questions))

            if dk.get("field_maturity"):
                dk_parts.append(f"**Field Maturity:** {dk['field_maturity'][:200]}")

            if dk_parts:
                parts.append("## Domain Knowledge Overview\n\n" + "\n\n".join(dk_parts))

        return "\n\n" + "═" * 60 + "\n\n".join(parts) if parts else ""
