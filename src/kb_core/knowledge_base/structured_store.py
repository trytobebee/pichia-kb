"""JSON-backed structured store for domain entities and process knowledge."""

from __future__ import annotations

import json
from pathlib import Path

from ..schema.dialectical import DialecticalReview
from ..schema_engine import ExtractionResult, PaperExperiments, ProjectSchemas


class StructuredStore:
    """Persists and queries structured ExtractionResult objects and process knowledge."""

    _PROCESS_KB_FILE = "process_knowledge.json"
    _DIALECTICAL_FILE = "dialectical_review.json"
    _FIGURES_DIR = "figures"
    _DOMAIN_KB_FILE = "domain_knowledge.json"

    def __init__(self, store_path: Path) -> None:
        self._path = store_path
        self._path.mkdir(parents=True, exist_ok=True)
        self._pk_file = self._path / self._PROCESS_KB_FILE

    # ── Entity extraction store ───────────────────────────────────────────────

    def save(self, result: ExtractionResult) -> None:
        safe_name = result.source_file.replace("/", "_").replace(" ", "_")
        out_file = self._path / f"{safe_name}.json"
        out_file.write_text(
            result.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )

    def load_all(self) -> list[ExtractionResult]:
        results = []
        # Per-paper extraction files end in .pdf.json; the process knowledge
        # / domain / dialectical files live under different names.
        excluded = {self._PROCESS_KB_FILE, self._DIALECTICAL_FILE, self._DOMAIN_KB_FILE}
        for f in self._path.glob("*.pdf.json"):
            if f.name in excluded:
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                results.append(ExtractionResult.from_dict(data))
            except Exception:
                pass
        return results

    def get_all_entities(self, entity_type: str) -> list[dict]:
        """Return entities of a given extraction_key across all papers,
        each annotated with `_source_doc`."""
        entities = []
        for r in self.load_all():
            for item in r.entities.get(entity_type, []):
                d = dict(item)
                d["_source_doc"] = r.source_file
                entities.append(d)
        return entities

    def summary(self) -> dict:
        """Counts per entity type discovered in the stored extractions,
        plus document count. Entity type list is whatever appears in the
        data (driven by the project's knowledge schema, not hardcoded)."""
        per_type: dict[str, int] = {}
        n_docs = 0
        for r in self.load_all():
            n_docs += 1
            for key, items in r.entities.items():
                per_type[key] = per_type.get(key, 0) + len(items)
        out: dict[str, int] = {"documents": n_docs}
        for key in sorted(per_type):
            out[key] = per_type[key]
        return out

    # ── Process knowledge store ───────────────────────────────────────────────

    def save_process_knowledge(self, source_file: str, data: dict) -> None:
        """Upsert synthesized process knowledge for one paper into the combined KB."""
        existing = self._load_pk_raw()
        entry = {"source_file": source_file, **data}
        for i, e in enumerate(existing):
            if e.get("source_file") == source_file:
                existing[i] = entry
                break
        else:
            existing.append(entry)
        self._pk_file.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_process_knowledge(self, category: str) -> list[dict]:
        """Return flat list of dicts for a process knowledge category across all papers.

        Categories: control_principles, process_stages, fermentation_protocols,
                    troubleshooting, product_quality_factors
        """
        all_items = []
        for entry in self._load_pk_raw():
            source = entry.get("source_file", "")
            for item in entry.get(category, []):
                item = dict(item)
                item["_source_doc"] = source
                all_items.append(item)
        return all_items

    def process_knowledge_summary(self) -> dict:
        pk = self._load_pk_raw()
        counts = {
            "control_principles": 0,
            "process_stages": 0,
            "fermentation_protocols": 0,
            "troubleshooting": 0,
            "product_quality_factors": 0,
        }
        for entry in pk:
            for k in counts:
                counts[k] += len(entry.get(k, []))
        return counts

    def _load_pk_raw(self) -> list[dict]:
        if not self._pk_file.exists():
            return []
        try:
            return json.loads(self._pk_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    # ── Figure store ─────────────────────────────────────────────────────────

    def save_figure_data(self, figures: list) -> int:
        """Save extracted figure data; upsert by (source_file, figure_id). Returns saved count."""
        from ..schema.figures import FigureData
        fig_dir = self._path / self._FIGURES_DIR
        fig_dir.mkdir(exist_ok=True)
        saved = 0
        for fd in figures:
            if isinstance(fd, FigureData):
                data = fd.model_dump(exclude_none=True)
            else:
                data = fd
            src = data.get("source_file", "unknown")
            fid = data.get("figure_id", "unknown")
            safe = src.replace("/", "_").replace(" ", "_")
            out = fig_dir / f"{safe}__{fid}.json"
            out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            saved += 1
        return saved

    def load_figures(self, source_file: str | None = None) -> list[dict]:
        """Load all figure data, optionally filtered by source_file."""
        fig_dir = self._path / self._FIGURES_DIR
        if not fig_dir.exists():
            return []
        results = []
        for f in sorted(fig_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if source_file is None or data.get("source_file") == source_file:
                    results.append(data)
            except Exception:
                pass
        return results

    def figure_summary(self) -> dict:
        figs = self.load_figures()
        by_type: dict[str, int] = {}
        for fd in figs:
            t = fd.get("figure_type", "other")
            by_type[t] = by_type.get(t, 0) + 1
        return {"total_figures": len(figs), "by_type": by_type}

    # ── Domain knowledge store ────────────────────────────────────────────────

    def save_domain_knowledge(self, data: dict) -> None:
        out = self._path / self._DOMAIN_KB_FILE
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_domain_knowledge(self) -> dict | None:
        out = self._path / self._DOMAIN_KB_FILE
        if not out.exists():
            return None
        try:
            return json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            return None

    # ── Dialectical review store ──────────────────────────────────────────────

    def save_dialectical_review(self, review: DialecticalReview) -> None:
        dr_file = self._path / self._DIALECTICAL_FILE
        dr_file.write_text(
            review.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )

    def load_dialectical_review(self) -> DialecticalReview | None:
        dr_file = self._path / self._DIALECTICAL_FILE
        if not dr_file.exists():
            return None
        try:
            return DialecticalReview(**json.loads(dr_file.read_text(encoding="utf-8")))
        except Exception:
            return None

    def has_dialectical_review(self) -> bool:
        return (self._path / self._DIALECTICAL_FILE).exists()

    # ── Experiment store ─────────────────────────────────────────────────────

    @staticmethod
    def _experiments_filename(source_file: str) -> str:
        safe = source_file.replace("/", "_").replace(" ", "_")
        if safe.endswith(".pdf"):
            safe = safe[:-4]
        return f"{safe}.experiments.json"

    def save_experiments(self, paper_exps: PaperExperiments) -> None:
        out = self._path / self._experiments_filename(paper_exps.source_file)
        paper_exps.to_disk(out)

    def load_experiments(
        self,
        source_file: str,
        schemas: ProjectSchemas | None = None,
    ) -> PaperExperiments | None:
        out = self._path / self._experiments_filename(source_file)
        if not out.exists():
            return None
        exp_cls, lin_cls = self._exp_classes(schemas)
        try:
            return PaperExperiments.from_disk(out, exp_cls, lin_cls)
        except Exception:
            return None

    def load_all_experiments(
        self,
        schemas: ProjectSchemas | None = None,
    ) -> list[PaperExperiments]:
        exp_cls, lin_cls = self._exp_classes(schemas)
        results = []
        for f in sorted(self._path.glob("*.experiments.json")):
            try:
                results.append(PaperExperiments.from_disk(f, exp_cls, lin_cls))
            except Exception:
                pass
        return results

    @staticmethod
    def _exp_classes(schemas: ProjectSchemas | None):
        if schemas is None:
            return None, None
        return (
            schemas.experiments_models.get("ExperimentRun"),
            schemas.experiments_models.get("ExperimentLineageEdge"),
        )

    def experiment_summary(self) -> dict:
        # Doesn't need typed access — counts only
        all_papers = self.load_all_experiments()
        return {
            "papers_with_experiments": len(all_papers),
            "total_experiments": sum(len(p.experiments) for p in all_papers),
        }
