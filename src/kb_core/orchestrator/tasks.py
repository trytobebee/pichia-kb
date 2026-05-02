"""tasks.jsonl — append-only log of every extraction task that ran on this project.

Each entry captures one stage run on one paper: when it started/finished,
status, and any cost / notes. The current state of any (paper, stage)
pair is the most recent entry that mentions it.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal


StageName = Literal["chunks", "entities", "figures", "experiments"]
TaskStatus = Literal["success", "fail"]


def _tasks_path(project_dir: Path) -> Path:
    return project_dir / "tasks.jsonl"


def append_task(
    project_dir: Path,
    *,
    paper_id: str,
    stage: StageName,
    status: TaskStatus,
    started_at: float,
    finished_at: float,
    notes: str = "",
    error: str = "",
    llm_calls: int | None = None,
    cost_usd: float | None = None,
) -> str:
    """Append one task entry. Returns the task_id."""
    tid = uuid.uuid4().hex[:12]
    entry: dict[str, Any] = {
        "task_id": tid,
        "paper_id": paper_id,
        "stage": stage,
        "status": status,
        "started_at_iso": datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
        "finished_at_iso": datetime.fromtimestamp(finished_at, tz=timezone.utc).isoformat(),
        "duration_seconds": round(finished_at - started_at, 2),
    }
    if notes:
        entry["notes"] = notes
    if error:
        entry["error"] = error
    if llm_calls is not None:
        entry["llm_calls"] = llm_calls
    if cost_usd is not None:
        entry["cost_usd"] = round(cost_usd, 4)

    path = _tasks_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return tid


def load_tasks(project_dir: Path) -> list[dict]:
    """Return all task entries in chronological order."""
    path = _tasks_path(project_dir)
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def latest_status(project_dir: Path) -> dict[tuple[str, str], dict]:
    """Map (paper_id, stage) → most recent task entry. None if never run."""
    out: dict[tuple[str, str], dict] = {}
    for t in load_tasks(project_dir):
        out[(t["paper_id"], t["stage"])] = t
    return out


@contextmanager
def task(
    project_dir: Path, paper_id: str, stage: StageName, *, notes: str = ""
) -> Iterator[dict]:
    """Context manager that times a task and writes one entry on exit.

    Usage:
        with task(proj, "wj.pdf", "entities") as ctx:
            ctx["llm_calls"] = 12
            ... do work ...
    """
    started = time.time()
    ctx: dict[str, Any] = {}
    try:
        yield ctx
        append_task(
            project_dir,
            paper_id=paper_id,
            stage=stage,
            status="success",
            started_at=started,
            finished_at=time.time(),
            notes=notes or ctx.get("notes", ""),
            llm_calls=ctx.get("llm_calls"),
            cost_usd=ctx.get("cost_usd"),
        )
    except Exception as e:
        append_task(
            project_dir,
            paper_id=paper_id,
            stage=stage,
            status="fail",
            started_at=started,
            finished_at=time.time(),
            notes=notes,
            error=f"{type(e).__name__}: {e}",
        )
        raise
