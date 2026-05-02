"""schema_audit.jsonl — append-only log of schema changes per project.

Every modification to a project's schema (curator agent or human) writes
one line here, capturing who/when/why so changes can be inspected,
reverted, or used to drive re-extraction.

Phase 3f ships only the writer; the curator agent (step 4) will be the
main caller.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_audit_entry(project_dir: Path, entry: dict[str, Any]) -> None:
    """Append one JSON line to <project_dir>/schema_audit.jsonl.

    The caller supplies the body; we add `timestamp_iso` if missing.
    """
    entry = dict(entry)
    entry.setdefault("timestamp_iso", datetime.now(timezone.utc).isoformat())
    path = project_dir / "schema_audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
