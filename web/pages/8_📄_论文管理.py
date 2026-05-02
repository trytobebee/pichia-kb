"""论文管理 — 上传 PDF + 触发抽取流程,无需 CLI。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import pandas as pd
import streamlit as st

st.set_page_config(page_title="论文管理 · 知识库", page_icon="📄", layout="wide")

from _project import use_project_sidebar, current_project_dir
from kb_core.orchestrator import (
    latest_status,
    run_figures,
    run_chunks_and_entities,
    run_experiments,
    status_matrix,
)

use_project_sidebar()
project_dir = current_project_dir()
papers_dir = project_dir / "papers"
papers_dir.mkdir(parents=True, exist_ok=True)

st.title("📄 论文管理")
st.caption(
    "上传 PDF + 跑抽取管道。每个 stage 独立运行,失败不影响其他 stage;"
    "所有任务写入 `tasks.jsonl`(可在表格里点格子查历史)。"
)


# ── Upload ───────────────────────────────────────────────────────────────────
st.subheader("⬆️ 上传 PDF")
uploads = st.file_uploader(
    "拖拽或选择 PDF 文件(支持多选)",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)
if uploads:
    new_count = 0
    for uf in uploads:
        target = papers_dir / uf.name
        if target.exists():
            st.warning(f"`{uf.name}` 已存在,跳过")
            continue
        target.write_bytes(uf.getbuffer())
        new_count += 1
    if new_count:
        st.success(f"已保存 {new_count} 个 PDF 到 `{papers_dir.relative_to(project_dir.parent.parent)}/`")
        st.rerun()


# ── Status matrix ────────────────────────────────────────────────────────────
st.subheader("📊 抽取状态")
papers = sorted(p.name for p in papers_dir.glob("*.pdf"))
if not papers:
    st.info("还没有论文。在上面上传 PDF 即可开始。")
    st.stop()

status = status_matrix(project_dir, papers)
recent = latest_status(project_dir)


def cell(filled: bool) -> str:
    return "✅" if filled else "❌"


def short(name: str) -> str:
    return name.removesuffix(".pdf").split("_")[-1] or name


rows = []
for p in papers:
    s = status[p]
    rows.append({
        "Paper": short(p),
        "Filename": p,
        "Chunks": cell(s["chunks"]),
        "Entities": cell(s["entities"]),
        "Figures": cell(s["figures"]),
        "Experiments": cell(s["experiments"]),
    })
df = pd.DataFrame(rows)
st.dataframe(df, hide_index=True, width="stretch")


# ── Per-paper run controls ───────────────────────────────────────────────────
st.subheader("▶️ 触发抽取")

with st.form("run_stage_form"):
    cols = st.columns([3, 2, 1])
    with cols[0]:
        chosen_paper = st.selectbox(
            "选择论文",
            papers,
            format_func=short,
        )
    with cols[1]:
        chosen_stage = st.selectbox(
            "选择 stage",
            ["entities (chunks + 实体抽取)", "figures (图表抽取)", "experiments (实验工艺单)"],
        )
    with cols[2]:
        st.write("")  # spacer
        st.write("")
        run_clicked = st.form_submit_button("🚀 Run", width="stretch")

if run_clicked:
    paper_path = papers_dir / chosen_paper
    stage_label = chosen_stage.split(" ")[0]
    fn_map = {
        "entities": run_chunks_and_entities,
        "figures": run_figures,
        "experiments": run_experiments,
    }
    fn = fn_map[stage_label]

    # Cost estimate ballpark for user awareness
    estimate = {
        "entities": "30-90 LLM calls (gemini-2.5-flash), ~30s-2min",
        "figures": "5-50 LLM calls (gemini-2.5-flash), ~1-5min",
        "experiments": "1 LLM call (gemini-2.5-pro), ~1-3min",
    }[stage_label]
    st.info(f"预计:{estimate}")

    with st.spinner(f"Running {stage_label} on {short(chosen_paper)}..."):
        try:
            n = fn(project_dir, paper_path)
            st.success(f"✅ 完成 — produced {n} {stage_label} item(s)")
        except Exception as e:
            st.error(f"❌ 失败: `{type(e).__name__}: {e}`")
    st.rerun()


# ── Recent task log ──────────────────────────────────────────────────────────
st.subheader("📜 最近任务历史")
recent_tasks = list(reversed(sorted(recent.values(), key=lambda t: t["finished_at_iso"])))[:15]
if not recent_tasks:
    st.caption("(还没有任务记录)")
else:
    log_rows = [
        {
            "Finished": t["finished_at_iso"][11:19],
            "Paper": short(t["paper_id"]),
            "Stage": t["stage"],
            "Status": "✅" if t["status"] == "success" else "❌",
            "Duration": f"{t.get('duration_seconds', 0):.1f}s",
            "Notes": t.get("notes", "") or t.get("error", "")[:60],
        }
        for t in recent_tasks
    ]
    st.dataframe(pd.DataFrame(log_rows), hide_index=True, width="stretch")
