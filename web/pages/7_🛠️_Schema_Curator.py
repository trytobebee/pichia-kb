"""Schema Curator — chat with an LLM agent that can edit your project schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import streamlit as st

st.set_page_config(page_title="Schema Curator · 知识库", page_icon="🛠️", layout="wide")

from _project import use_project_sidebar, get_curator_agent, current_project_dir

use_project_sidebar()
agent = get_curator_agent()
project_dir = current_project_dir()


# ── Header ───────────────────────────────────────────────────────────────────
st.title("🛠️ Schema Curator")
st.caption(
    "通过对话编辑项目 schema。Curator 会先用 inspection 工具查证(查字段填充率、"
    "查找包含某字段的论文、读 audit log),再决定是否动 schema。每次修改自动写入 "
    "`schema_audit.jsonl`。"
)

# ── Sidebar: schema overview + audit log ─────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("📁 当前 schema")
    schema_dir = project_dir / "schema"
    for system_file in ("knowledge.json", "experiments.json", "data.json"):
        path = schema_dir / system_file
        if not path.exists():
            st.caption(f"⚪ {system_file} (缺失)")
            continue
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
            n_types = len(spec.get("entity_types", []))
            type_names = [et.get("name", "?") for et in spec.get("entity_types", [])]
            with st.expander(f"📄 {system_file} — {n_types} types"):
                for n in type_names:
                    st.markdown(f"- `{n}`")
        except Exception as e:
            st.caption(f"❌ {system_file}: {e}")

    st.divider()
    st.subheader("📜 Recent audit (最新 5 条)")
    audit_path = project_dir / "schema_audit.jsonl"
    if audit_path.exists():
        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        for line in reversed(lines[-5:]):
            try:
                e = json.loads(line)
                target = e.get("target", {})
                summary = " / ".join(f"{k}={v}" for k, v in target.items())[:60]
                st.caption(
                    f"**{e.get('action')}** · `{e.get('system')}` · {summary} "
                    f"_({e.get('actor', '?')})_"
                )
            except Exception:
                continue
    else:
        st.caption("(no audit log yet)")

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.curator_messages = []
        # Drop the agent so the next call re-creates it (resets history)
        st.session_state.pop("curator_agent", None)
        st.session_state.pop("curator_agent_project", None)
        st.rerun()


# ── Chat history rendering ───────────────────────────────────────────────────
if "curator_messages" not in st.session_state:
    st.session_state.curator_messages = []

for msg in st.session_state.curator_messages:
    with st.chat_message(msg["role"], avatar="🛠️" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])

# ── Hint banner if first time ────────────────────────────────────────────────
if not st.session_state.curator_messages:
    st.info(
        "💡 试试这些问题:\n\n"
        "- *Strain 实体上 genotype 字段在所有论文里的填充率是多少?*\n"
        "- *查一下 ExpressionVector 的 tag 字段被哪些论文用过*\n"
        "- *读一下 schema audit log 的最近 10 条*\n\n"
        "也可以直接让它修改 schema(都会写入 audit):\n"
        "- *给 ExperimentRun 加一个 cost_estimate_usd 字段(float, optional)*\n"
        "- *把 Strain 的 genotype 字段改名为 marker_genes,理由是更清晰*"
    )

# ── Chat input ───────────────────────────────────────────────────────────────
prompt = st.chat_input("跟 curator 说话…")
if prompt:
    st.session_state.curator_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🛠️"):
        with st.spinner("Curator 思考中…(可能在调用工具)"):
            try:
                response = agent.chat(prompt)
            except Exception as e:
                response = f"⚠️ 出错: `{type(e).__name__}: {e}`"
        st.markdown(response)

    st.session_state.curator_messages.append({"role": "assistant", "content": response})
    st.rerun()
