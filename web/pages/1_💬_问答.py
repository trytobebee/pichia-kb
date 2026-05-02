"""问答页面 — 多轮对话"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import streamlit as st

st.set_page_config(page_title="问答 · 知识库", page_icon="💬", layout="wide")

sys.path.insert(0, str(Path(__file__).parent.parent))
from _project import use_project_sidebar, get_assistant
from kb_core.qa import PichiaAssistant

use_project_sidebar()

# ── Session state (per-project assistant + messages) ──────────────────────────

assistant: PichiaAssistant = get_assistant()
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Header ────────────────────────────────────────────────────────────────────

st.title("💬 发酵实验问答")
st.caption("问答系统综合论文原文、控制原则和跨论文辩证综合，给出有来源依据的回答。")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("设置")
    model = st.selectbox("模型", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
    n_chunks = st.slider("检索段落数", 3, 12, 6)

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        # Drop assistant so get_assistant() rebuilds it next run
        st.session_state.pop("assistant", None)
        st.session_state.pop("assistant_project", None)
        st.rerun()

    st.divider()
    st.subheader("示例问题")
    examples = [
        "甲醇诱导阶段的最佳温度和pH是多少？",
        "如何提高III型胶原蛋白的羟基化效率？",
        "P4H共表达有哪些策略，各有什么优缺点？",
        "发酵过程中蛋白质降解严重该怎么办？",
        "甲醇流加控制有哪些方法？",
        "如何判断诱导阶段是否进入碳源饥饿？",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:10]}"):
            st.session_state._pending_question = ex
            st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🔬" if msg["role"] == "user" else "🧫"):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────

pending = st.session_state.pop("_pending_question", None)
prompt = st.chat_input("输入问题，例如：诱导温度对胶原蛋白产量有什么影响？") or pending

if prompt:
    # Update assistant settings from sidebar
    assistant.model = model
    assistant.n_chunks = n_chunks

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🔬"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧫"):
        response = st.write_stream(assistant.stream_chunks(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})
