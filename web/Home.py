"""首页 — 知识库概览"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import streamlit as st
from pichia_kb.knowledge_base import KnowledgeBase

DATA_DIR = Path(__file__).parent.parent / "data"

st.set_page_config(
    page_title="毕赤酵母发酵知识库",
    page_icon="🧫",
    layout="wide",
)

@st.cache_resource
def get_kb():
    return KnowledgeBase(data_dir=DATA_DIR)

kb = get_kb()
summary = kb.summary()
dr = kb.get_dialectical_review()

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🧫 毕赤酵母发酵知识库")
st.markdown(
    "基于多篇领域论文构建的知识问答系统，"
    "聚焦**重组人源胶原蛋白**在毕赤酵母（*Pichia pastoris*）中的表达与发酵控制。"
)
st.divider()

# ── Metrics ───────────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📄 收录论文", summary.get("documents", 0))
col2.metric("🔍 检索文段", summary.get("vector_chunks", 0))
col3.metric("📌 控制原则", summary.get("control_principles", 0))
col4.metric("🎯 目标产物", summary.get("target_products", 0))
col5.metric("⚖️ 辩证主题", summary.get("dialectical_topics", 0))

st.divider()

# ── Two-column layout ─────────────────────────────────────────────────────────

left, right = st.columns(2)

with left:
    st.subheader("📚 收录论文")
    papers_dir = DATA_DIR / "papers"
    pdfs = sorted(papers_dir.glob("*.pdf"))
    for p in pdfs:
        st.markdown(f"- {p.stem}")

    st.subheader("⚡ 快速开始")
    st.markdown("""
**左侧导航选择功能：**

| 页面 | 用途 |
|------|------|
| 💬 问答 | 向知识库提问，获得有来源依据的回答 |
| 🔬 控制原则 | 浏览发酵控制规则，按阶段/优先级筛选 |
| ⚖️ 辩证综合 | 查看跨论文的共识与冲突分析 |
| 🔍 语义搜索 | 从论文原文中检索相关段落 |
""")

with right:
    st.subheader("🏆 高置信度发现")
    if dr and dr.highest_confidence_findings:
        for finding in dr.highest_confidence_findings:
            st.success(finding)
    else:
        st.info("运行 `pichia-kb review` 生成辩证综合后显示")

    st.subheader("⚠️ 需谨慎的不确定区域")
    if dr and dr.most_uncertain_areas:
        for area in dr.most_uncertain_areas:
            st.warning(area)
    else:
        st.info("暂无数据")
