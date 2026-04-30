"""语义搜索 — 检索论文原文段落"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import streamlit as st
from pichia_kb.knowledge_base import KnowledgeBase

DATA_DIR = Path(__file__).parent.parent.parent / "data"

st.set_page_config(page_title="语义搜索 · 毕赤酵母知识库", page_icon="🔍", layout="wide")

@st.cache_resource
def get_kb():
    return KnowledgeBase(data_dir=DATA_DIR)

kb = get_kb()

st.title("🔍 语义搜索")
st.caption("从论文原文中检索与问题语义最相关的段落，可用于查看原始依据。")

col1, col2 = st.columns([4, 1])
query = col1.text_input("搜索内容", placeholder="例如：甲醇流加控制策略、胶原蛋白羟基化、溶氧控制...")
n = col2.number_input("返回条数", min_value=1, max_value=15, value=5)

if query:
    with st.spinner("检索中..."):
        hits = kb.semantic_search(query, n=n)

    if not hits:
        st.warning("未找到相关段落。")
    else:
        st.caption(f"找到 {len(hits)} 条相关段落")
        for i, h in enumerate(hits, 1):
            relevance = h.get("relevance", 0)
            bar = "█" * int(relevance * 10) + "░" * (10 - int(relevance * 10))
            source = h.get("source_file", "")
            section = h.get("section") or "—"

            with st.expander(
                f"[{i}] {source}  |  {section}  |  相关度 {relevance:.2f}  {bar}",
                expanded=(i == 1),
            ):
                st.markdown(h["content"])
                keywords = [k for k in h.get("keywords", []) if k]
                if keywords:
                    st.caption("关键词：" + "、".join(keywords[:8]))
