"""辩证综合 — 跨论文共识与冲突分析"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import streamlit as st

st.set_page_config(page_title="辩证综合 · 知识库", page_icon="⚖️", layout="wide")

from _project import use_project_sidebar, current_kb, current_project_dir

use_project_sidebar()
DATA_DIR = current_project_dir()

CONF_COLOR = {
    "high": ("🟢", "green"),
    "medium": ("🟡", "orange"),
    "low": ("🔴", "red"),
    "conflicting": ("🔴", "red"),
    "uncertain": ("🟡", "orange"),
}
CONF_LABEL = {
    "high": "高置信",
    "medium": "中等置信",
    "low": "低置信",
    "conflicting": "存在冲突",
    "uncertain": "不确定",
}

kb = current_kb()
dr = kb.get_dialectical_review()

st.title("⚖️ 跨论文辩证综合")
st.caption("对比各论文的相同议题，识别共识（提升可信度）与冲突（提示实验风险）。")

if not dr:
    st.warning("尚未生成辩证综合。运行 `uv run kb review` 后刷新页面。")
    st.stop()

# ── Overview ──────────────────────────────────────────────────────────────────

st.info(dr.overall_summary)

col_hc, col_ua = st.columns(2)
with col_hc:
    st.subheader("✅ 高置信度发现")
    st.caption("多篇论文独立印证，可直接用于实验设计")
    for f in dr.highest_confidence_findings:
        st.success(f)

with col_ua:
    st.subheader("⚠️ 不确定区域")
    st.caption("论文间存在矛盾或数据不足，建议在实验中设置对照")
    for u in dr.most_uncertain_areas:
        st.warning(u)

st.divider()

# ── Topic filter ──────────────────────────────────────────────────────────────

all_topics = [ts.topic_area for ts in dr.topic_syntheses]
selected = st.multiselect("筛选主题（留空显示全部）", all_topics)
topics_to_show = [ts for ts in dr.topic_syntheses if not selected or ts.topic_area in selected]

# ── Topic cards ───────────────────────────────────────────────────────────────

for ts in topics_to_show:
    conf_val = str(ts.overall_confidence)
    icon, _ = CONF_COLOR.get(conf_val, ("⚪", "grey"))
    label = CONF_LABEL.get(conf_val, conf_val)
    n_c = len(ts.consensus_points)
    n_f = len(ts.conflict_points)

    with st.expander(
        f"{icon} **{ts.topic_area}** &nbsp;&nbsp; `{label}` &nbsp; "
        f"{n_c} 条共识 · {n_f} 条冲突",
        expanded=(conf_val in ("high",)),
    ):
        st.markdown(ts.summary)

        if ts.actionable_recommendation:
            st.info(f"**综合建议：** {ts.actionable_recommendation}")

        # Consensus
        if ts.consensus_points:
            st.markdown("#### ✅ 共识")
            for cp in ts.consensus_points:
                cp_conf = str(cp.evidence_strength)
                cp_icon, _ = CONF_COLOR.get(cp_conf, ("⚪", "grey"))
                val_str = f"　**推荐值：{cp.recommended_value}**" if cp.recommended_value else ""
                st.success(
                    f"{cp_icon} {cp.consensus_claim}{val_str}\n\n"
                    f"{cp.practical_implication}"
                )
                with st.container():
                    for sp in cp.supporting_papers:
                        ctx = f" *（{sp.experimental_context}）*" if sp.experimental_context else ""
                        val = f"：{sp.value}" if sp.value else ""
                        st.caption(f"  · {sp.paper}{ctx}{val} — {sp.claim}")

        # Conflicts
        if ts.conflict_points:
            st.markdown("#### ❗ 冲突")
            for cfp in ts.conflict_points:
                risk_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(cfp.risk_level, "⚪")
                st.error(
                    f"{risk_icon} **{cfp.topic}**（风险：{cfp.risk_level}）\n\n"
                    f"**原因：** {cfp.divergence_explanation}"
                )
                for pos in cfp.positions:
                    val_str = f" = `{pos.value}`" if pos.value else ""
                    ctx = f" *（{pos.experimental_context}）*" if pos.experimental_context else ""
                    st.caption(f"  · {pos.paper}{val_str}{ctx} — {pos.claim}")
                st.warning(f"**建议：** {cfp.recommended_approach}")
                if cfp.open_questions:
                    st.caption("待解答：" + "；".join(cfp.open_questions))

        # Uncertainties
        if ts.key_uncertainties:
            st.markdown("#### ❓ 剩余不确定性")
            for u in ts.key_uncertainties:
                st.caption(f"  ? {u}")
