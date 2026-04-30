"""控制原则浏览器"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import streamlit as st
from pichia_kb.knowledge_base import KnowledgeBase

DATA_DIR = Path(__file__).parent.parent.parent / "data"

st.set_page_config(page_title="控制原则 · 毕赤酵母知识库", page_icon="🔬", layout="wide")

@st.cache_resource
def get_kb():
    return KnowledgeBase(data_dir=DATA_DIR)

kb = get_kb()

st.title("🔬 发酵控制原则")
st.caption("从各论文中提炼的发酵控制规则，涵盖参数设定依据、机理说明和具体建议。")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📌 控制原则", "🗂️ 工艺阶段", "🔧 故障排查", "🎯 产品质量因子"])

# ── 控制原则 ──────────────────────────────────────────────────────────────────

with tab1:
    principles = kb.get_process_knowledge("control_principles")

    col1, col2, col3 = st.columns(3)
    stages = sorted({p.get("stage") or "通用" for p in principles})
    priorities = sorted({p.get("priority", "") for p in principles if p.get("priority")})

    selected_stage = col1.selectbox("发酵阶段", ["全部"] + stages)
    selected_priority = col2.selectbox("优先级", ["全部"] + priorities)
    search_kw = col3.text_input("关键词搜索", placeholder="温度 / pH / 甲醇 / 羟基化...")

    filtered = principles
    if selected_stage != "全部":
        filtered = [p for p in filtered if (p.get("stage") or "通用") == selected_stage]
    if selected_priority != "全部":
        filtered = [p for p in filtered if p.get("priority") == selected_priority]
    if search_kw:
        kw = search_kw.lower()
        filtered = [p for p in filtered if kw in str(p).lower()]

    st.caption(f"显示 {len(filtered)} / {len(principles)} 条")

    for p in filtered:
        priority = p.get("priority", "")
        color = {"critical": "🔴", "important": "🟡", "advisory": "🟢"}.get(priority, "⚪")
        title = p.get("title", "—")
        stage = p.get("stage") or "通用"

        with st.expander(f"{color} **{title}** `{stage}`"):
            cols = st.columns(2)
            with cols[0]:
                if p.get("observation"):
                    st.markdown(f"**现象：** {p['observation']}")
                if p.get("mechanism"):
                    st.markdown(f"**机理：** {p['mechanism']}")
            with cols[1]:
                if p.get("recommendation"):
                    st.info(f"**建议：** {p['recommendation']}")
                if p.get("target_value"):
                    st.success(f"**目标值：** {p['target_value']}")
                if p.get("consequence_if_ignored"):
                    st.error(f"**忽视后果：** {p['consequence_if_ignored']}")
            if p.get("_source_doc"):
                st.caption(f"来源：{p['_source_doc']}")

# ── 工艺阶段 ──────────────────────────────────────────────────────────────────

with tab2:
    stages_data = kb.get_process_knowledge("process_stages")
    if not stages_data:
        st.info("暂无数据")
    else:
        stage_labels = {
            "seed_culture": "🌱 种子培养",
            "glycerol_batch": "🧪 甘油批培养",
            "glycerol_fed_batch": "🧪 甘油流加",
            "methanol_transition": "🔄 甲醇过渡",
            "methanol_induction": "⚗️ 甲醇诱导",
            "harvest": "🏁 收获",
            "downstream": "🔬 下游处理",
        }
        for s in stages_data:
            stage_key = s.get("stage", "")
            label = stage_labels.get(stage_key, stage_key)
            with st.expander(f"**{label}**", expanded=(stage_key == "methanol_induction")):
                c1, c2, c3 = st.columns(3)
                c1.metric("温度 (°C)", s.get("temperature_celsius") or "—")
                c2.metric("pH", s.get("ph") or "—")
                c3.metric("DO (%)", s.get("dissolved_oxygen_percent") or "—")

                c4, c5 = st.columns(2)
                c4.metric("转速 (rpm)", s.get("agitation_rpm") or "—")
                c5.metric("时长 (h)", s.get("typical_duration") or "—")

                if s.get("carbon_source"):
                    st.markdown(f"**碳源：** {s['carbon_source']}")
                if s.get("feed_rate"):
                    st.markdown(f"**流加速率：** {s['feed_rate']}")
                if s.get("transition_trigger"):
                    st.info(f"**切换条件：** {s['transition_trigger']}")
                if s.get("key_monitoring"):
                    st.markdown("**关键监控：** " + "、".join(s["key_monitoring"]))
                if s.get("common_problems"):
                    st.warning("**常见问题：** " + "；".join(s["common_problems"]))
                if s.get("_source_doc"):
                    st.caption(f"来源：{s['_source_doc']}")

# ── 故障排查 ──────────────────────────────────────────────────────────────────

with tab3:
    ts_data = kb.get_process_knowledge("troubleshooting")
    if not ts_data:
        st.info("暂无数据")
    else:
        for t in ts_data:
            problem = t.get("problem", "未知问题")
            with st.expander(f"🔧 {problem}"):
                if t.get("root_causes"):
                    st.markdown("**根本原因：**")
                    for rc in t["root_causes"]:
                        st.markdown(f"  - {rc}")
                if t.get("solutions"):
                    st.markdown("**解决方案：**")
                    for sol in t["solutions"]:
                        st.success(f"  ✓ {sol}")
                if t.get("diagnostic_steps"):
                    st.markdown("**诊断步骤：**")
                    for ds in t["diagnostic_steps"]:
                        st.markdown(f"  {ds}")
                if t.get("prevention"):
                    st.info(f"**预防：** {t['prevention']}")
                if t.get("_source_doc"):
                    st.caption(f"来源：{t['_source_doc']}")

# ── 产品质量因子 ──────────────────────────────────────────────────────────────

with tab4:
    qf_data = kb.get_process_knowledge("product_quality_factors")
    if not qf_data:
        st.info("暂无数据")
    else:
        for qf in qf_data:
            factor = qf.get("factor", "—")
            stage = qf.get("stage") or "通用"
            with st.expander(f"🎯 **{factor}** `{stage}`"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    if qf.get("effect_on_structure"):
                        st.markdown(f"**→ 结构影响**\n\n{qf['effect_on_structure']}")
                with c2:
                    if qf.get("effect_on_modification"):
                        st.markdown(f"**→ 修饰影响**\n\n{qf['effect_on_modification']}")
                with c3:
                    if qf.get("effect_on_activity"):
                        st.markdown(f"**→ 活性影响**\n\n{qf['effect_on_activity']}")
                if qf.get("optimal_range"):
                    st.success(f"**最优范围：** {qf['optimal_range']}")
                if qf.get("_source_doc"):
                    st.caption(f"来源：{qf['_source_doc']}")
