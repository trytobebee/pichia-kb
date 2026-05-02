"""跨论文对比 — 在所有论文的 47 条工艺单上做条件筛选,并对选中的实验做并列 diff。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import pandas as pd
import streamlit as st

st.set_page_config(page_title="跨论文对比 · 知识库", page_icon="📊", layout="wide")

from _project import use_project_sidebar, current_kb, current_project_dir

use_project_sidebar()
DATA_DIR = current_project_dir()
kb = current_kb()
all_papers = kb.structured_store.load_all_experiments()

st.title("📊 跨论文对比")
st.caption("把 7 篇论文的 47 个发酵实验拉到同一张表上,按规模/宿主/启动子/产物筛选,并对感兴趣的实验做并列 diff。")

if not all_papers:
    st.warning("无实验数据。先运行 `pichia-kb extract-experiments`。")
    st.stop()


# ── Helpers ──────────────────────────────────────────────────────────────────
def short_paper(name: str) -> str:
    return name.split("_")[-1].replace(".pdf", "")


_YIELD_RE = re.compile(
    r"([\d.]+)\s*(?:±\s*[\d.]+)?\s*(g|mg|μg|ug)\s*[/·]\s*L",
    flags=re.IGNORECASE,
)


def parse_yield_g_per_L(s: str | None) -> float | None:
    """Parse yield string to g/L. Returns None if not parseable."""
    if not s:
        return None
    m = _YIELD_RE.search(str(s))
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    unit = m.group(2).lower()
    if unit == "g":
        return v
    if unit == "mg":
        return v / 1000
    if unit in ("μg", "ug"):
        return v / 1_000_000
    return None


def scale_bucket(s: str | None) -> str:
    if not s:
        return "未知"
    s_l = s.lower()
    if "shake" in s_l or "flask" in s_l or "摇瓶" in s:
        return "摇瓶"
    m = re.search(r"([\d.]+)\s*l", s_l)
    if m:
        try:
            v = float(m.group(1))
        except ValueError:
            v = None
        if v is not None:
            if v <= 2:
                return "≤2L 罐"
            if v <= 5:
                return "5L 罐"
            return ">5L 罐"
    if "bioreactor" in s_l or "罐" in s:
        return "罐(规模未指明)"
    if "lab" in s_l or "purification" in s_l or "commercial" in s_l:
        return "其他"
    return "其他"


def has_keyword(values: list[str], keywords: list[str]) -> bool:
    if not keywords:
        return True
    blob = " ".join(values).lower()
    return any(k.lower() in blob for k in keywords if k.strip())


_AXIS_BUCKETS = [
    (re.compile(r"signal[_\s-]*peptide|信号肽", re.I), "信号肽"),
    (re.compile(r"copy[_\s-]*number|copies|拷贝", re.I), "拷贝数"),
    (re.compile(r"chaperone|分子伴侣|hac1", re.I), "分子伴侣 / HAC1 共表达"),
    (re.compile(r"transcription[_\s-]*factor|\btf\b|转录因子", re.I), "转录因子共表达"),
    (re.compile(r"translation[_\s-]*factor|\btif\b|翻译", re.I), "翻译因子共表达"),
    (re.compile(r"induction[_\s-]*temperature|诱导温度|temperature", re.I), "诱导温度"),
    (re.compile(r"methanol|甲醇|sorbitol|山梨醇|feeding", re.I), "甲醇 / 补料策略"),
    (re.compile(r"do[_\s-]?stat|溶解氧|溶氧|dissolved[_\s-]*oxygen", re.I), "溶氧 (DO)"),
    (re.compile(r"medium|培养基|bsm|bmgy", re.I), "培养基"),
    (re.compile(r"promoter|启动子", re.I), "启动子"),
    (re.compile(r"scale|规模|reactor|罐|bioreactor|fermentor", re.I), "规模 / 放大"),
    (re.compile(r"strain|菌株", re.I), "菌株筛选"),
    (re.compile(r"variant|mutant|突变|cleavage|kex2", re.I), "蛋白变体 / 突变"),
    (re.compile(r"\bph\b", re.I), "pH"),
    (re.compile(r"protease|蛋白酶|prb1", re.I), "蛋白酶敲除"),
    (re.compile(r"induction[_\s-]*time|诱导时间|time", re.I), "诱导时间"),
    (re.compile(r"nitrogen|氮源", re.I), "氮源"),
    (re.compile(r"vector|载体|质粒", re.I), "载体"),
]


def axis_bucket(text: str) -> str:
    for pat, name in _AXIS_BUCKETS:
        if pat.search(text):
            return name
    return "其他"


def parse_axes(varied: list[str]) -> list[str]:
    """Parse the canonical axis bucket from each entry of varied_parameters."""
    out: list[str] = []
    for v in varied or []:
        head = re.split(r"[:=：]", str(v), 1)[0].strip()
        if not head:
            continue
        out.append(axis_bucket(head))
    # de-dupe preserve order
    seen = set()
    res = []
    for a in out:
        if a not in seen:
            seen.add(a)
            res.append(a)
    return res


_PRODUCT_BUCKETS = [
    (re.compile(r"col3|hlcoliii|type[\s-]*iii.*collagen|ⅲ.*胶原|三型胶原|iii型胶原|col3a1", re.I), "Ⅲ 型胶原"),
    (re.compile(r"col2|type[\s-]*ii.*collagen|ⅱ.*胶原|二型胶原|ii型胶原", re.I), "Ⅱ 型胶原"),
    (re.compile(r"col1|type[\s-]*i.*collagen|ⅰ.*胶原|一型胶原|\bi型胶原", re.I), "Ⅰ 型胶原"),
    (re.compile(r"p4h|prolyl.*hydroxylase|羟化酶", re.I), "P4H 羟化酶"),
    (re.compile(r"hac1", re.I), "HAC1"),
    (re.compile(r"\bhl1\b", re.I), "HL1 重组蛋白"),
    (re.compile(r"adh1", re.I), "ADH1 醇脱氢酶"),
    (re.compile(r"chaperone|分子伴侣|kar2|lhs1|bmh2|aft1|kex2|ssa4", re.I), "其他伴侣 / 调控因子"),
    (re.compile(r"transcription factor|prm1|mit1|mxr1", re.I), "转录因子(共表达组分)"),
    (re.compile(r"translation factor|rli1|eif|pab1", re.I), "翻译因子(共表达组分)"),
]


def product_bucket(products: list[str]) -> str:
    blob = " | ".join(products or [])
    for pat, name in _PRODUCT_BUCKETS:
        if pat.search(blob):
            return name
    return (blob.split(" | ")[0][:30] if blob else "未指明")


def flatten() -> pd.DataFrame:
    rows = []
    for paper in all_papers:
        for e in paper.experiments:
            sc = e.strain_construct
            o = e.outcome
            scale_raw = e.setup.scale or ""
            yield_g = parse_yield_g_per_L(o.max_yield)
            axes = parse_axes(e.goal.varied_parameters or [])
            prod = product_bucket(sc.target_products or [])
            rows.append({
                "paper": short_paper(paper.source_file),
                "source_file": paper.source_file,
                "exp_id": e.experiment_id,
                "title": e.title or "",
                "scale_raw": scale_raw,
                "scale": scale_bucket(scale_raw),
                "host": sc.host_strain or "",
                "parent_strain": sc.parent_strain or "",
                "promoters": ", ".join(sc.promoters or []),
                "signal_peptide": sc.signal_peptide or "",
                "copies": sc.copy_number or "",
                "products": " | ".join(sc.target_products or []),
                "product_bucket": prod,
                "axes": axes,
                "axes_str": "、".join(axes) if axes else "",
                "varied_raw": " ; ".join(str(v) for v in (e.goal.varied_parameters or [])),
                "phase_pattern": " → ".join(p.phase_name for p in e.phases) if e.phases else "",
                "max_yield": o.max_yield or "",
                "yield_g_per_L": yield_g,
                "max_wet_cell_weight": o.max_wet_cell_weight or "",
                "time_to_max_yield_hours": o.time_to_max_yield_hours or "",
                "phases_n": len(e.phases),
            })
    return pd.DataFrame(rows)


df_all = flatten()


# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.header("🔎 筛选")

# Presets
st.sidebar.subheader("快捷预设")
preset_state_keys = ["filt_papers", "filt_scales", "filt_hosts", "filt_promoters", "filt_kw", "filt_has_yield"]


def reset_filters() -> None:
    for k in preset_state_keys:
        if k in st.session_state:
            del st.session_state[k]


def apply_preset(name: str) -> None:
    reset_filters()
    if name == "high_density":
        st.session_state["filt_scales"] = [s for s in df_all["scale"].unique() if s in ("5L 罐", ">5L 罐", "≤2L 罐")]
        st.session_state["filt_has_yield"] = True
    elif name == "type3_collagen":
        st.session_state["filt_kw"] = "Ⅲ 胶原 III collagen COL3 hlCOLIII COL3A1"
    elif name == "best_per_paper":
        # Will be handled below by post-filter pick
        st.session_state["_top_per_paper"] = True


col_p1, col_p2, col_p3, col_p4 = st.sidebar.columns(4)
if col_p1.button("罐+产量", use_container_width=True):
    apply_preset("high_density"); st.rerun()
if col_p2.button("Ⅲ型胶原", use_container_width=True):
    apply_preset("type3_collagen"); st.rerun()
if col_p3.button("每篇最高产", use_container_width=True):
    apply_preset("best_per_paper"); st.rerun()
if col_p4.button("清空", use_container_width=True):
    if "_top_per_paper" in st.session_state:
        del st.session_state["_top_per_paper"]
    reset_filters(); st.rerun()

st.sidebar.divider()

papers_pick = st.sidebar.multiselect(
    "📄 论文",
    sorted(df_all["paper"].unique()),
    key="filt_papers",
)
scales_pick = st.sidebar.multiselect(
    "⚗️ 规模",
    sorted(df_all["scale"].unique()),
    key="filt_scales",
)
hosts_pick = st.sidebar.multiselect(
    "🧬 宿主",
    sorted([h for h in df_all["host"].unique() if h]),
    key="filt_hosts",
)
promoters_all = sorted({p.strip() for s in df_all["promoters"] for p in s.split(",") if p.strip()})
promoters_pick = st.sidebar.multiselect(
    "🎯 启动子",
    promoters_all,
    key="filt_promoters",
)
kw = st.sidebar.text_input(
    "🔍 关键词(在 title/products/parent_strain 里搜)",
    key="filt_kw",
    placeholder="例如:COL3, Ⅲ 胶原, P4H, BSM…",
)
has_yield = st.sidebar.checkbox(
    "仅显示有可解析产量的实验", key="filt_has_yield"
)


# ── Apply filters ────────────────────────────────────────────────────────────
df = df_all.copy()
if papers_pick:
    df = df[df["paper"].isin(papers_pick)]
if scales_pick:
    df = df[df["scale"].isin(scales_pick)]
if hosts_pick:
    df = df[df["host"].isin(hosts_pick)]
if promoters_pick:
    pat = "|".join(re.escape(p) for p in promoters_pick)
    df = df[df["promoters"].str.contains(pat, na=False, regex=True)]
if kw and kw.strip():
    keywords = [k.strip() for k in re.split(r"[ ,，;；/\|]+", kw) if k.strip()]
    pat = "|".join(re.escape(k) for k in keywords)
    if pat:
        df = df[
            df["title"].str.contains(pat, case=False, regex=True, na=False)
            | df["products"].str.contains(pat, case=False, regex=True, na=False)
            | df["parent_strain"].str.contains(pat, case=False, regex=True, na=False)
        ]
if has_yield:
    df = df[df["yield_g_per_L"].notna()]
if st.session_state.get("_top_per_paper"):
    # Keep only the top-yield exp per paper (where yield is parseable)
    keep = (
        df[df["yield_g_per_L"].notna()]
        .sort_values("yield_g_per_L", ascending=False)
        .drop_duplicates(subset=["paper"], keep="first")
    )
    df = keep

# ── Metrics + table ──────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("匹配实验", len(df))
m2.metric("覆盖论文", df["paper"].nunique() if len(df) else 0)
m3.metric(
    "最高产量(g/L)",
    f"{df['yield_g_per_L'].max():.2f}" if df["yield_g_per_L"].notna().any() else "—",
)
m4.metric(
    "中位产量(g/L)",
    f"{df['yield_g_per_L'].median():.2f}" if df["yield_g_per_L"].notna().any() else "—",
)

st.divider()


tab_cluster, tab_table, tab_diff = st.tabs(["🔗 可比组(自动聚类)", "📋 全表", "🔬 并列 diff"])


# ── Tab: clusters by (axis × product) ────────────────────────────────────────
with tab_cluster:
    st.caption(
        "按变量轴 × 产物自动聚类:同组内实验都在研究同一个工艺维度(例如『拷贝数』)"
        "且做的是同一类产物,因此对比有意义。仅展示 ≥2 个成员的组。"
    )

    # Build clusters: (axis, product_bucket) → list of rows
    clusters: dict[tuple[str, str], list[dict]] = {}
    for _, r in df.iterrows():
        if not r["axes"]:
            continue
        for ax in r["axes"]:
            key = (ax, r["product_bucket"])
            clusters.setdefault(key, []).append(r.to_dict())

    qualified = sorted(
        [(k, v) for k, v in clusters.items() if len(v) >= 2],
        key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1]),
    )

    if not qualified:
        st.info("当前筛选下没有 ≥2 成员的可比组。试试清空筛选或扩大范围。")
    else:
        st.markdown(f"**共 {len(qualified)} 组可比实验**(覆盖 {sum(len(v) for _,v in qualified)} 条工艺单)")
        for (axis, prod), members in qualified:
            papers_in = sorted({m["paper"] for m in members})
            with st.expander(
                f"**{axis}** × **{prod}**  ·  "
                f"{len(members)} 个实验 / {len(papers_in)} 篇论文 "
                f"({', '.join(papers_in)})",
                expanded=(len(qualified) <= 4),
            ):
                cluster_df = pd.DataFrame([
                    {
                        "paper": m["paper"],
                        "exp_id": m["exp_id"],
                        "title": m["title"],
                        "scale": m["scale"],
                        "host": m["host"],
                        "promoters": m["promoters"],
                        "varied_parameters": m["varied_raw"][:200],
                        "max_yield": m["max_yield"],
                        "yield_g_per_L": m["yield_g_per_L"],
                    }
                    for m in members
                ]).sort_values("yield_g_per_L", ascending=False, na_position="last")
                st.dataframe(
                    cluster_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "title": st.column_config.TextColumn(width="medium"),
                        "varied_parameters": st.column_config.TextColumn(width="large"),
                        "yield_g_per_L": st.column_config.NumberColumn("yield (g/L)", format="%.3f"),
                    },
                )
                # Cross-paper aggregated chart if multiple papers + parseable yields
                yields_present = cluster_df["yield_g_per_L"].notna().sum()
                if cluster_df["paper"].nunique() >= 2 and yields_present >= 2:
                    chart_df = cluster_df.dropna(subset=["yield_g_per_L"]).copy()
                    chart_df["label"] = chart_df["paper"] + " · " + chart_df["exp_id"]
                    st.bar_chart(
                        chart_df.set_index("label")["yield_g_per_L"],
                        use_container_width=True,
                    )


# ── Tab: full filtered table ─────────────────────────────────────────────────
with tab_table:
    display_cols = [
        "paper", "exp_id", "title", "scale", "host", "promoters", "signal_peptide",
        "copies", "axes_str", "max_yield", "yield_g_per_L", "max_wet_cell_weight",
        "time_to_max_yield_hours", "phases_n",
    ]
    df_show = df[display_cols].sort_values(
        "yield_g_per_L", ascending=False, na_position="last"
    ).reset_index(drop=True)

    st.dataframe(
        df_show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "title": st.column_config.TextColumn("title", width="large"),
            "axes_str": st.column_config.TextColumn("变量轴"),
            "yield_g_per_L": st.column_config.NumberColumn("yield (g/L,解析)", format="%.3f"),
            "phases_n": st.column_config.NumberColumn("阶段#", width="small"),
        },
    )


# ── Tab: side-by-side diff ───────────────────────────────────────────────────
with tab_diff:
    st.caption("在筛选结果里挑 2–4 个实验做并列字段对比。先看上方『共同前提』再判断对比是否成立。")

    label_to_key: dict[str, tuple[str, str]] = {}
    for _, r in df.iterrows():
        label = f"[{r['paper']}/{r['exp_id']}] {r['title'][:60]}"
        label_to_key[label] = (r["source_file"], r["exp_id"])

    picked = st.multiselect(
        "选实验",
        list(label_to_key.keys()),
        max_selections=4,
        placeholder="挑 2-4 条做对比",
    )

    if len(picked) >= 2:
        paper_by_file = {p.source_file: p for p in all_papers}
        picked_rows = [df[(df["source_file"] == label_to_key[lbl][0]) & (df["exp_id"] == label_to_key[lbl][1])].iloc[0]
                       for lbl in picked]

        # ─ Common-premise detection ─
        def _intersect(values: list) -> set:
            sets = [set(v) if isinstance(v, (list, set, tuple)) else {v} for v in values]
            return set.intersection(*sets) if sets else set()

        shared_products = _intersect([{r["product_bucket"]} for r in picked_rows])
        shared_scales = _intersect([{r["scale"]} for r in picked_rows])
        shared_axes = _intersect([set(r["axes"] or []) for r in picked_rows])
        shared_hosts = _intersect([{r["host"]} for r in picked_rows if r["host"]])
        shared_promoters = _intersect([
            set(p.strip() for p in r["promoters"].split(",") if p.strip())
            for r in picked_rows if r["promoters"]
        ])
        shared_phases = _intersect([{r["phase_pattern"]} for r in picked_rows if r["phase_pattern"]])

        premise_pairs: list[tuple[str, str]] = []
        if shared_products and "未指明" not in shared_products:
            premise_pairs.append(("产物", "、".join(sorted(shared_products))))
        if shared_scales and "未知" not in shared_scales:
            premise_pairs.append(("规模", "、".join(sorted(shared_scales))))
        if shared_axes:
            premise_pairs.append(("研究的变量轴", "、".join(sorted(shared_axes))))
        if shared_hosts:
            premise_pairs.append(("宿主菌株", "、".join(sorted(shared_hosts))))
        if shared_promoters:
            premise_pairs.append(("启动子", "、".join(sorted(shared_promoters))))
        if shared_phases and "" not in shared_phases:
            premise_pairs.append(("阶段流程", "、".join(sorted(shared_phases))))

        if premise_pairs:
            badge_html = "  ".join(
                f"<span style='background:#dcfce7;color:#166534;padding:2px 8px;border-radius:6px;margin-right:6px;font-size:0.9em;'>"
                f"<b>{k}</b>: {v}</span>"
                for k, v in premise_pairs
            )
            st.markdown(
                f"**✅ 共同前提**(对比依据)<br/>{badge_html}",
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                "⚠️ 这几个实验在产物/规模/变量轴/宿主上都没有共同前提。"
                "并列对比可以照常看,但结果差异可能更多反映的是『实验目标不同』而非『策略不同』。"
            )

        st.divider()

        cols = st.columns(len(picked))

        def kv(items: dict, skip: tuple = ()) -> None:
            rendered = False
            for k, v in items.items():
                if k in skip or v in (None, "", [], {}):
                    continue
                if isinstance(v, list):
                    v = "、".join(str(x) for x in v) if v else ""
                    if not v:
                        continue
                st.markdown(f"- **{k}**: {v}")
                rendered = True
            if not rendered:
                st.caption("—")

        for i, label in enumerate(picked):
            src, eid = label_to_key[label]
            paper_exps = paper_by_file.get(src)
            exp = next((e for e in (paper_exps.experiments if paper_exps else []) if e.experiment_id == eid), None)
            with cols[i]:
                st.markdown(f"##### {short_paper(src)}  ·  `{eid}`")
                if not exp:
                    st.warning("未找到")
                    continue
                st.markdown(f"**{exp.title}**")
                if exp.outcome.max_yield:
                    st.success(f"产量: {exp.outcome.max_yield}")
                with st.expander("🎯 目标", expanded=True):
                    g = exp.goal
                    if g.summary:
                        st.markdown(g.summary)
                    if g.varied_parameters:
                        st.markdown("**变化**")
                        for p in g.varied_parameters:
                            st.markdown(f"- {p}")
                    if g.fixed_parameters:
                        st.markdown("**固定**")
                        for p in g.fixed_parameters[:6]:
                            st.markdown(f"- {p}")
                with st.expander("🧬 构建", expanded=True):
                    kv(exp.strain_construct.model_dump(exclude_none=True))
                with st.expander("⚗️ 设置", expanded=True):
                    kv(exp.setup.model_dump(exclude_none=True))
                if exp.phases:
                    with st.expander(f"📊 阶段 ({len(exp.phases)})", expanded=True):
                        df_p = pd.DataFrame([p.model_dump(exclude_none=True) for p in exp.phases])
                        st.dataframe(df_p, use_container_width=True, hide_index=True)
                with st.expander("🎁 结果", expanded=True):
                    kv(exp.outcome.model_dump(exclude_none=True))
    elif picked:
        st.info("再选一个实验开始对比。")
