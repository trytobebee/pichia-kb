"""实验抽取 — 展示从论文中抽取的发酵实验工艺单及其原文依据。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import pandas as pd
import streamlit as st

from _project import current_kb, current_project_dir

DATA_DIR = current_project_dir()
FIG_IMG_DIR = DATA_DIR / "figures"
PDF_DIR = DATA_DIR / "papers"
PAGE_CACHE_DIR = DATA_DIR / "cache" / "pdf_pages"


@st.cache_data(show_spinner=False)
def render_pdf_page(source_file: str, page_number: int, dpi: int = 150) -> str | None:
    """Render a PDF page to PNG (cached on disk). Returns the PNG path as a string."""
    import fitz  # pymupdf

    pdf_path = PDF_DIR / source_file
    if not pdf_path.exists() or not isinstance(page_number, int) or page_number < 1:
        return None
    out_dir = PAGE_CACHE_DIR / source_file.replace(".pdf", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"page_{page_number:03d}_dpi{dpi}.png"
    if out.exists():
        return str(out)
    try:
        doc = fitz.open(str(pdf_path))
        if page_number > len(doc):
            doc.close()
            return None
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = doc[page_number - 1].get_pixmap(matrix=mat)
        pix.save(str(out))
        doc.close()
        return str(out)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def locate_pages(source_file: str, queries: tuple[str, ...]) -> list[int]:
    """Find PDF body pages whose text contains any query string.

    A match in the first 25% of the document is treated as TOC and dropped
    if there are also matches later. Returns 1-indexed page numbers.
    """
    import fitz

    pdf_path = PDF_DIR / source_file
    if not pdf_path.exists() or not queries:
        return []
    cleaned = [q.strip() for q in queries if q and q.strip()]
    if not cleaned:
        return []
    try:
        doc = fitz.open(str(pdf_path))
        n = len(doc)
        hits: list[int] = []
        for i in range(n):
            text = (doc[i].get_text("text") or "").replace("　", " ")
            text_compact = "".join(text.split())
            for q in cleaned:
                q_compact = "".join(q.split())
                if q_compact and q_compact in text_compact:
                    hits.append(i + 1)
                    break
        doc.close()
        if not hits:
            return []
        body_threshold = max(int(n * 0.25), 5)
        body_hits = [h for h in hits if h >= body_threshold]
        return body_hits or hits
    except Exception:
        return []

st.set_page_config(page_title="实验抽取 · 知识库", page_icon="🧪", layout="wide")

from _project import use_project_sidebar
use_project_sidebar()

kb = current_kb()
all_papers = kb.structured_store.load_all_experiments()

st.title("🧪 实验抽取效果")
st.caption("逐篇展示论文中识别出的发酵实验工艺单(参数快照 + 阶段 + 结果),以及对应的原文图片与文字片段。")

if not all_papers:
    st.warning("未找到实验抽取数据。先运行 `pichia-kb extract-experiments data/papers/`。")
    st.stop()

# ── Top metrics ───────────────────────────────────────────────────────────────
total_exps = sum(len(p.experiments) for p in all_papers)
total_linked = sum(len(e.linked_figure_ids) for p in all_papers for e in p.experiments)
total_phases = sum(len(e.phases) for p in all_papers for e in p.experiments)

m1, m2, m3, m4 = st.columns(4)
m1.metric("📄 论文", len(all_papers))
m2.metric("🧪 实验工艺单", total_exps)
m3.metric("📊 阶段参数总数", total_phases)
m4.metric("🖼️ 链接的图表", total_linked)

st.divider()


def short_paper(name: str) -> str:
    return name.split("_")[-1].replace(".pdf", "")


# ── Paper selector + per-paper experiment cursor ──────────────────────────────
paper_options = {short_paper(p.source_file): p for p in all_papers}
paper_col, _ = st.columns([1, 2])
paper_pick = paper_col.selectbox(
    "📄 论文",
    list(paper_options.keys()),
    format_func=lambda k: f"{k}  ({len(paper_options[k].experiments)} 实验)",
)
paper = paper_options[paper_pick]

exp_state_key = f"exp_idx::{paper_pick}"
if exp_state_key not in st.session_state:
    st.session_state[exp_state_key] = 0
# Clamp in case experiment list shrank between reloads
st.session_state[exp_state_key] = min(
    st.session_state[exp_state_key], len(paper.experiments) - 1
)
exp_idx = st.session_state[exp_state_key]
exp = paper.experiments[exp_idx]


# ── Lineage graph (intra-paper experiment dependencies) ───────────────────────
RELATION_STYLE = {
    "applies_optimum":   ("#1f6feb", "继承最优"),
    "scales_up":         ("#9333ea", "放大"),
    "varies_parameter":  ("#16a34a", "变量"),
    "replaces_component":("#ea580c", "替换组分"),
    "branches":          ("#888888", "分支"),
    "derives_from":      ("#555555", "衍生"),
    "parallel":          ("#cccccc", "并列"),
}


def _short_label(e_obj) -> str:
    title = (e_obj.title or "").strip()
    if len(title) > 36:
        title = title[:34] + "…"
    yld = e_obj.outcome.max_yield if e_obj.outcome else None
    line2 = f"\\nyield={yld}" if yld else ""
    return f"{e_obj.experiment_id}\\n{title}{line2}"


def _wrap_cjk(text: str, n: int = 32) -> str:
    """Insert graphviz line breaks every n characters (CJK-safe, char-based)."""
    text = text.replace('"', "'")
    if not text:
        return ""
    return "\\n".join(text[i:i + n] for i in range(0, len(text), n))


def lineage_dot(paper, selected_id: str) -> str:
    lines = [
        'digraph G {',
        '  rankdir=TB;',
        '  bgcolor="transparent";',
        '  ranksep=0.3;',
        '  nodesep=0.25;',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, width=1.9, margin="0.08,0.04"];',
        '  edge [fontname="Helvetica", fontsize=8];',
    ]
    for e_obj in paper.experiments:
        is_sel = e_obj.experiment_id == selected_id
        fill = "#fde68a" if is_sel else "#f3f4f6"
        border = "#b45309" if is_sel else "#9ca3af"
        penw = 2 if is_sel else 1
        label = _short_label(e_obj).replace('"', "'")
        lines.append(
            f'  "{e_obj.experiment_id}" [label="{label}", '
            f'fillcolor="{fill}", color="{border}", penwidth={penw}];'
        )
    for ed in (getattr(paper, "lineage", None) or []):
        color, _ = RELATION_STYLE.get(ed.relation, ("#888888", ed.relation))
        lines.append(
            f'  "{ed.from_id}" -> "{ed.to_id}" '
            f'[label="{ed.relation}", color="{color}", fontcolor="{color}"];'
        )

    # Side callouts beside the selected node: 上一步 (incoming) / 下一步 (outgoing)
    lineage_edges = getattr(paper, "lineage", None) or []
    incoming_sel = [ed for ed in lineage_edges if ed.to_id == selected_id]
    outgoing_sel = [ed for ed in lineage_edges if ed.from_id == selected_id]

    def _callout_label(prefix: str, edges) -> str:
        body = "\\n".join(
            f"• {ed.from_id if prefix.startswith('⬅') else ed.to_id} ({ed.relation})\\n"
            f"  {_wrap_cjk(ed.summary, 32)}"
            for ed in edges
        )
        return f"{prefix}\\n{body}"

    # Place callouts at the rank of the parent (above selected) / child (below selected)
    # so the spatial position physically mirrors the "上一步" / "下一步" semantics.
    if incoming_sel:
        lbl = _callout_label("⬅ 上一步", incoming_sel)
        lines.append(
            f'  "_callout_in" [label="{lbl}", shape=note, '
            f'fillcolor="#fef3c7", color="#b45309", fontsize=8, width=4.0, margin="0.12,0.08"];'
        )
        parent_anchor = incoming_sel[0].from_id
        lines.append(f'  {{ rank=same; "{parent_anchor}"; "_callout_in"; }}')

    if outgoing_sel:
        lbl = _callout_label("➡ 下一步", outgoing_sel)
        lines.append(
            f'  "_callout_out" [label="{lbl}", shape=note, '
            f'fillcolor="#dcfce7", color="#166534", fontsize=8, width=4.0, margin="0.12,0.08"];'
        )
        child_anchor = outgoing_sel[0].to_id
        lines.append(f'  {{ rank=same; "{child_anchor}"; "_callout_out"; }}')

    lines.append('}')
    return "\n".join(lines)


if getattr(paper, "lineage", None):
    with st.expander(
        f"🧬 实验链路({len(paper.experiments)} 实验 / {len(getattr(paper, "lineage", None))} 关系)",
        expanded=True,
    ):
        graph_col, legend_col = st.columns([4, 1])
        with graph_col:
            st.graphviz_chart(lineage_dot(paper, exp.experiment_id), use_container_width=False)
        with legend_col:
            st.markdown("**关系图例**")
            for rel, (color, cn) in RELATION_STYLE.items():
                st.markdown(
                    f"<div style='white-space:nowrap;line-height:1.7;'>"
                    f"<span style='color:{color};font-weight:600'>● {rel}</span><br/>"
                    f"<span style='color:#666;font-size:0.82em;margin-left:1em'>{cn}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
else:
    st.caption("（此论文未抽取实验链路。运行 `pichia-kb extract-lineage --source <paper>` 生成。）")

st.divider()

# ── Experiment navigator (prev / current / next) ──────────────────────────────
def _exp_prev():
    st.session_state[exp_state_key] = max(0, st.session_state[exp_state_key] - 1)


def _exp_next():
    st.session_state[exp_state_key] = min(
        len(paper.experiments) - 1, st.session_state[exp_state_key] + 1
    )


nav_l, nav_c, nav_r = st.columns([1, 6, 1])
nav_l.button(
    "⬅ 上一个",
    use_container_width=True,
    disabled=(exp_idx == 0),
    on_click=_exp_prev,
    key=f"exp_prev::{paper_pick}",
)
nav_c.markdown(
    f"<div style='text-align:center;font-size:1.02em;padding:0.35em 0;'>"
    f"🧪 <b>[{exp.experiment_id}]</b> {exp.title or '(无标题)'} "
    f"<span style='color:#888;font-size:0.9em'>· {exp_idx + 1} / {len(paper.experiments)}</span>"
    f"</div>",
    unsafe_allow_html=True,
)
nav_r.button(
    "下一个 ➡",
    use_container_width=True,
    disabled=(exp_idx == len(paper.experiments) - 1),
    on_click=_exp_next,
    key=f"exp_next::{paper_pick}",
)

# ── Two-column layout ─────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")


def kv_block(items: dict, skip: tuple = ()) -> None:
    """Render a dict as a markdown bullet list, hiding empty values."""
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


# Left: structured experiment data ────────────────────────────────────────────
with left:
    st.subheader("📋 结构化实验数据")
    if exp.description:
        st.markdown(f"_{exp.description}_")
    if exp.paper_section:
        st.caption(f"📍 论文章节: {exp.paper_section}")

    # Lineage neighbors — show what this experiment derives from / leads to
    _lineage = getattr(paper, "lineage", None) or []
    incoming = [ed for ed in _lineage if ed.to_id == exp.experiment_id]
    outgoing = [ed for ed in _lineage if ed.from_id == exp.experiment_id]
    if incoming or outgoing:
        if incoming:
            for ed in incoming:
                st.markdown(
                    f"⬅ **上一步** `{ed.from_id}` · _{ed.relation}_ — {ed.summary}"
                )
        else:
            st.caption("⬅ 上一步:无(链路起点)")
        if outgoing:
            for ed in outgoing:
                st.markdown(
                    f"➡ **下一步** `{ed.to_id}` · _{ed.relation}_ — {ed.summary}"
                )
        else:
            st.caption("➡ 下一步:无(链路终点)")

    with st.expander("🎯 实验目标", expanded=True):
        g = exp.goal
        st.markdown(f"**摘要**: {g.summary or '—'}")
        if g.fixed_parameters:
            st.markdown("**固定参数**")
            for p in g.fixed_parameters:
                st.markdown(f"- {p}")
        if g.varied_parameters:
            st.markdown("**变化参数**")
            for p in g.varied_parameters:
                st.markdown(f"- {p}")
        if g.observation_targets:
            st.markdown("**观测指标**: " + "、".join(g.observation_targets))

    with st.expander("🧬 菌株/载体构建", expanded=True):
        kv_block(exp.strain_construct.model_dump(exclude_none=True))

    with st.expander("⚗️ 发酵设置", expanded=True):
        kv_block(exp.setup.model_dump(exclude_none=True))

    if exp.phases:
        with st.expander(f"📊 阶段参数 ({len(exp.phases)} 阶段)", expanded=True):
            df = pd.DataFrame([p.model_dump(exclude_none=True) for p in exp.phases])
            st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("🎁 实验结果", expanded=True):
        kv_block(exp.outcome.model_dump(exclude_none=True))

    if exp.purification_method or exp.analytical_methods:
        with st.expander("🔬 分析与纯化"):
            if exp.purification_method:
                st.markdown(f"- **纯化**: {exp.purification_method}")
            if exp.analytical_methods:
                st.markdown(f"- **分析方法**: " + "、".join(exp.analytical_methods))


def safe_paper_stem(source_file: str) -> str:
    """Match the convention used in figure_extractor (PDF stem) and structured_store (replaces / and space)."""
    return source_file.replace(".pdf", "")


# Right: source PDF pages + linked figures ───────────────────────────────────
with right:
    st.subheader("🖼️ 原文依据")

    # 1) Locate the page(s) where this experiment is described in the body text.
    queries: list[str] = []
    if exp.paper_section:
        queries.append(exp.paper_section)
        # also try without leading numbering, e.g. "拷贝数优化"
        cleaned = exp.paper_section
        while cleaned and (cleaned[0].isdigit() or cleaned[0] in ".  　·、"):
            cleaned = cleaned[1:]
        cleaned = cleaned.strip()
        if cleaned and cleaned != exp.paper_section:
            queries.append(cleaned)
    # also fall back to the experiment title (especially for English titles in CN papers)
    if exp.title:
        queries.append(exp.title)

    section_pages = locate_pages(paper.source_file, tuple(queries)) if queries else []
    # Avoid duplicates with the figure pages we'll show below
    fig_pages = set()
    if exp.linked_figure_ids:
        figs_lookup_for_pages = {
            f["figure_id"]: f
            for f in kb.structured_store.load_figures(source_file=paper.source_file)
        }
        for fid in exp.linked_figure_ids:
            p = (figs_lookup_for_pages.get(fid) or {}).get("page_number")
            if isinstance(p, int):
                fig_pages.add(p)
    section_pages_unique = [p for p in section_pages if p not in fig_pages]

    if section_pages_unique:
        # Keep at most 3 pages so we don't render an entire chapter
        shown = section_pages_unique[:3]
        with st.expander(
            f"📍 实验描述所在原文页(p{', p'.join(str(p) for p in shown)}"
            + (f" 等 {len(section_pages_unique)} 页" if len(section_pages_unique) > 3 else "")
            + ")",
            expanded=not bool(exp.linked_figure_ids),
        ):
            st.caption(f"按章节标题或实验 title 在 PDF 中匹配到 {len(section_pages_unique)} 页,显示前 {len(shown)} 页。")
            for p in shown:
                page_img = render_pdf_page(paper.source_file, p)
                if page_img:
                    st.markdown(f"**第 {p} 页**")
                    st.image(page_img, use_container_width=True)
                else:
                    st.warning(f"第 {p} 页渲染失败")

    if not exp.linked_figure_ids:
        if not section_pages_unique:
            st.info(
                "未找到匹配的原文页。可能 `paper_section` 未抽取或 PDF 文本中标题被换行/插图打断。"
            )
    else:
        figs_for_paper = {
            f["figure_id"]: f
            for f in kb.structured_store.load_figures(source_file=paper.source_file)
        }
        stem = safe_paper_stem(paper.source_file)

        for fid in exp.linked_figure_ids:
            fig_data = figs_for_paper.get(fid)
            page = (fig_data or {}).get("page_number", "?")
            ftype = (fig_data or {}).get("figure_type", "")
            st.markdown(f"#### 📌 {fid}  ·  p{page}  ·  _{ftype}_")

            img_path = FIG_IMG_DIR / f"{stem}__{fid}.png"
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.warning(f"图片文件缺失: `{img_path.name}`")

            if not fig_data:
                st.caption("(此图未抽取出结构化数据)")
                st.divider()
                continue

            if fig_data.get("caption"):
                st.markdown(f"**图注**: {fig_data['caption']}")

            notable = fig_data.get("notable_points") or []
            data_points = fig_data.get("data_points") or []

            if notable:
                with st.expander(f"⭐ 关键点 ({len(notable)} 个)", expanded=True):
                    for n in notable:
                        pt = n.get("point_type", "")
                        desc = n.get("condition_description", "")
                        val = n.get("value_description", "")
                        note = n.get("note") or ""
                        st.markdown(f"- **({pt})** {desc} → **{val}**" + (f" — _{note}_" if note else ""))

            if data_points:
                with st.expander(f"📋 数据点 ({len(data_points)} 条)"):
                    rows = []
                    for dp in data_points:
                        row = {**(dp.get("conditions") or {}), **(dp.get("values") or {})}
                        if dp.get("note"):
                            row["note"] = dp["note"]
                        rows.append(row)
                    df_dp = pd.DataFrame(rows)
                    st.dataframe(df_dp, use_container_width=True, hide_index=True)

            if fig_data.get("author_conclusion"):
                st.markdown(f"📝 **作者结论**: {fig_data['author_conclusion']}")
            if fig_data.get("industrial_note"):
                st.markdown(f"🏭 **工业建议**: {fig_data['industrial_note']}")

            page_img = render_pdf_page(paper.source_file, page) if isinstance(page, int) else None
            sur = fig_data.get("surrounding_text") or ""
            if page_img or sur:
                with st.expander(f"📄 原文(第 {page} 页)", expanded=False):
                    if page_img:
                        st.image(page_img, use_container_width=True)
                    elif sur:
                        st.caption("(PDF 页渲染失败,展示抽取的文字片段)")
                        st.text(sur[:1800] + ("..." if len(sur) > 1800 else ""))

            st.divider()
