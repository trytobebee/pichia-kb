"""Shared project-context helpers for Streamlit pages.

Usage in each page:

    from _project import use_project_sidebar, current_kb, current_project_dir

    st.set_page_config(...)
    use_project_sidebar()  # renders the sidebar selector once
    kb = current_kb()
    DATA_DIR = current_project_dir()

`use_project_sidebar()` is the only function that touches Streamlit
widgets. The accessors are pure session-state reads, so they're safe
to call multiple times per page.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from kb_core.knowledge_base import KnowledgeBase

_REPO_ROOT = Path(__file__).parent.parent
PROJECTS_ROOT = _REPO_ROOT / "data" / "projects"


def list_projects() -> list[str]:
    if not PROJECTS_ROOT.is_dir():
        return []
    return sorted(p.name for p in PROJECTS_ROOT.iterdir() if p.is_dir())


def use_project_sidebar() -> None:
    """Render the sidebar project selector + new-project affordance. Call ONCE per page."""
    projects = list_projects()

    with st.sidebar:
        if projects:
            if "project" not in st.session_state or st.session_state.project not in projects:
                st.session_state.project = projects[0]
            chosen = st.selectbox(
                "📁 Project",
                projects,
                index=projects.index(st.session_state.project),
            )
            if chosen != st.session_state.project:
                st.session_state.project = chosen
        else:
            st.warning("No projects yet — create one below.")

        with st.expander("➕ New project", expanded=not projects):
            _render_new_project_form()

    if not projects:
        st.info("Create a project in the sidebar to begin.")
        st.stop()


def _render_new_project_form() -> None:
    """Inline form: pick template + slug + name → fork into data/projects/<slug>/."""
    from kb_core.cli import _TEMPLATES_ROOT, list_templates

    templates = list_templates()
    if not templates:
        st.caption("(no templates installed)")
        return

    with st.form("new_project_form", clear_on_submit=True):
        slug = st.text_input(
            "Slug (lowercase, hyphens)",
            placeholder="e.g. ecoli-protease",
        )
        name = st.text_input(
            "Display name",
            placeholder="e.g. E. coli protease expression KB",
        )
        template = st.selectbox("Template", templates, index=0)
        submitted = st.form_submit_button("Create", width="stretch")

    if not submitted:
        return
    if not slug.strip():
        st.error("Slug is required")
        return
    target = PROJECTS_ROOT / slug.strip()
    if target.exists():
        st.error(f"Project '{slug}' already exists")
        return

    import shutil
    src = _TEMPLATES_ROOT / template
    target.mkdir(parents=True)
    shutil.copy(src / "config.yaml", target / "config.yaml")
    shutil.copytree(src / "schema", target / "schema")
    for sub in ("papers", "cache", "figures", "structured"):
        (target / sub).mkdir()

    cfg_path = target / "config.yaml"
    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg_text = cfg_text.replace("slug: REPLACE_ME", f"slug: {slug}")
    cfg_text = cfg_text.replace("name: REPLACE_ME", f"name: {name or slug}")
    cfg_path.write_text(cfg_text, encoding="utf-8")

    st.session_state.project = slug
    st.success(f"✓ Created project '{slug}' from template '{template}'. Reloading…")
    st.rerun()


def current_project() -> str:
    """Pure read — returns the active project slug."""
    if "project" not in st.session_state:
        projects = list_projects()
        if not projects:
            st.error(f"No projects found under {PROJECTS_ROOT}. Create one first.")
            st.stop()
        st.session_state.project = projects[0]
    return st.session_state.project


def current_project_dir() -> Path:
    return PROJECTS_ROOT / current_project()


@st.cache_resource
def _kb_for(slug: str) -> KnowledgeBase:
    return KnowledgeBase(data_dir=PROJECTS_ROOT / slug)


def current_kb() -> KnowledgeBase:
    return _kb_for(current_project())


def get_assistant():
    """Return a chat assistant scoped to the current project.

    The assistant carries its own conversation history; we re-create it
    whenever the active project changes.
    """
    from kb_core.config import load_project_config
    from kb_core.qa import Assistant

    slug = current_project()
    if (
        "assistant" not in st.session_state
        or st.session_state.get("assistant_project") != slug
    ):
        cfg = load_project_config(current_project_dir())
        st.session_state.assistant = Assistant(kb=current_kb(), domain=cfg.domain)
        st.session_state.assistant_project = slug
        st.session_state.messages = []
    return st.session_state.assistant


def get_curator_agent():
    """Return a CuratorAgent scoped to the current project.

    Holds its own conversation history (separate from the QA assistant)
    so the user can independently iterate on schema changes.
    """
    from kb_core.curator import CuratorAgent

    slug = current_project()
    if (
        "curator_agent" not in st.session_state
        or st.session_state.get("curator_agent_project") != slug
    ):
        st.session_state.curator_agent = CuratorAgent(current_project_dir())
        st.session_state.curator_agent_project = slug
        st.session_state.curator_messages = []
    return st.session_state.curator_agent
