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

from pichia_kb.knowledge_base import KnowledgeBase

_REPO_ROOT = Path(__file__).parent.parent
PROJECTS_ROOT = _REPO_ROOT / "data" / "projects"


def list_projects() -> list[str]:
    if not PROJECTS_ROOT.is_dir():
        return []
    return sorted(p.name for p in PROJECTS_ROOT.iterdir() if p.is_dir())


def use_project_sidebar() -> None:
    """Render the sidebar project selector. Call ONCE per page."""
    projects = list_projects()
    if not projects:
        st.error(f"No projects found under {PROJECTS_ROOT}. Create one first.")
        st.stop()

    if "project" not in st.session_state:
        st.session_state.project = projects[0]

    with st.sidebar:
        chosen = st.selectbox(
            "📁 Project",
            projects,
            index=projects.index(st.session_state.project),
        )
        if chosen != st.session_state.project:
            st.session_state.project = chosen


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
    from pichia_kb.qa import PichiaAssistant

    slug = current_project()
    if (
        "assistant" not in st.session_state
        or st.session_state.get("assistant_project") != slug
    ):
        st.session_state.assistant = PichiaAssistant(kb=current_kb())
        st.session_state.assistant_project = slug
        st.session_state.messages = []
    return st.session_state.assistant
