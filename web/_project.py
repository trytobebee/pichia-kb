"""Shared project-context helpers for Streamlit pages.

Each page imports `current_project_dir()` and `current_kb()`. The sidebar
project selector is rendered automatically by `current_project()`.
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


def current_project() -> str:
    """Render sidebar selector and return the active project slug."""
    projects = list_projects()
    if not projects:
        st.error(f"No projects found under {PROJECTS_ROOT}. Create one first.")
        st.stop()

    if "project" not in st.session_state:
        st.session_state.project = projects[0]

    with st.sidebar:
        st.session_state.project = st.selectbox(
            "📁 Project",
            projects,
            index=projects.index(st.session_state.project),
        )
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
