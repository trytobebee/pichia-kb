"""CuratorAgent — Gemini-powered conversational schema editor.

The agent receives natural-language messages from the user, decides
whether to call any of its registered tools (mutations + inspection),
executes them locally, and returns a plain-text response.

Tools live in ``mutations.py`` and ``inspection.py`` as plain Python
functions; here we expose them to Gemini via FunctionDeclarations.

Project context (project_dir) is bound at construction — all tool calls
target that one project.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any, Callable

from google import genai
from google.genai import types

from . import inspection, mutations


_SYSTEM_PROMPT = textwrap.dedent("""
You are the Schema Curator — an LLM agent that helps a researcher maintain
a project's knowledge schema (kb-core framework). The schema lives in JSON
files at <project>/schema/{knowledge,experiments,data}.json. Each file
defines entity types and their fields.

Your job: when the user asks to add / remove / rename fields, or proposes a
new entity type, you:
  1. Use inspection tools first to ground your proposal — check whether
     the field is already there, how often it would be filled, what
     similar fields look like, what the audit log says.
  2. Make the schema change via mutation tools (add_field / remove_field /
     rename_field / add_entity_type). Always pass a `rationale` that
     captures *why* the change is being made.
  3. After making changes that may invalidate existing extractions
     (semantic field changes, new fields, removed fields), TELL the user
     which papers may need re-extraction and what it would cost. Do NOT
     trigger re-extraction yourself in this version — just inform.
  4. If the user wants to reject a proposal you made, use record_rejection
     so you don't suggest the same thing again.

When unsure, ASK the user before mutating. Pure rename (label-only) is
safe — no re-extraction needed; new field / removed field / type change
require re-extraction.

Always be concrete: name actual entity types and field names; quote
audit-log entries when relevant; show counts from completeness checks.
Answer in the same language as the user (Chinese or English).
""").strip()


class CuratorAgent:
    """Conversational schema curator scoped to one project."""

    def __init__(
        self,
        project_dir: Path,
        model: str = "gemini-2.5-flash",
    ) -> None:
        self.project_dir = project_dir
        self.model = model
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self._history: list[types.Content] = []

        # Build tool dispatch + Gemini FunctionDeclarations together.
        self._dispatch: dict[str, Callable[..., Any]] = {}
        self._tool_decls: list[types.FunctionDeclaration] = []
        self._register_tools()

    # ── public ───────────────────────────────────────────────────────────────

    def reset_history(self) -> None:
        self._history.clear()

    def chat(self, user_message: str, max_tool_rounds: int = 6) -> str:
        """Send one user message and return the final text response.

        The model may issue multiple rounds of tool calls before producing
        a final answer; we cap at `max_tool_rounds` to avoid runaway loops.
        """
        self._history.append(
            types.Content(role="user", parts=[types.Part(text=user_message)])
        )

        for _ in range(max_tool_rounds):
            response = self.client.models.generate_content(
                model=self.model,
                contents=self._history,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    tools=[types.Tool(function_declarations=self._tool_decls)],
                    temperature=0.1,
                ),
            )

            candidate = response.candidates[0] if response.candidates else None
            if candidate is None:
                return "(no response)"

            content = candidate.content
            self._history.append(content)

            tool_calls = [
                p.function_call for p in (content.parts or [])
                if getattr(p, "function_call", None) is not None
            ]
            if not tool_calls:
                # No more tool calls — the model produced its final text.
                text_parts = [
                    p.text for p in (content.parts or [])
                    if getattr(p, "text", None)
                ]
                return "".join(text_parts) or "(empty response)"

            # Execute each tool call, append a function response.
            for fc in tool_calls:
                result = self._execute(fc.name, dict(fc.args or {}))
                self._history.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response={"result": result},
                            )
                        )],
                    )
                )

        return "(reached max tool-call rounds without a final answer)"

    # ── tool execution ───────────────────────────────────────────────────────

    def _execute(self, name: str, args: dict) -> Any:
        """Run one tool. All errors are returned as JSON-friendly dicts so
        the LLM can react instead of crashing the chat loop."""
        fn = self._dispatch.get(name)
        if fn is None:
            return {"error": f"unknown tool: {name}"}
        try:
            return fn(**args)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # ── tool registration ────────────────────────────────────────────────────

    def _register_tools(self) -> None:
        proj = self.project_dir

        # ---- inspection ----
        def t_find_papers_with_field(system: str, entity_type: str, field_name: str) -> Any:
            return inspection.find_papers_with_field(proj, system, entity_type, field_name)

        def t_compute_field_completeness(system: str, entity_type: str, field_names: list[str]) -> Any:
            return inspection.compute_field_completeness(proj, system, entity_type, field_names)

        def t_query_schema_provenance(entity_type: str | None = None, field_name: str | None = None,
                                       system: str | None = None, actor: str | None = None) -> Any:
            return inspection.query_schema_provenance(
                proj, entity_type=entity_type, field_name=field_name,
                system=system, actor=actor,
            )

        # ---- mutations ----
        def t_add_field(system: str, entity_type: str, name: str, type: str,
                         required: bool = False, description: str = "",
                         item_type: str | None = None, enum_values: list[str] | None = None,
                         rationale: str | None = None) -> Any:
            field: dict[str, Any] = {"name": name, "type": type}
            if required:
                field["required"] = True
            if description:
                field["description"] = description
            if item_type:
                field["item_type"] = item_type
            if enum_values:
                field["enum_values"] = enum_values
            mutations.add_field(proj, system, entity_type, field, rationale=rationale)
            return {"ok": True, "added": name}

        def t_remove_field(system: str, entity_type: str, field_name: str,
                           rationale: str | None = None) -> Any:
            mutations.remove_field(proj, system, entity_type, field_name, rationale=rationale)
            return {"ok": True, "removed": field_name}

        def t_rename_field(system: str, entity_type: str, old_name: str, new_name: str,
                           rationale: str | None = None) -> Any:
            mutations.rename_field(proj, system, entity_type, old_name, new_name, rationale=rationale)
            return {"ok": True, "from": old_name, "to": new_name}

        def t_add_entity_type(system: str, name: str, description: str = "",
                               inherits: list[str] | None = None,
                               extraction_key: str | None = None,
                               fields: list[dict] | None = None,
                               rationale: str | None = None) -> Any:
            mutations.add_entity_type(
                proj, system, name,
                description=description, inherits=inherits or [],
                extraction_key=extraction_key, fields=fields or [],
                rationale=rationale,
            )
            return {"ok": True, "added_entity_type": name}

        def t_record_rejection(system: str, proposal_summary: str, rationale: str) -> Any:
            mutations.record_rejection(
                proj, system,
                proposal={"summary": proposal_summary},
                rationale=rationale,
            )
            return {"ok": True, "logged_rejection_of": proposal_summary}

        # Register everything
        self._dispatch = {
            "find_papers_with_field": t_find_papers_with_field,
            "compute_field_completeness": t_compute_field_completeness,
            "query_schema_provenance": t_query_schema_provenance,
            "add_field": t_add_field,
            "remove_field": t_remove_field,
            "rename_field": t_rename_field,
            "add_entity_type": t_add_entity_type,
            "record_rejection": t_record_rejection,
        }

        _SYSTEM_ENUM = {
            "type": "string",
            "enum": ["knowledge", "experiments", "data"],
            "description": "Which schema file the change targets.",
        }

        self._tool_decls = [
            types.FunctionDeclaration(
                name="find_papers_with_field",
                description=(
                    "List every recorded value of a field across the project's "
                    "papers. For experiments use dotted paths (e.g. "
                    "'outcome.max_yield')."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "system": _SYSTEM_ENUM,
                        "entity_type": {"type": "string"},
                        "field_name": {"type": "string"},
                    },
                    "required": ["system", "entity_type", "field_name"],
                },
            ),
            types.FunctionDeclaration(
                name="compute_field_completeness",
                description=(
                    "For each field, return total entities, how many have it "
                    "filled, and the fill-rate. Use to spot dead fields."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "system": _SYSTEM_ENUM,
                        "entity_type": {"type": "string"},
                        "field_names": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["system", "entity_type", "field_names"],
                },
            ),
            types.FunctionDeclaration(
                name="query_schema_provenance",
                description=(
                    "Read the schema audit log, optionally filtered. Use to "
                    "see what's been accepted/rejected before, and by whom."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "entity_type": {"type": "string"},
                        "field_name": {"type": "string"},
                        "system": _SYSTEM_ENUM,
                        "actor": {"type": "string"},
                    },
                },
            ),
            types.FunctionDeclaration(
                name="add_field",
                description=(
                    "Append a new field to an existing entity type. "
                    "ALWAYS supply a `rationale`."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "system": _SYSTEM_ENUM,
                        "entity_type": {"type": "string"},
                        "name": {"type": "string"},
                        "type": {
                            "type": "string",
                            "description": "FieldSpec.type: str|int|float|bool|list|dict|enum|object|ref:<system>.<EntityType>",
                        },
                        "required": {"type": "boolean"},
                        "description": {"type": "string"},
                        "item_type": {"type": "string", "description": "Required when type='list'"},
                        "enum_values": {"type": "array", "items": {"type": "string"}, "description": "Required when type='enum'"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["system", "entity_type", "name", "type", "rationale"],
                },
            ),
            types.FunctionDeclaration(
                name="remove_field",
                description="Remove a field from an entity type. ALWAYS supply a `rationale`.",
                parameters={
                    "type": "object",
                    "properties": {
                        "system": _SYSTEM_ENUM,
                        "entity_type": {"type": "string"},
                        "field_name": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["system", "entity_type", "field_name", "rationale"],
                },
            ),
            types.FunctionDeclaration(
                name="rename_field",
                description=(
                    "Rename a field on an entity type. Pure label change — no "
                    "re-extraction needed."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "system": _SYSTEM_ENUM,
                        "entity_type": {"type": "string"},
                        "old_name": {"type": "string"},
                        "new_name": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["system", "entity_type", "old_name", "new_name", "rationale"],
                },
            ),
            types.FunctionDeclaration(
                name="add_entity_type",
                description="Add a brand new entity type to a schema file.",
                parameters={
                    "type": "object",
                    "properties": {
                        "system": _SYSTEM_ENUM,
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "inherits": {"type": "array", "items": {"type": "string"}},
                        "extraction_key": {"type": "string"},
                        "fields": {"type": "array", "items": {"type": "object"}},
                        "rationale": {"type": "string"},
                    },
                    "required": ["system", "name", "rationale"],
                },
            ),
            types.FunctionDeclaration(
                name="record_rejection",
                description=(
                    "Log a rejected proposal to the audit log so you don't "
                    "re-propose the same idea later."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "system": _SYSTEM_ENUM,
                        "proposal_summary": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["system", "proposal_summary", "rationale"],
                },
            ),
        ]
