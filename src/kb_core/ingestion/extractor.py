"""LLM-based structured knowledge extraction from paper chunks using Gemini.

The schema (entity types, fields, descriptions) is now data-driven: each
project's `schema/knowledge.json` defines what to extract. The extractor
builds the LLM prompt by walking the entity types, and validates each
returned entity through the dynamically-generated Pydantic class.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel

from ..config import DomainContext
from ..schema import KnowledgeChunk
from ..schema_engine import ExtractionResult, FieldSpec, SchemaFile
from .pdf_processor import PDFProcessor


_EXTRACTION_SYSTEM = textwrap.dedent("""
You are an expert in {expert_field}. Extract structured knowledge from research
paper text.

The papers may be written in Chinese, English, or both. Extract ALL relevant entities
regardless of language. Be precise and faithful to the source text.
Return ONLY valid JSON — no explanation, no markdown fences.
If a field value is not mentioned, omit it or use null.
""").strip()


def _format_field_line(f: FieldSpec) -> str:
    type_repr = f.type
    if f.type == "list":
        type_repr = f"list[{f.item_type or 'str'}]"
    elif f.type == "enum":
        type_repr = f"enum({' | '.join(f.enum_values or [])})"
    flag = "required" if f.required else "optional"
    desc = f" — {f.description}" if f.description else ""
    return f"    - {f.name} ({type_repr}, {flag}){desc}"


def _format_entity_block(et, hint: str) -> str:
    lines = [f"\n\"{et.resolved_extraction_key()}\" (type {et.name}): {et.description or ''}"]
    if hint:
        lines.append(f"  Hint: {hint.strip()}")
    if et.fields:
        lines.append("  Fields:")
        for f in et.fields:
            # Skip provenance fields — the extractor injects them after the LLM call.
            if f.name in {"sources", "chunk_ids", "raw_mention", "extraction_confidence"}:
                continue
            lines.append(_format_field_line(f))
    return "\n".join(lines)


def _build_prompt_template(schema_spec: SchemaFile, hints: dict[str, str]) -> str:
    """Build the per-chunk extraction prompt from the project's knowledge schema."""
    blocks = []
    for et in schema_spec.entity_types:
        # Hint key is the extraction_key (e.g. "strains") so projects can
        # configure hints per JSON output key.
        hint = hints.get(et.resolved_extraction_key(), "")
        blocks.append(_format_entity_block(et, hint))
    blocks_text = "\n".join(blocks)

    return textwrap.dedent("""
        Extract structured knowledge from the text below.
        Return a JSON object whose top-level keys are the entity types listed
        below. Each value is a list of objects matching that entity type's
        fields. Only include keys that have data; omit empty arrays.

        === ENTITY TYPES ===
        {blocks}

        === SOURCE INFO ===
        Section: {{section}}
        File: {{source_file}}

        === TEXT TO EXTRACT FROM ===
        {{text}}
        """).strip().format(blocks=blocks_text)


class KnowledgeExtractor:
    """Uses Gemini to extract structured entities from paper chunks."""

    def __init__(
        self,
        domain: DomainContext,
        knowledge_spec: SchemaFile,
        knowledge_models: dict[str, type[BaseModel]],
        model: str = "gemini-2.5-flash",
        cache_dir: Path | None = None,
        request_timeout_ms: int = 120_000,
        keywords: list[str] | None = None,
    ) -> None:
        # Without an explicit timeout the SDK can hang indefinitely on a
        # stalled connection — we lost a 70-min ingest run to this on
        # 2026-04-29. 120s is generous for a single chunk extraction.
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=request_timeout_ms),
        )
        self.model = model
        self.domain = domain
        self.knowledge_spec = knowledge_spec
        self.knowledge_models = knowledge_models
        self.pdf_processor = PDFProcessor(cache_dir=cache_dir, keywords=keywords or [])
        self._system = _EXTRACTION_SYSTEM.format(expert_field=domain.expert_field)
        self._prompt_template = _build_prompt_template(
            knowledge_spec, domain.entity_hints
        )
        # Map extraction_key → (entity_type_name, model_class)
        self._key_to_model: dict[str, tuple[str, type[BaseModel]]] = {}
        for et in knowledge_spec.entity_types:
            cls = knowledge_models.get(et.name)
            if cls is None:
                continue
            self._key_to_model[et.resolved_extraction_key()] = (et.name, cls)

    def extract_from_pdf(self, pdf_path: Path, source_ref: str | None = None) -> ExtractionResult:
        chunks = self.pdf_processor.process(pdf_path)
        return self.extract_from_chunks(chunks, str(pdf_path.name), source_ref)

    def extract_from_chunks(
        self,
        chunks: list[KnowledgeChunk],
        source_file: str,
        source_ref: str | None = None,
    ) -> ExtractionResult:
        merged = ExtractionResult(source_file=source_file, source_ref=source_ref)
        total = len(chunks)
        print(f"  {total} chunks — extracting...")

        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}/{total}", end="\r")
            partial = self._extract_chunk(chunk)
            if partial:
                self._merge(merged, partial, chunk.source_file, chunk.chunk_id)

        print(f"  Done — {source_file}" + " " * 30)
        return merged

    # ── private ──────────────────────────────────────────────────────────────

    def _extract_chunk(self, chunk: KnowledgeChunk) -> dict | None:
        prompt = self._prompt_template.format(
            section=chunk.section or "unknown",
            source_file=chunk.source_file,
            text=chunk.content,
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self._system,
                    max_output_tokens=8192,
                    temperature=0.1,
                ),
            )
        except Exception as e:
            print(
                f"  [warn] API call failed for chunk={chunk.chunk_id}: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return None
        raw = (response.text or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(
                f"  [warn] JSON parse failed for chunk={chunk.chunk_id} "
                f"section={chunk.section!r}: {e}",
                file=sys.stderr,
            )
            print(f"         raw[:300]={raw[:300]!r}", file=sys.stderr)
            return None

    def _merge(
        self,
        target: ExtractionResult,
        data: dict,
        source_file: str,
        chunk_id: str,
    ) -> None:
        ref = [source_file]
        chunk_ref = [chunk_id]
        for key, (type_name, cls) in self._key_to_model.items():
            items = data.get(key, [])
            if not items:
                continue
            existing_list = target.entities.setdefault(key, [])
            for item_data in items:
                # The LLM occasionally emits ``None`` for list-typed fields
                # and produces placeholder entries with ``name: null`` that
                # aren't really entities. Drop None values so pydantic
                # defaults kick in, then skip silently if the identifying
                # ``name`` is missing.
                item_data = {k: v for k, v in item_data.items() if v is not None}
                if "name" in cls.model_fields and not item_data.get("name"):
                    continue
                item_data.setdefault("sources", ref)
                item_data.setdefault("chunk_ids", chunk_ref)
                try:
                    validated = cls(**item_data)
                    existing_list.append(validated.model_dump(exclude_none=True))
                except Exception as e:
                    print(
                        f"  [warn] {type_name} validation failed: "
                        f"{type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    print(f"         data={item_data!r}", file=sys.stderr)
