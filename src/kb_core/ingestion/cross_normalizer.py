"""Layer 3.5 cross-paper entity registry builder.

Collects all normalized entities from per-paper extraction JSONs, groups
synonyms (rule-based + LLM-assisted), and writes a single canonical
entity_registry.json that cross-links entities across papers.

The per-paper JSONs are treated as read-only; the registry is a derived view.
"""

from __future__ import annotations

import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from .normalizer import _norm_key, _merge_group, _union, _best_confidence, _merge_notes


# ── Entity type configuration ─────────────────────────────────────────────────

_ENTITY_KEY: dict[str, str] = {
    "strains":            "name",
    "promoters":          "name",
    "vectors":            "name",
    "media":              "name",
    "target_products":    "name",
    "analytical_methods": "name",
    "process_parameters": "parameter_name",
}

# Entity types where LLM synonym clustering is worth the API call
# (high name variation, mixed languages, abbreviations)
_LLM_CLUSTER_TYPES = {"target_products", "analytical_methods", "promoters"}

# Default human-readable descriptions per entity type. Project config can
# override via DomainContext.cross_entity_descriptions.
_DEFAULT_ENTITY_TYPE_DESC: dict[str, str] = {
    "strains":            "host strains",
    "promoters":          "transcriptional promoters",
    "vectors":            "expression vectors / plasmids",
    "media":              "culture media formulations",
    "target_products":    "target recombinant proteins or metabolites",
    "analytical_methods": "analytical / characterization methods",
    "process_parameters": "quantitative process parameters",
}


# ── Data collection ───────────────────────────────────────────────────────────

def _collect(structured_dir: Path) -> dict[str, list[dict]]:
    """Load all per-paper JSONs; return {entity_type: [entity_dict, ...]}
    where each entity_dict has an extra '_paper' key.

    Tolerates both legacy layout (entity lists at top level) and current
    layout (under ``entities``).
    """
    collected: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(structured_dir.glob("*.pdf.json")):
        paper = path.name
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        entity_root = data.get("entities") if isinstance(data.get("entities"), dict) else data
        for etype in _ENTITY_KEY:
            for entity in entity_root.get(etype, []):
                e = dict(entity)
                e["_paper"] = paper
                collected[etype].append(e)
    return dict(collected)


# ── Rule-based grouping (normalized name exact match) ────────────────────────

def _rule_groups(entities: list[dict], key_field: str) -> dict[str, list[dict]]:
    """Group entities by normalized key_field value.  Returns {norm_key: [entity, ...]}."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entities:
        raw = e.get(key_field, "")
        if raw:
            groups[_norm_key(str(raw))].append(e)
    return dict(groups)


# ── LLM synonym clustering ────────────────────────────────────────────────────

_SYN_SYSTEM_TEMPLATE = (
    "You are an expert in {expert_field}. "
    "Answer only with valid JSON — no explanation, no markdown fences."
)

_SYN_PROMPT = textwrap.dedent("""
You are given a list of {entity_desc} names extracted from research papers (mixed English/Chinese).
Group names that refer to EXACTLY the same real-world entity.

Grouping rules:
- Same entity = different spelling, language, abbreviation, or notation
  (e.g. "SDS-PAGE" = "SDS-PAGE电泳" = "SDS-PAGE analysis", "AOX1" = "P_AOX1" = "AOX1 promoter")
- Do NOT group related-but-distinct entities
  (e.g. a generic category and a specific subtype are different — keep separate)
- Singletons (no synonym) are valid groups
- Every input name must appear in exactly one group

Names:
{names}

Return a JSON object with a single key "clusters":
{{"clusters": [
  {{"canonical": "most standard/informative name", "members": ["name1", "name2"]}}
]}}
Include ALL names from the input. Return ONLY valid JSON.
""").strip()


def _llm_cluster(
    entity_type: str,
    norm_groups: dict[str, list[dict]],
    key_field: str,
    llm,  # LLMBackend (avoid circular import)
    expert_field: str,
    entity_desc: str,
) -> dict[str, list[str]]:
    """Ask the LLM to merge synonymous norm_keys among cross-paper candidates.

    Only names that appear in 2+ papers are sent to the LLM; single-paper
    entities become automatic singletons.  Returns {rep_norm_key: [norm_key, ...]}.
    """
    # Identify which norm_keys appear in 2+ distinct papers
    def papers_for(entities: list[dict]) -> set[str]:
        return {e.get("_paper", "?") for e in entities}

    cross_nks = {nk for nk, ents in norm_groups.items() if len(papers_for(ents)) > 1}

    # Start with singletons for everything
    result: dict[str, list[str]] = {nk: [nk] for nk in norm_groups}

    if not cross_nks:
        return result

    # Build display name map for cross-paper candidates only
    norm_to_display: dict[str, str] = {}
    for nk in cross_nks:
        raw = norm_groups[nk][0].get(key_field, nk)
        norm_to_display[nk] = str(raw)

    names_block = "\n".join(f"- {display}" for display in norm_to_display.values())
    prompt = _SYN_PROMPT.format(
        entity_desc=entity_desc,
        names=names_block,
    )

    try:
        data = llm.chat_json(
            prompt,
            system=_SYN_SYSTEM_TEMPLATE.format(expert_field=expert_field),
            temperature=0.1, max_tokens=8192,
        )
    except Exception as exc:
        print(f"  [warn] LLM call failed for {entity_type}: {exc}", file=sys.stderr)
        return result

    clusters = data.get("clusters", [])
    if not isinstance(clusters, list):
        print(f"  [warn] LLM cluster response missing 'clusters' array for {entity_type}", file=sys.stderr)
        return result

    # Map display names back to norm_keys (only cross-paper candidates)
    display_to_norm = {v: k for k, v in norm_to_display.items()}
    assigned: set[str] = set()

    for cluster in clusters:
        canonical_display = cluster.get("canonical", "")
        member_displays = cluster.get("members", [canonical_display])

        member_nks = []
        for disp in member_displays:
            nk = display_to_norm.get(disp) or display_to_norm.get(disp.lstrip("- "))
            if nk and nk not in assigned:
                member_nks.append(nk)
                assigned.add(nk)

        if not member_nks:
            continue

        rep_nk = display_to_norm.get(canonical_display) or member_nks[0]
        if rep_nk not in member_nks:
            member_nks.insert(0, rep_nk)
        result[rep_nk] = member_nks

    # Any cross_nk not handled by LLM → stays singleton (already in result)
    return result


# ── Cross-paper entity merge ──────────────────────────────────────────────────

def _merge_cross_paper(entities: list[dict], key_field: str) -> dict[str, Any]:
    """Merge entities (possibly from different papers) into one registry entry."""
    # Strip internal _paper field before delegating to normalizer merge
    clean = [{k: v for k, v in e.items() if k != "_paper"} for e in entities]
    merged = _merge_group(clean, key_field)

    # Annotate per-paper chunk_ids; register the paper even when chunk_ids is empty
    paper_chunks: dict[str, list[str]] = {}
    for e in entities:
        paper = e.get("_paper", "unknown")
        if paper not in paper_chunks:
            paper_chunks[paper] = []
        for cid in e.get("chunk_ids", []):
            if cid not in paper_chunks[paper]:
                paper_chunks[paper].append(cid)
    merged["chunk_ids_by_paper"] = paper_chunks
    merged["papers"] = sorted(paper_chunks.keys())

    return merged


# ── Registry builder ──────────────────────────────────────────────────────────

def build_registry(
    structured_dir: Path,
    expert_field: str,
    entity_descriptions: dict[str, str] | None = None,
    model: str = "gemini-2.5-flash",
) -> dict:
    """Build and return the cross-paper entity registry dict.

    `expert_field` is injected into the LLM clustering prompt; pass the
    project's `domain.expert_field`. `entity_descriptions` (defaults to
    framework-generic) override the human-readable descriptions used in
    the prompt.
    """
    from ..llm import get_llm
    llm = get_llm(model)
    descs = {**_DEFAULT_ENTITY_TYPE_DESC, **(entity_descriptions or {})}

    print("Collecting entities from per-paper JSONs...")
    all_entities = _collect(structured_dir)

    registry: dict[str, Any] = {}

    for etype, key_field in _ENTITY_KEY.items():
        entities = all_entities.get(etype, [])
        if not entities:
            registry[etype] = []
            continue

        print(f"  {etype}: {len(entities)} raw records", end="")

        # Step 1: rule-based grouping
        norm_groups = _rule_groups(entities, key_field)
        print(f" → {len(norm_groups)} unique names", end="")

        # Step 2: LLM synonym clustering (only for high-variation types)
        if etype in _LLM_CLUSTER_TYPES and len(norm_groups) > 1:
            print(f" → LLM clustering...", end="", flush=True)
            cluster_map = _llm_cluster(
                etype, norm_groups, key_field, llm,
                expert_field=expert_field,
                entity_desc=descs.get(etype, etype),
            )
        else:
            cluster_map = {nk: [nk] for nk in norm_groups}

        collapsed = sum(1 for members in cluster_map.values() if len(members) > 1)
        print(f" → {len(cluster_map)} canonical entities ({collapsed} synonym groups)")

        # Step 3: build merged registry entries
        entries = []
        for rep_nk, member_nks in cluster_map.items():
            # Collect all entities for this synonym group
            group_entities = []
            all_aliases: list[str] = []
            for nk in member_nks:
                grp = norm_groups.get(nk, [])
                group_entities.extend(grp)
                # Collect non-rep raw names as aliases
                if nk != rep_nk:
                    for e in grp:
                        raw = e.get(key_field, "")
                        if raw:
                            all_aliases.append(str(raw))

            if not group_entities:
                continue

            merged = _merge_cross_paper(group_entities, key_field)

            # canonical_name = first entity's raw name in the rep norm group
            rep_entities = norm_groups.get(rep_nk, group_entities)
            canonical_name = str(rep_entities[0].get(key_field, rep_nk))

            # Combine LLM-detected aliases with entity-level aliases
            entity_aliases = merged.pop("aliases", []) or []
            all_candidates = list(dict.fromkeys(
                [a for a in all_aliases if _norm_key(a) != _norm_key(canonical_name)]
                + entity_aliases
            ))

            entry: dict[str, Any] = {
                "canonical_id": f"{etype}::{rep_nk}",
                "canonical_name": canonical_name,
                "aliases": all_candidates,
            }
            entry.update({k: v for k, v in merged.items() if k not in ("sources", "raw_mention")})
            entry["source_papers"] = merged.get("sources", [])
            entries.append(entry)

        # Sort by canonical_name for readability
        entries.sort(key=lambda e: e["canonical_name"].lower())
        registry[etype] = entries

    return registry


def save_registry(registry: dict, structured_dir: Path) -> Path:
    import datetime
    registry["synthesis_date"] = datetime.date.today().isoformat()
    out_path = structured_dir / "entity_registry.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    return out_path
