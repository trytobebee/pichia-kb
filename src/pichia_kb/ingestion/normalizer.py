"""Layer 3.5: within-paper entity normalization (deduplication + field merging).

Groups entities by normalized name, merges provenance and fields, and collapses
name variants into ``aliases``.  Operates directly on ExtractionResult JSON dicts
so it works without importing schema models (avoids circular deps and is easier
to test standalone).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ── Unicode roman numeral → ASCII ─────────────────────────────────────────────

_ROMAN = {
    'Ⅰ': 'I', 'Ⅱ': 'II', 'Ⅲ': 'III', 'Ⅳ': 'IV', 'Ⅴ': 'V',
    'Ⅵ': 'VI', 'Ⅶ': 'VII', 'Ⅷ': 'VIII', 'Ⅸ': 'IX', 'Ⅹ': 'X',
}


def _norm_key(s: str) -> str:
    """Canonical grouping key: ASCII roman numerals, strip, lowercase."""
    for uc, asc in _ROMAN.items():
        s = s.replace(uc, asc)
    return s.strip().lower()


# ── Field-level merge helpers ──────────────────────────────────────────────────

def _best_confidence(values: list[str | None]) -> str | None:
    for v in values:
        if v == 'explicit':
            return 'explicit'
    for v in values:
        if v == 'inferred':
            return 'inferred'
    return None


def _union(lists: list[list]) -> list:
    seen: dict = {}
    for lst in lists:
        for item in lst:
            key = json.dumps(item, sort_keys=True) if isinstance(item, dict) else item
            seen[key] = item
    return list(seen.values())


def _merge_dicts(a: dict | None, b: dict | None) -> dict | None:
    if a is None:
        return b
    if b is None:
        return a
    return {**b, **a}  # a (earlier record) takes precedence


def _merge_notes(*notes: str) -> str:
    parts = [n.strip() for n in notes if n and n.strip()]
    seen: dict[str, bool] = {}
    unique = []
    for p in parts:
        if p not in seen:
            seen[p] = True
            unique.append(p)
    return ' | '.join(unique)


# ── Group + merge one entity type ─────────────────────────────────────────────

_ENTITY_KEY: dict[str, str] = {
    'strains':               'name',
    'promoters':             'name',
    'vectors':               'name',
    'media':                 'name',
    'target_products':       'name',
    'analytical_methods':    'name',
    'process_parameters':    'parameter_name',
    # FermentationConditionFact and GlycosylationPattern have no stable single-
    # field key, so they are passed through without merging.
}

_PROVENANCE_FIELDS = {'sources', 'chunk_ids', 'raw_mention', 'extraction_confidence'}


def _merge_group(group: list[dict[str, Any]], key_field: str) -> dict[str, Any]:
    """Merge a list of entity dicts that share the same normalized key."""
    if len(group) == 1:
        return group[0]

    # Canonical name = first occurrence's value
    canonical = group[0][key_field]
    name_variants = list(dict.fromkeys(
        e[key_field] for e in group
        if e.get(key_field) and _norm_key(e[key_field]) != _norm_key(canonical)
    ))

    # Provenance union
    all_sources  = _union([e.get('sources', [])   for e in group])
    all_chunks   = _union([e.get('chunk_ids', []) for e in group])
    all_mentions = [e['raw_mention'] for e in group if e.get('raw_mention')]
    merged_mention = ' | '.join(dict.fromkeys(all_mentions)) or None

    # Start from first record, then fold in subsequent ones
    merged: dict[str, Any] = dict(group[0])
    merged['sources']              = all_sources
    merged['chunk_ids']            = all_chunks
    merged['raw_mention']          = merged_mention
    merged['extraction_confidence'] = _best_confidence(
        [e.get('extraction_confidence') for e in group]
    )

    for entity in group[1:]:
        for field, val in entity.items():
            if field in _PROVENANCE_FIELDS or field == key_field:
                continue

            if isinstance(val, list):
                merged[field] = _union([merged.get(field) or [], val])
            elif isinstance(val, dict):
                merged[field] = _merge_dicts(merged.get(field), val)
            elif isinstance(val, bool):
                if merged.get(field) is None and val is not None:
                    merged[field] = val
                elif val is True:
                    merged[field] = True
            elif field == 'notes':
                merged['notes'] = _merge_notes(merged.get('notes', ''), val)
            else:
                # Optional[str] — first non-empty wins
                if not merged.get(field) and val:
                    merged[field] = val

    # Aliases: name variants + existing aliases from all records
    existing_aliases = _union([e.get('aliases', []) for e in group])
    all_candidates = list(dict.fromkeys(name_variants + existing_aliases))
    merged['aliases'] = [a for a in all_candidates if _norm_key(a) != _norm_key(canonical)]

    return merged


def _normalize_entity_list(entities: list[dict], key_field: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    ungrouped: list[dict] = []

    for entity in entities:
        raw = entity.get(key_field)
        if not raw:
            ungrouped.append(entity)
            continue
        key = _norm_key(str(raw))
        groups.setdefault(key, []).append(entity)

    merged = [_merge_group(g, key_field) for g in groups.values()]
    return merged + ungrouped


# ── Top-level API ──────────────────────────────────────────────────────────────

def normalize_result(result: dict) -> tuple[dict, dict[str, int]]:
    """Normalize one ExtractionResult dict in place; return (result, stats).

    stats maps entity_type → records_removed.
    """
    stats: dict[str, int] = {}
    for entity_type, key_field in _ENTITY_KEY.items():
        original = result.get(entity_type, [])
        if not original:
            continue
        normalized = _normalize_entity_list(original, key_field)
        removed = len(original) - len(normalized)
        stats[entity_type] = removed
        result[entity_type] = normalized
    return result, stats


def normalize_all(structured_dir: Path, dry_run: bool = False) -> dict[str, dict[str, int]]:
    """Normalize all extraction JSONs in *structured_dir*.

    Returns per-file stats: {filename: {entity_type: records_removed}}.
    """
    all_stats: dict[str, dict[str, int]] = {}

    for json_path in sorted(structured_dir.glob('*.pdf.json')):
        with json_path.open(encoding='utf-8') as f:
            data = json.load(f)

        data, stats = normalize_result(data)
        all_stats[json_path.name] = stats

        if not dry_run:
            with json_path.open('w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    return all_stats
