"""Load, validate and version the operation taxonomy."""
from __future__ import annotations

from pathlib import Path

from ..config import load_yaml
from ..storage.database import Database, dumps

SPECIAL_STATUSES = ("unknown", "ambiguous", "multiple_operations", "insufficient_context")
CONCEPT_TYPES = ("general", "pharo_specific", "framework_specific")


class TaxonomyError(ValueError):
    pass


SELECTOR_KEYS = ("selectors", "defines_selectors")


def _selector(item) -> str:
    """Unquoted keyword selectors in YAML flow lists (``[select:, at:put:]``) are
    parsed as one-key mappings ({'select': None}); turn them back into selectors."""
    if isinstance(item, dict) and len(item) == 1:
        (k, v), = item.items()
        if v is None:
            return f"{k}:"
        raise TaxonomyError(f"cannot interpret selector entry {item!r}; quote it")
    return str(item)


def normalize_rules(tax: dict) -> dict:
    for c in tax.get("categories", []):
        for r in c.get("rules") or []:
            for key in SELECTOR_KEYS:
                if key in r and r[key] is not None:
                    r[key] = [_selector(x) for x in r[key]]
    return tax


def load_taxonomy(path: Path) -> dict:
    tax = normalize_rules(load_yaml(Path(path)))
    validate(tax)
    return tax


def validate(tax: dict):
    cats = tax.get("categories") or []
    ids = [c.get("id") for c in cats]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise TaxonomyError(f"duplicate category ids: {sorted(dup)}")
    known = set(ids)
    for c in cats:
        if not c.get("id") or not c.get("level"):
            raise TaxonomyError(f"category without id/level: {c}")
        if c["id"] in SPECIAL_STATUSES:
            raise TaxonomyError(f"{c['id']} is reserved for method statuses")
        if c["level"] > 1 and c.get("parent") not in known:
            raise TaxonomyError(f"{c['id']}: unknown parent {c.get('parent')!r}")
        if c.get("concept_type", "general") not in CONCEPT_TYPES:
            raise TaxonomyError(f"{c['id']}: concept_type must be one of {CONCEPT_TYPES}")
        if c["level"] == 3 and not c.get("rules"):
            # allowed (e.g. categories proposed by annotators) but flagged
            c.setdefault("status", "no_rules")


def by_id(tax: dict) -> dict[str, dict]:
    return {c["id"]: c for c in tax.get("categories", [])}


def ancestors(tax_index: dict[str, dict], cid: str) -> list[str]:
    out = []
    cur = tax_index.get(cid, {}).get("parent")
    while cur:
        out.append(cur)
        cur = tax_index.get(cur, {}).get("parent")
    return out


def store_taxonomy(db: Database, tax: dict, status: str = "seed"):
    version = str(tax.get("version"))
    for c in tax.get("categories", []):
        db.upsert("taxonomy_categories", {
            "taxonomy_version": version, "category_id": c["id"], "parent_id": c.get("parent"), "level": c["level"],
            "name": c.get("name", c["id"]), "definition": c.get("definition"), "include_json": dumps(c.get("include") or []),
            "exclude_json": dumps(c.get("exclude") or []), "concept_type": c.get("concept_type", "general"),
            "status": c.get("status", status), "rules_json": dumps(c.get("rules") or []),
            "related_json": dumps(c.get("related") or [])}, ["taxonomy_version", "category_id"])
    db.commit()
