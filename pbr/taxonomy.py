"""Taxonomy manager: hierarchical behaviour taxonomy with stable IDs, aliases, versions and revisions.

Three levels (§2): category (``C001``) → subcategory (``S001``) → concrete behaviour (``B001``). IDs are
opaque and assigned once: renaming a concept keeps its ID; merging retires the absorbed IDs (pointing
to the survivor); splitting retires the old ID (pointing to the new ones). A revision never edits a
published version in place: it writes ``taxonomy_v<N+1>.json`` with ``parent_version``, the list of
changes and the map old ID → new IDs, so annotations made with an earlier version stay valid and can be
translated explicitly (never silently).

Two reserved answers exist in every version: ``UNKNOWN`` (no behaviour of the taxonomy fits) and
``INSUFFICIENT`` (the item does not give enough context to decide).

The evaluation task type (§2, second classification) is a separate, fixed vocabulary: ``TASK_TYPES``.
"""
from __future__ import annotations

from collections import Counter
import re

from .common import clone, now_iso, read_json, write_json

UNKNOWN = "UNKNOWN"
INSUFFICIENT = "INSUFFICIENT"
RESERVED = (UNKNOWN, INSUFFICIENT)

TASK_TYPES = {
    "code_generation": "write code (a function, class, program or repository) from a natural-language specification",
    "code_completion": "complete partial code (a body from its signature and documentation, the next lines)",
    "bug_repair": "change existing code so that it behaves as specified (fix a bug, resolve an issue)",
    "test_generation": "write tests, test inputs or expected outputs for given code",
    "other": "any other evaluated capability (predict execution results, act in a terminal, ...)",
}

LEVELS = {"category": ("categories", "C"), "subcategory": ("subcategories", "S"), "behavior": ("behaviors", "B")}


def empty(version: str) -> dict:
    return {"version": version, "created": now_iso(), "parent_version": None, "status": "draft",
            "categories": [], "subcategories": [], "behaviors": [], "changes": [], "id_map": {},
            "reserved": {UNKNOWN: "no behaviour of the taxonomy fits the item",
                         INSUFFICIENT: "the item does not give enough context to decide"},
            "task_types": TASK_TYPES}


def load(path) -> dict:
    tax = read_json(path)
    if tax is None:
        raise FileNotFoundError(path)
    return tax


def save(tax: dict, path) -> None:
    errors = validate(tax)
    if errors:
        raise ValueError("invalid taxonomy:\n- " + "\n- ".join(errors[:20]))
    write_json(path, tax)


def active(tax: dict, level: str) -> list[dict]:
    return [c for c in tax[LEVELS[level][0]] if c.get("status", "active") == "active"]


def index(tax: dict) -> dict:
    """id -> concept (every level, active or retired)."""
    return {c["id"]: c for lv in LEVELS.values() for c in tax[lv[0]]}


def next_id(tax: dict, level: str, registry: dict | None = None) -> str:
    """The next unused ID of a level (also unused in any earlier version recorded in ``registry``)."""
    key, prefix = LEVELS[level]
    used = {c["id"] for c in tax[key]} | set((registry or {}).get(level, []))
    n = max([int(i[1:]) for i in used if re.fullmatch(prefix + r"\d+", i)] + [0]) + 1
    return f"{prefix}{n:03d}"


def validate(tax: dict) -> list[str]:
    """Structural checks: unique IDs and names, existing parents, consistent category of each behaviour,
    definitions present, retired concepts pointing somewhere valid."""
    errors = []
    idx = {}
    for level, (key, prefix) in LEVELS.items():
        for c in tax[key]:
            cid = c.get("id", "")
            if not re.fullmatch(prefix + r"\d{3,}", cid):
                errors.append(f"{level} id {cid!r} does not match {prefix}NNN")
            if cid in idx:
                errors.append(f"duplicate id {cid}")
            idx[cid] = c
            if not (c.get("name") or "").strip():
                errors.append(f"{cid}: empty name")
            if c.get("status", "active") == "active" and not (c.get("definition") or "").strip():
                errors.append(f"{cid} {c.get('name')}: no definition")
    for level in LEVELS:
        names = Counter((c["name"].strip().lower(), c.get("category_id"), c.get("subcategory_id"))
                        for c in active(tax, level))
        errors += [f"{level} name {n[0]!r} used twice under the same parent" for n, k in names.items() if k > 1]
    for s in tax["subcategories"]:
        p = idx.get(s.get("category_id"))
        if not p or not p["id"].startswith("C"):
            errors.append(f"{s['id']}: unknown category {s.get('category_id')}")
        elif s.get("status", "active") == "active" and p.get("status", "active") != "active":
            errors.append(f"{s['id']}: active under retired category {p['id']}")
    for b in tax["behaviors"]:
        sub = idx.get(b.get("subcategory_id"))
        if not sub or not sub["id"].startswith("S"):
            errors.append(f"{b['id']}: unknown subcategory {b.get('subcategory_id')}")
            continue
        if b.get("category_id") != sub.get("category_id"):
            errors.append(f"{b['id']}: category {b.get('category_id')} differs from its subcategory's {sub.get('category_id')}")
        if b.get("status", "active") == "active" and sub.get("status", "active") != "active":
            errors.append(f"{b['id']}: active under retired subcategory {sub['id']}")
    for c in idx.values():
        for t in c.get("replaced_by") or []:
            if t not in idx and t not in RESERVED:      # a rejected concept points to UNKNOWN
                errors.append(f"{c['id']}: replaced_by unknown {t}")
    return errors


def behavior_path(tax: dict, bid: str) -> tuple[str, str, str]:
    """(category id, subcategory id, behaviour id); reserved answers map to themselves."""
    if bid in RESERVED:
        return bid, bid, bid
    b = index(tax).get(bid)
    if not b:
        return UNKNOWN, UNKNOWN, UNKNOWN
    return b["category_id"], b["subcategory_id"], b["id"]


def label(tax: dict, cid: str) -> str:
    if cid in RESERVED:
        return cid.capitalize()
    c = index(tax).get(cid)
    return f"{c['name']}" if c else cid


def prompt_text(tax: dict, with_examples: bool = False) -> str:
    """The taxonomy as annotators and experts read it: every active behaviour with its ID, definition,
    inclusion and exclusion rules, grouped by category and subcategory."""
    subs = {s["id"]: s for s in active(tax, "subcategory")}
    lines = []
    for c in active(tax, "category"):
        lines.append(f"## {c['id']} {c['name']}: {c['definition']}")
        for s in [s for s in subs.values() if s["category_id"] == c["id"]]:
            lines.append(f"### {s['id']} {s['name']}: {s['definition']}")
            for b in [b for b in active(tax, "behavior") if b["subcategory_id"] == s["id"]]:
                inc = "; ".join(b.get("include") or [])
                exc = "; ".join(b.get("exclude") or [])
                lines.append(f"- {b['id']} {b['name']}: {b['definition']}"
                             + (f" USE WHEN: {inc}." if inc else "") + (f" NOT WHEN: {exc}." if exc else ""))
                if with_examples:
                    for e in (b.get("examples") or [])[:1]:
                        if e.get("snippet"):
                            lines.append(f"    e.g. {e['snippet'][:160]}")
    lines.append(f"- {UNKNOWN}: {tax['reserved'][UNKNOWN]}")
    lines.append(f"- {INSUFFICIENT}: {tax['reserved'][INSUFFICIENT]}")
    return "\n".join(lines)


def behavior_ids(tax: dict) -> list[str]:
    return [b["id"] for b in active(tax, "behavior")]


# ---------------------------------------------------------------------------------------------- revisions
REVIEW_ACTIONS = ("accept", "rename", "redefine", "merge", "split", "reject", "add", "move")


def apply_decisions(tax: dict, decisions: list[dict], new_version: str, source: str) -> tuple[dict, list[str]]:
    """A new version from expert decisions (rows of the Taxonomy Review sheet).

    decision: {id, action, new_name, new_definition, merge_into, split_into ("name A | name B"),
               parent_id, include, exclude, reviewer, comment}
    Returns (new taxonomy, problems). Problems (unknown IDs, invalid actions) are reported, never guessed.
    """
    new = clone(tax)
    new.update(version=new_version, parent_version=tax["version"], created=now_iso(), changes=[], id_map={})
    idx = index(new)
    problems = []
    id_map = {c["id"]: [c["id"]] for c in idx.values()}

    def retire(c, targets, why):
        c["status"] = "retired"
        c["replaced_by"] = targets
        c["retired_reason"] = why
        id_map[c["id"]] = targets

    for d in decisions:
        action = (d.get("action") or "").strip().lower()
        cid = (d.get("id") or "").strip()
        who = d.get("reviewer") or "?"
        if not action or action == "accept":
            continue
        if action not in REVIEW_ACTIONS:
            problems.append(f"{cid}: unknown action {action!r}")
            continue
        if action != "add" and cid not in idx:
            problems.append(f"{cid}: unknown id for action {action}")
            continue
        c = idx.get(cid)
        change = {"action": action, "id": cid, "reviewer": who, "comment": d.get("comment") or "", "source": source}
        if action == "rename":
            if not d.get("new_name"):
                problems.append(f"{cid}: rename without new_name")
                continue
            change["old"], c["name"] = c["name"], d["new_name"].strip()
            c.setdefault("aliases", []).append(change["old"])
        elif action == "redefine":
            change["old"] = c.get("definition")
            if d.get("new_definition"):
                c["definition"] = d["new_definition"].strip()
            if d.get("include"):
                c["include"] = [x.strip() for x in str(d["include"]).split(";") if x.strip()]
            if d.get("exclude"):
                c["exclude"] = [x.strip() for x in str(d["exclude"]).split(";") if x.strip()]
        elif action == "merge":
            tgt = (d.get("merge_into") or "").strip()
            if tgt not in idx or tgt == cid or tgt[0] != cid[0]:
                problems.append(f"{cid}: merge_into {tgt!r} must be another existing id of the same level")
                continue
            idx[tgt].setdefault("aliases", []).extend([c["name"]] + (c.get("aliases") or []))
            idx[tgt].setdefault("examples", []).extend(c.get("examples") or [])
            if c.get("proposed_by") is not None:      # the survivor was proposed by every model of both
                idx[tgt]["proposed_by"] = sorted(set(idx[tgt].get("proposed_by") or []) | set(c["proposed_by"]))
            if c.get("sources") is not None:          # ... and comes from every source of both
                idx[tgt]["sources"] = sorted(set(idx[tgt].get("sources") or []) | set(c["sources"]))
            if c.get("items") is not None:            # distinct items classified under either (an item can be in both)
                idx[tgt]["items"] = list(dict.fromkeys((idx[tgt].get("items") or []) + c["items"]))
                idx[tgt]["n_items"] = len(idx[tgt]["items"])
            elif c.get("n_items") is not None:
                idx[tgt]["n_items"] = (idx[tgt].get("n_items") or 0) + c["n_items"]
            retire(c, [tgt], f"merged into {tgt} by {who}")
            # children of a merged parent move to the survivor
            for child in new["subcategories"] + new["behaviors"]:
                if child.get("category_id") == cid:
                    child["category_id"] = tgt
                if child.get("subcategory_id") == cid:
                    child["subcategory_id"] = tgt
                    child["category_id"] = idx[tgt].get("category_id", child.get("category_id"))
            change["into"] = tgt
        elif action == "split":
            names = [x.strip() for x in str(d.get("split_into") or "").split("|") if x.strip()]
            if len(names) < 2 or not cid.startswith("B"):
                problems.append(f"{cid}: split needs a behaviour id and 'name A | name B'")
                continue
            ids = []
            for nm in names:
                nid = next_id(new, "behavior")
                new["behaviors"].append({"id": nid, "name": nm, "subcategory_id": c["subcategory_id"],
                                         "category_id": c["category_id"],
                                         "definition": f"{nm} (split from {cid} {c['name']}; definition to be written)",
                                         "include": [], "exclude": [], "examples": [], "status": "active",
                                         "origin": f"split of {cid} by {who}"})
                idx[nid] = new["behaviors"][-1]
                ids.append(nid)
            retire(c, ids, f"split by {who}")
            change["into"] = ids
        elif action == "reject":
            retire(c, [UNKNOWN], f"rejected by {who}")
            if cid[0] in "CS":     # a rejected parent takes its children with it
                for child in new["subcategories"] + new["behaviors"]:
                    if child.get("status", "active") == "active" and cid in (child.get("category_id"), child.get("subcategory_id")):
                        retire(child, [UNKNOWN], f"parent {cid} rejected by {who}")
        elif action == "move":
            tgt = (d.get("parent_id") or "").strip()
            if cid.startswith("B") and tgt in idx and tgt.startswith("S"):
                c["subcategory_id"], c["category_id"] = tgt, idx[tgt]["category_id"]
            elif cid.startswith("S") and tgt in idx and tgt.startswith("C"):
                c["category_id"] = tgt
                for b in new["behaviors"]:
                    if b["subcategory_id"] == cid:
                        b["category_id"] = tgt
            else:
                problems.append(f"{cid}: move needs parent_id of the level above")
                continue
            change["to"] = tgt
        elif action == "add":
            tgt = (d.get("parent_id") or "").strip()
            name = (d.get("new_name") or "").strip()
            level = "behavior" if tgt.startswith("S") else "subcategory" if tgt.startswith("C") else "category"
            if not name or (tgt and tgt not in idx):
                problems.append(f"add: needs new_name and an existing parent_id (or none for a category): {d}")
                continue
            nid = next_id(new, level)
            item = {"id": nid, "name": name, "definition": (d.get("new_definition") or "").strip() or name,
                    "include": [], "exclude": [], "examples": [], "status": "active", "origin": f"added by {who}"}
            if level == "behavior":
                item.update(subcategory_id=tgt, category_id=idx[tgt]["category_id"])
            elif level == "subcategory":
                item.update(category_id=tgt)
            new[LEVELS[level][0]].append(item)
            idx[nid] = item
            change["new_id"] = nid
        new["changes"].append(change)
    new["id_map"] = {k: v for k, v in id_map.items() if v != [k]}
    problems += validate(new)
    return new, problems


def translate(ids: list[str], tax_from: dict, tax_to: dict) -> list[str]:
    """Behaviour IDs of an older version expressed in a newer one, following id_map chains (explicit,
    reported translation: the original annotation is never rewritten)."""
    out = []
    idx_to = index(tax_to)
    for i in ids:
        seen, frontier = set(), [i]
        while frontier:
            x = frontier.pop()
            if x in seen:
                continue
            seen.add(x)
            c = idx_to.get(x)
            if x in RESERVED or (c and c.get("status", "active") == "active"):
                out.append(x)
            elif c:
                frontier += c.get("replaced_by") or [UNKNOWN]
            else:
                out.append(UNKNOWN)
    return list(dict.fromkeys(out))
