"""Module 8 — rule-based candidate operations with evidence.

Rules come from the taxonomy file. Each match yields an evidence record that
names the matched selector/global/feature and its source line. Method-level
status follows the specification's special statuses: ``classified``,
``ambiguous`` (only weak evidence), ``multiple_operations``, ``unknown`` and
``insufficient_context``. Labels are always ``review_status = pending``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Clause:
    category: str
    strength: str
    selectors: set = field(default_factory=set)
    selector_patterns: list = field(default_factory=list)
    globals: set = field(default_factory=set)
    global_patterns: list = field(default_factory=list)
    defines_selectors: set = field(default_factory=set)
    defines_selector_patterns: list = field(default_factory=list)
    features: list = field(default_factory=list)
    side: str | None = None
    is_test: bool | None = None


def compile_rules(taxonomy: dict) -> list[Clause]:
    from ..taxonomy.category_manager import normalize_rules
    normalize_rules(taxonomy)
    out = []
    for cat in taxonomy.get("categories", []):
        for r in cat.get("rules") or []:
            out.append(Clause(
                category=cat["id"], strength=r.get("strength", "strong"),
                selectors=set(map(str, r.get("selectors") or [])),
                selector_patterns=[re.compile(p) for p in r.get("selector_patterns") or []],
                globals=set(r.get("globals") or []),
                global_patterns=[re.compile(p) for p in r.get("global_patterns") or []],
                defines_selectors=set(map(str, r.get("defines_selectors") or [])),
                defines_selector_patterns=[re.compile(p) for p in r.get("defines_selector_patterns") or []],
                features=list(r.get("features") or []), side=r.get("side"), is_test=r.get("is_test"),
            ))
    return out


def _match_clause(c: Clause, m: dict, features: dict) -> list[dict] | None:
    """Return evidence items if every criterion of the clause matches, else None."""
    ev = []
    if c.side and m["side"] != c.side:
        return None
    if c.is_test is not None and bool(m.get("is_test")) != c.is_test:
        return None
    if c.features:
        if not all(features.get(f) for f in c.features):
            return None
        ev.append({"kind": "structural_feature", "features": c.features})
    if c.defines_selectors or c.defines_selector_patterns:
        sel = m["selector"]
        if sel in c.defines_selectors or any(p.search(sel) for p in c.defines_selector_patterns):
            ev.append({"kind": "defined_selector", "selector": sel, "line": m.get("line")})
        else:
            return None
    if c.selectors or c.selector_patterns:
        hits = [s for s in m.get("message_sends") or []
                if s["selector"] in c.selectors or any(p.search(s["selector"]) for p in c.selector_patterns)]
        if not hits:
            return None
        seen = set()
        for s in hits:
            if s["selector"] in seen:
                continue
            seen.add(s["selector"])
            ev.append({"kind": "message_send", "selector": s["selector"], "line": s.get("line"),
                       "ast_node": "message_send", "receiver_kind": s.get("receiver_kind")})
    if c.globals or c.global_patterns:
        hits = [g for g in m.get("referenced_globals") or []
                if g["name"] in c.globals or any(p.search(g["name"]) for p in c.global_patterns)]
        if not hits:
            return None
        seen = set()
        for g in hits:
            if g["name"] in seen:
                continue
            seen.add(g["name"])
            ev.append({"kind": "global_reference", "name": g["name"], "line": g.get("line"), "ast_node": "variable"})
    return ev or None


@dataclass
class MethodClassification:
    status: str
    labels: dict  # category -> {"strength": strong|weak, "evidence": [...]}


class RuleClassifier:
    def __init__(self, taxonomy: dict, multiple_threshold: int = 4, insufficient_max_nodes: int = 3):
        self.taxonomy = taxonomy
        self.version = str(taxonomy.get("version", "0"))
        self.clauses = compile_rules(taxonomy)
        self.parent = {c["id"]: c.get("parent") for c in taxonomy.get("categories", [])}
        self.multiple_threshold = multiple_threshold
        self.insufficient_max_nodes = insufficient_max_nodes

    def covered_selectors(self) -> set[str]:
        s = set()
        for c in self.clauses:
            s |= c.selectors
        return s

    def classify(self, method: dict, features: dict) -> MethodClassification:
        labels: dict[str, dict] = {}
        for c in self.clauses:
            ev = _match_clause(c, method, features)
            if ev is None:
                continue
            cur = labels.setdefault(c.category, {"strength": "weak", "evidence": []})
            if c.strength == "strong":
                cur["strength"] = "strong"
            for e in ev:
                e = dict(e, rule_strength=c.strength)
                if e not in cur["evidence"]:
                    cur["evidence"].append(e)
        strong = {k for k, v in labels.items() if v["strength"] == "strong"}
        l2 = {self.parent.get(k) for k in strong}
        if not labels:
            small = (features.get("n_ast_nodes", 0) <= self.insufficient_max_nodes or features.get("is_empty")
                     or features.get("n_message_sends", 0) == 0)
            status = "insufficient_context" if small else "unknown"
        elif not strong:
            status = "ambiguous"
        elif len(l2) >= self.multiple_threshold:
            status = "multiple_operations"
        else:
            status = "classified"
        return MethodClassification(status, labels)
