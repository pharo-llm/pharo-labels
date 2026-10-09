"""Open (bottom-up) operation discovery, avoiding circular reasoning (Module 8).

The rule taxonomy can only find what it already names. These helpers surface
what it misses:

* ``uncovered_vocabulary`` — selectors no rule mentions, ranked by how many
  independent repositories use them;
* ``signature_groups`` — groups of unknown/ambiguous methods sharing their most
  informative selectors (IDF-weighted), i.e. candidate new operations;
* ``description_groups`` — groups of LLM method summaries by word overlap;
* ``select_methods_for_semantic_analysis`` — a stratified, capped subset of
  methods for LLM description (never the whole corpus).
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
import re

GENERIC = {"new", "yourself", "=", "==", "value", "ifTrue:", "ifFalse:", "ifTrue:ifFalse:", "isNil", "notNil",
           "ifNil:", "ifNotNil:", "size", "add:", "at:", "at:put:", "first", "last", "class", "name", "printString",
           "+", "-", "*", "/", ",", "<", ">", "<=", ">=", "~=", "and:", "or:", "not", "initialize", "do:", "isEmpty",
           "notEmpty", "assert:", "assert:equals:", "->", "@", "asString", "value:"}


def uncovered_vocabulary(rows, covered: set[str], min_repos: int = 2, top: int = 300) -> list[dict]:
    """rows: iterable of (repo_id, method_id, selector, class_name, method_selector)."""
    repos, methods, example = defaultdict(set), Counter(), {}
    for repo_id, method_id, sel, cls, msel in rows:
        if sel in covered:
            continue
        repos[sel].add(repo_id)
        methods[sel] += 1
        example.setdefault(sel, (method_id, f"{cls}>>{msel}"))
    out = [{"selector": s, "repositories": len(r), "methods": methods[s], "example_method_id": example[s][0],
            "example": example[s][1]} for s, r in repos.items() if len(r) >= min_repos]
    out.sort(key=lambda x: (-x["repositories"], -x["methods"]))
    return out[:top]


def signature_groups(methods: list[dict], n_repos_total: int, min_repos: int = 3, top: int = 200) -> list[dict]:
    """methods: dicts with method_id, repo_id, selectors (set), label (method label status)."""
    df = Counter()
    for m in methods:
        df.update({(m["repo_id"], s) for s in m["selectors"]})
    repo_df = Counter(s for (_r, s) in df)
    idf = {s: math.log((1 + n_repos_total) / (1 + c)) + 1 for s, c in repo_df.items()}
    groups = defaultdict(list)
    for m in methods:
        cands = [s for s in m["selectors"] if s not in GENERIC]
        if not cands:
            continue
        sig = tuple(sorted(sorted(cands, key=lambda s: (-idf.get(s, 0), s))[:2]))
        groups[sig].append(m)
    out = []
    for sig, ms in groups.items():
        rs = {m["repo_id"] for m in ms}
        if len(rs) < min_repos:
            continue
        out.append({"signature": " + ".join(sig), "methods": len(ms), "repositories": len(rs),
                    "example_method_ids": [m["method_id"] for m in ms[:5]],
                    "statuses": dict(Counter(m["label"] for m in ms))})
    out.sort(key=lambda g: (-g["repositories"], -g["methods"]))
    return out[:top]


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", (text or "").lower())
            if w not in {"the", "and", "with", "from", "into", "this", "that", "for", "each", "its", "returns", "method", "given"}}


def description_groups(descriptions: list[dict], threshold: float = 0.35) -> list[dict]:
    """Greedy leader clustering of LLM summaries by Jaccard word overlap."""
    leaders: list[dict] = []
    for d in descriptions:
        w = _words(d["summary"])
        best, best_sim = None, 0.0
        for g in leaders:
            sim = len(w & g["words"]) / len(w | g["words"]) if w | g["words"] else 0
            if sim > best_sim:
                best, best_sim = g, sim
        if best is not None and best_sim >= threshold:
            best["members"].append(d)
            best["words"] |= w
        else:
            leaders.append({"words": set(w), "members": [d]})
    out = []
    for g in leaders:
        top_words = Counter(x for m in g["members"] for x in _words(m["summary"])).most_common(6)
        out.append({"keywords": [w for w, _ in top_words], "size": len(g["members"]),
                    "repositories": len({m["repo_id"] for m in g["members"]}),
                    "example_summaries": [m["summary"] for m in g["members"][:3]],
                    "method_ids": [m["method_id"] for m in g["members"]]})
    out.sort(key=lambda g: -g["size"])
    return out


def select_methods_for_semantic_analysis(candidates: list[dict], n: int, seed: int, per_repo: int = 5) -> list[dict]:
    """candidates: dicts with method_id, repo_id, domain, status. Half the budget goes to
    unknown/ambiguous/multiple_operations methods, half to representative classified ones,
    stratified by domain and capped per repository."""
    rng = random.Random(seed)
    pool_hard = [c for c in candidates if c["status"] in ("unknown", "ambiguous", "multiple_operations")]
    pool_rep = [c for c in candidates if c["status"] == "classified"]
    chosen, per = [], Counter()

    def draw(pool, budget):
        by_domain = defaultdict(list)
        for c in pool:
            by_domain[c.get("domain") or "unknown"].append(c)
        for v in by_domain.values():
            rng.shuffle(v)
        taken = 0
        while taken < budget and any(by_domain.values()):
            for d in sorted(by_domain):
                if taken >= budget or not by_domain[d]:
                    continue
                c = by_domain[d].pop()
                if per[c["repo_id"]] >= per_repo:
                    continue
                per[c["repo_id"]] += 1
                chosen.append(c)
                taken += 1

    draw(pool_hard, n // 2)
    draw(pool_rep, n - len(chosen))
    return chosen
