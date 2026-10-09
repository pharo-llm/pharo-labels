"""Link taxonomy categories to concrete, diverse source evidence."""
from __future__ import annotations

from collections import defaultdict
import random

from ..storage.database import Database, loads


def evidence_examples(db: Database, operation_id: str, label_source: str = "rule_based", k: int = 5,
                      seed: int = 42) -> list[dict]:
    """Up to ``k`` examples from distinct repositories (one per repository)."""
    rows = db.query(
        "SELECT ml.method_id, m.repo_id, r.full_name, s.commit_sha, m.source_path, m.class_name, m.side, m.selector, m.line"
        " FROM method_labels ml JOIN methods m ON m.method_id = ml.method_id"
        " JOIN repositories r ON r.repo_id = m.repo_id JOIN repository_snapshots s ON s.snapshot_id = m.snapshot_id"
        " WHERE ml.operation_id=? AND ml.label_source=? AND m.duplicate_of_method_id IS NULL", (operation_id, label_source))
    by_repo = defaultdict(list)
    for r in rows:
        by_repo[r["repo_id"]].append(r)
    rng = random.Random(seed)
    repos = sorted(by_repo)
    rng.shuffle(repos)
    out = []
    for rid in repos[:k]:
        r = rng.choice(by_repo[rid])
        ev = db.query("SELECT evidence_json FROM operation_evidence WHERE method_id=? AND operation_id=? LIMIT 3",
                      (r["method_id"], operation_id))
        out.append({"repository": r["full_name"], "commit": r["commit_sha"], "file": r["source_path"],
                    "class": r["class_name"] + (" class" if r["side"] == "class" else ""), "method": r["selector"],
                    "line": r["line"], "method_id": r["method_id"],
                    "evidence": [loads(e["evidence_json"]) for e in ev]})
    return out


def cooccurrence(db: Database, label_source: str = "rule_based", min_methods: int = 20) -> dict[str, list[dict]]:
    """Related categories by lift of method-level co-occurrence."""
    rows = db.query("SELECT method_id, operation_id FROM method_labels WHERE label_source=?", (label_source,))
    by_method = defaultdict(set)
    for r in rows:
        by_method[r["method_id"]].add(r["operation_id"])
    n = len(by_method) or 1
    single = defaultdict(int)
    pair = defaultdict(int)
    for ops in by_method.values():
        ops = sorted(ops)
        for i, a in enumerate(ops):
            single[a] += 1
            for b in ops[i + 1:]:
                pair[(a, b)] += 1
    rel = defaultdict(list)
    for (a, b), c in pair.items():
        if c < min_methods:
            continue
        lift = (c / n) / ((single[a] / n) * (single[b] / n))
        if lift > 1.5:
            rel[a].append({"category": b, "lift": round(lift, 2), "methods": c})
            rel[b].append({"category": a, "lift": round(lift, 2), "methods": c})
    for k in rel:
        rel[k].sort(key=lambda x: -x["lift"])
        rel[k] = rel[k][:5]
    return rel
