"""Map benchmark tasks to taxonomy categories (explicit declarations only).

A task covers an operation only if its ``operations`` list contains the
operation id or one of its descendants. No keyword matching is performed.
"""
from __future__ import annotations

from collections import defaultdict

from ..storage.database import Database, loads


def task_coverage(db: Database, parent: dict[str, str | None]) -> dict[str, list[dict]]:
    """operation id (any level) -> list of covering tasks."""
    cov = defaultdict(list)
    for t in db.query("SELECT * FROM benchmark_tasks"):
        ops = loads(t["operations_json"], []) or []
        covered = set()
        for o in ops:
            while o:
                covered.add(o)
                o = parent.get(o)
        for o in covered:
            cov[o].append({"benchmark": t["benchmark"], "task_id": t["task_id"], "transfer_status": t["transfer_status"],
                           "difficulty": t["difficulty"], "preserves_capability": t["preserves_capability"],
                           "origin": t["origin"]})
    return cov
