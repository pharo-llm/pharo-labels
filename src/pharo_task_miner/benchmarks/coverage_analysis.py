"""Modules 15–16 — benchmark coverage and candidate recommendations.

For every taxonomy operation: covering tasks, independent benchmarks,
portable vs Pharo-native coverage, adaptation needs, difficulty concentration
and capability preservation. Recommendations use explicit, documented rules
(no opaque score) and are always ``review_status = pending``.
"""
from __future__ import annotations

from collections import Counter

NATIVE = {"native_pharo", "transferred_validated"}
PORTABLE = {"portable", "not_evaluated"}

RULES_DOC = [
    "needs_further_investigation: category not validated by humans, or repository prevalence below threshold",
    "reuse_existing_task: >=1 Pharo-native or validated-transferred task preserving the capability",
    "adapt_explicitly: covered only by tasks needing adaptation / not preserving the capability, "
    "or a Pharo-specific capability covered only by portable tasks",
    "transfer_to_pharo: general capability covered by portable tasks not yet transferred",
    "create_native_pharo_task: not covered by any task",
]


def coverage_rows(operations: list[dict], coverage: dict[str, list[dict]], min_full: int) -> list[dict]:
    """operations: dicts with id, name, level, concept_type, status, repo_prevalence, repositories, domains."""
    rows = []
    for op in operations:
        tasks = coverage.get(op["id"], [])
        statuses = Counter(t["transfer_status"] for t in tasks)
        preserving = [t for t in tasks if t["preserves_capability"] not in ("false", "no", "0")]
        native = [t for t in preserving if t["transfer_status"] in NATIVE]
        portable = [t for t in tasks if t["transfer_status"] in PORTABLE]
        adapt = [t for t in tasks if t["transfer_status"] == "needs_adaptation" or t["preserves_capability"] in ("false", "no", "0")]
        difficulties = Counter(t["difficulty"] for t in tasks if t["difficulty"])
        if not tasks:
            level = "none"
        elif len(preserving) >= min_full and len({t["benchmark"] for t in preserving}) >= 1:
            level = "full"
        else:
            level = "partial"
        rows.append({
            "operation": op["id"], "name": op.get("name"), "level": op.get("level"), "concept_type": op.get("concept_type"),
            "taxonomy_status": op.get("status"), "repo_prevalence": op.get("repo_prevalence"),
            "observed_in_repositories": op.get("repositories"), "domain_count": op.get("domains"),
            "covering_tasks": len(tasks), "independent_benchmarks": len({t["benchmark"] for t in tasks}),
            "native_or_validated_tasks": len(native), "portable_tasks": len(portable), "tasks_needing_adaptation": len(adapt),
            "transfer_statuses": dict(statuses), "difficulties": dict(difficulties),
            "difficulty_concentrated": bool(difficulties) and (max(difficulties.values()) / sum(difficulties.values()) > 0.8)
                                       and sum(difficulties.values()) >= 3,
            "existing_benchmark_coverage": level,
            "_tasks": tasks,
        })
    return rows


def recommend(row: dict, min_prevalence: float, validated: bool, expert: dict | None) -> dict:
    reasons = []
    prev = row.get("repo_prevalence") or 0.0
    if not validated:
        rec = "needs_further_investigation"
        reasons.append("taxonomy category not yet validated by human annotation")
    elif prev < min_prevalence:
        rec = "needs_further_investigation"
        reasons.append(f"repository prevalence {prev:.1%} below {min_prevalence:.0%}")
    elif row["native_or_validated_tasks"] >= 1:
        rec = "reuse_existing_task"
        reasons.append("a Pharo-native or validated transferred task preserves the capability")
    elif row["tasks_needing_adaptation"] >= 1 or (row["concept_type"] != "general" and row["portable_tasks"] >= 1):
        rec = "adapt_explicitly"
        reasons.append("existing coverage requires explicit adaptation to preserve the Pharo capability")
    elif row["portable_tasks"] >= 1:
        rec = "transfer_to_pharo"
        reasons.append("portable tasks exist; behavioural contract can be transferred")
    else:
        rec = "create_native_pharo_task"
        reasons.append("no existing benchmark task covers this operation")
    out = {
        "operation": row["operation"], "observed_in_repositories": row.get("observed_in_repositories"),
        "repo_prevalence": row.get("repo_prevalence"), "domain_count": row.get("domain_count"),
        "concept_type": row.get("concept_type"), "existing_benchmark_coverage": row["existing_benchmark_coverage"],
        "recommendation": rec, "reasons": reasons,
        "supporting_evidence": row.get("supporting_repositories", []),
        "expert_importance": expert, "review_status": "pending",
        "basis": "human_validated_taxonomy" if validated else "automatic_unvalidated",
    }
    return out
