"""Module 14 — export a stratified, blind annotation sample.

Files written to the validation directory (existing human-filled files are
never overwritten):

* ``annotation_sample.csv``  — what annotators see (no automatic labels: blind)
* ``annotation_key.csv``     — automatic labels per sample item (for analysis only)
* ``annotation_assignments.csv`` — which annotator reviews which item; an
  overlap subset is assigned to every annotator for agreement measurement
* ``reviewer_decisions.csv`` — template for annotator answers
* ``developer_feedback.csv`` — template for Pharo-developer taxonomy feedback
* ``annotation_guidelines.md`` — guidelines + current category list
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path
import random

from ..storage.database import Database, loads

REVIEWER_COLUMNS = ["sample_id", "annotator", "operations", "new_category", "too_broad", "too_narrow",
                    "multiple_operations", "evidence_sufficient", "missing_operations", "notes"]
FEEDBACK_COLUMNS = ["respondent", "respondent_role", "category_id", "recognizable_1to5", "important_in_work_1to5",
                    "deserves_benchmark_yes_no", "examples_representative_1to5", "missing_capabilities", "notes"]


def _write_csv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def select_sample(db: Database, size: int, seed: int, max_per_repo: int, include_tests: bool = True,
                  repo_ids: set[int] | None = None) -> list[dict]:
    rows = db.query(
        "SELECT m.method_id, m.repo_id, m.is_test, ms.status FROM methods m"
        " JOIN method_status ms ON ms.method_id=m.method_id AND ms.label_source='rule_based'"
        " WHERE m.parse_status='ok' AND m.duplicate_of_method_id IS NULL")
    labels = defaultdict(set)
    for r in db.query("SELECT method_id, operation_id FROM method_labels WHERE label_source='rule_based'"):
        labels[r["method_id"]].add(r["operation_id"])
    parent = {r["category_id"]: r["parent_id"] for r in db.query("SELECT category_id, parent_id FROM taxonomy_categories")}
    strata = defaultdict(list)
    for r in rows:
        if not include_tests and r["is_test"]:
            continue
        if repo_ids is not None and r["repo_id"] not in repo_ids:
            continue
        l2 = sorted({parent.get(o) for o in labels[r["method_id"]] if parent.get(o)})
        key = r["status"] if r["status"] != "classified" else (l2[0] if l2 else "classified")
        strata[key].append(dict(r))
    rng = random.Random(seed)
    for v in strata.values():
        rng.shuffle(v)
    chosen, per_repo = [], Counter()
    keys = sorted(strata)
    while len(chosen) < size and any(strata[k] for k in keys):
        for k in keys:
            while strata[k]:
                c = strata[k].pop()
                if per_repo[c["repo_id"]] < max_per_repo:
                    per_repo[c["repo_id"]] += 1
                    c["stratum"] = k
                    chosen.append(c)
                    break
            if len(chosen) >= size:
                break
    return chosen


def select_repositories(db: Database, repo_ids: set[int]) -> list[dict]:
    """Every parsed method of the given repositories (latest snapshot), except the methods copied
    from another repository. Same row shape as ``select_sample``."""
    if not repo_ids:
        return []
    labels = defaultdict(set)
    rows = db.query(
        "SELECT m.method_id, m.repo_id, m.is_test, ms.status FROM methods m"
        " JOIN method_status ms ON ms.method_id=m.method_id AND ms.label_source='rule_based'"
        " WHERE m.parse_status='ok' AND m.duplicate_of_method_id IS NULL AND m.repo_id IN (%s)"
        " AND m.snapshot_id IN (SELECT MAX(snapshot_id) FROM repository_snapshots GROUP BY repo_id)"
        " ORDER BY m.repo_id, m.method_id" % ",".join(map(str, sorted(repo_ids))))
    for r in db.query("SELECT ml.method_id, ml.operation_id FROM method_labels ml JOIN methods m ON m.method_id=ml.method_id"
                      " WHERE ml.label_source='rule_based' AND m.repo_id IN (%s)" % ",".join(map(str, sorted(repo_ids)))):
        labels[r["method_id"]].add(r["operation_id"])
    parent = {r["category_id"]: r["parent_id"] for r in db.query("SELECT category_id, parent_id FROM taxonomy_categories")}
    out = []
    for r in rows:
        l2 = sorted({parent.get(o) for o in labels[r["method_id"]] if parent.get(o)})
        out.append({**dict(r), "stratum": r["status"] if r["status"] != "classified" else (l2[0] if l2 else "classified")})
    return out


def export(db: Database, out_dir: Path, size: int, seed: int, max_per_repo: int, annotators: list[str],
           overlap_fraction: float, guidelines_src: Path | None, taxonomy_rows: list[dict],
           repo_ids: set[int] | None = None, chosen: list[dict] | None = None) -> dict:
    """``chosen``: the methods to export when they were already selected (whole repositories);
    otherwise a stratified sample of ``size`` methods is drawn."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "annotation_sample.csv"
    if sample_path.exists():
        return {"status": "kept_existing", "path": str(sample_path),
                "note": "annotation_sample.csv already exists; delete it explicitly to draw a new sample"}
    if chosen is None:
        chosen = select_sample(db, size, seed, max_per_repo, repo_ids=repo_ids)
    ids = [c["method_id"] for c in chosen]
    info = {}
    for start in range(0, len(ids), 5000):  # whole repositories: too many ids for one statement
        q = ("SELECT m.method_id, r.full_name, s.commit_sha, m.source_path, m.line, m.class_name, m.side, m.selector,"
             " m.protocol, m.source FROM methods m JOIN repositories r ON r.repo_id=m.repo_id"
             " JOIN repository_snapshots s ON s.snapshot_id=m.snapshot_id WHERE m.method_id IN (%s)"
             % ",".join(map(str, ids[start:start + 5000])))
        info.update({r["method_id"]: r for r in db.query(q)})
    domains = defaultdict(list)
    for r in db.query("SELECT repo_id, label FROM domain_labels WHERE method='deterministic_rules'"):
        domains[r["repo_id"]].append(r["label"])
    auto = defaultdict(list)
    for r in db.query("SELECT method_id, operation_id FROM method_labels WHERE label_source='rule_based'"):
        auto[r["method_id"]].append(r["operation_id"])
    sample_rows, key_rows = [], []
    for i, c in enumerate(chosen, 1):
        m = info[c["method_id"]]
        sid = f"S{i:05d}"
        sample_rows.append({"sample_id": sid, "method_id": c["method_id"], "repository": m["full_name"],
                            "commit": m["commit_sha"], "file": m["source_path"], "line": m["line"],
                            "class": m["class_name"] + (" class" if m["side"] == "class" else ""), "selector": m["selector"],
                            "protocol": m["protocol"], "repository_domains": ";".join(domains.get(c["repo_id"], [])),
                            "source": m["source"]})
        key_rows.append({"sample_id": sid, "method_id": c["method_id"], "stratum": c["stratum"],
                         "automatic_status": c["status"], "automatic_labels": ";".join(sorted(auto[c["method_id"]]))})
    cols = ["sample_id", "method_id", "repository", "commit", "file", "line", "class", "selector", "protocol",
            "repository_domains", "source"]
    _write_csv(sample_path, sample_rows, cols)
    _write_csv(out_dir / "annotation_key.csv", key_rows, ["sample_id", "method_id", "stratum", "automatic_status", "automatic_labels"])
    rng = random.Random(seed + 1)
    sids = [r["sample_id"] for r in sample_rows]
    overlap = set(rng.sample(sids, int(len(sids) * overlap_fraction))) if sids else set()
    assign, rr = [], 0
    for s in sids:
        if s in overlap:
            assign += [{"sample_id": s, "annotator": a, "overlap": 1} for a in annotators]
        else:
            assign.append({"sample_id": s, "annotator": annotators[rr % len(annotators)], "overlap": 0})
            rr += 1
    _write_csv(out_dir / "annotation_assignments.csv", assign, ["sample_id", "annotator", "overlap"])
    if not (out_dir / "reviewer_decisions.csv").exists():
        _write_csv(out_dir / "reviewer_decisions.csv",
                   [{"sample_id": a["sample_id"], "annotator": a["annotator"]} for a in assign], REVIEWER_COLUMNS)
    if not (out_dir / "developer_feedback.csv").exists():
        _write_csv(out_dir / "developer_feedback.csv",
                   [{"category_id": t["category_id"]} for t in taxonomy_rows if t["level"] == 3], FEEDBACK_COLUMNS)
    guide = (guidelines_src.read_text(encoding="utf-8") if guidelines_src and guidelines_src.exists() else "# Annotation guidelines\n")
    guide += "\n\n## Category reference (generated)\n\n| id | parent | definition |\n|---|---|---|\n"
    for t in taxonomy_rows:
        if t["level"] == 3:
            guide += f"| `{t['category_id']}` | {t['parent_id']} | {(t['definition'] or '').replace('|', '/')} |\n"
    (out_dir / "annotation_guidelines.md").write_text(guide, encoding="utf-8")
    return {"status": "exported", "items": len(sample_rows), "overlap_items": len(overlap),
            "strata": dict(Counter(k["stratum"] for k in key_rows)), "path": str(sample_path)}
