"""CSV exports of the evidence database (streamed, so large corpora fit in memory)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..storage.database import Database, loads


def _writer(path: Path, cols: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", encoding="utf-8", newline="")
    w = csv.writer(f)
    w.writerow(cols)
    return f, w


def repositories_csv(db: Database, path: Path) -> int:
    cols = ["repo_id", "full_name", "url", "description", "topics", "language", "stars", "forks", "contributors", "commit_count",
            "created_at", "updated_at", "pushed_at", "archived", "is_fork", "parent_full_name", "license", "default_branch",
            "head_commit_sha", "analysed_commit_sha", "pharo_status", "pharo_evidence", "verification_source", "discovery_strategies",
            "discovery_queries", "discovery_date", "metadata_source", "n_source_files", "n_methods", "n_parse_failures", "clone_status"]
    snaps = {r["repo_id"]: r for r in db.query(
        "SELECT * FROM repository_snapshots WHERE snapshot_id IN (SELECT MAX(snapshot_id) FROM repository_snapshots GROUP BY repo_id)")}
    f, w = _writer(path, cols)
    n = 0
    for r in db.query("SELECT * FROM repositories ORDER BY full_name"):
        disc = loads(r["discovery_json"], []) or []
        s = snaps.get(r["repo_id"])
        tree = loads(r["tree_summary_json"], {}) or {}
        w.writerow([r["repo_id"], r["full_name"], r["url"], r["description"], ";".join(loads(r["topics_json"], []) or []),
                    r["language"], r["stars"], r["forks"], r["contributors"], r["commit_count"], r["created_at"], r["updated_at"],
                    r["pushed_at"], r["archived"], r["is_fork"], r["parent_full_name"], r["license"], r["default_branch"],
                    r["head_commit_sha"], s["commit_sha"] if s else None, r["pharo_status"],
                    "; ".join(f"{e['kind']}: {e['detail']}" for e in loads(r["pharo_evidence_json"], []) or []),
                    r["verification_source"], ";".join(sorted({d["strategy"] for d in disc})),
                    " | ".join(sorted({d["query"] for d in disc})), r["discovery_date"], r["metadata_source"],
                    tree.get("n_source_files"), s["n_methods"] if s else None, s["n_parse_failures"] if s else None,
                    s["clone_status"] if s else None])
        n += 1
    f.close()
    return n


def repository_domains_csv(db: Database, path: Path) -> int:
    f, w = _writer(path, ["full_name", "label", "method", "score", "score_kind", "review_status", "model", "prompt_version", "evidence"])
    n = 0
    for r in db.query("SELECT r.full_name, d.* FROM domain_labels d JOIN repositories r ON r.repo_id=d.repo_id ORDER BY r.full_name, d.score DESC"):
        ev = loads(r["evidence_json"], []) or []
        w.writerow([r["full_name"], r["label"], r["method"], r["score"], r["score_kind"], r["review_status"], r["model"],
                    r["prompt_version"], " || ".join(f"[{e.get('source')}] {e.get('excerpt')}" for e in ev)])
        n += 1
    f.close()
    return n


def methods_csv(db: Database, path: Path) -> int:
    cols = ["method_id", "repository", "commit_sha", "package", "class", "side", "selector", "protocol", "source_path", "line",
            "container_format", "parse_status", "parse_errors", "is_test", "is_extension", "in_trait", "duplicate_of_method_id",
            "n_ast_nodes", "n_message_sends", "n_blocks", "max_nesting_depth", "rule_status", "rule_operations", "source_hash"]
    status = {r[0]: r[1] for r in db.query("SELECT method_id, status FROM method_status WHERE label_source='rule_based'")}
    labels = {}
    for r in db.query("SELECT method_id, operation_id FROM method_labels WHERE label_source='rule_based'"):
        labels.setdefault(r[0], []).append(r[1])
    f, w = _writer(path, cols)
    n = 0
    cur = db.execute("SELECT m.*, r.full_name, s.commit_sha FROM methods m JOIN repositories r ON r.repo_id=m.repo_id"
                     " JOIN repository_snapshots s ON s.snapshot_id=m.snapshot_id ORDER BY m.method_id")
    for m in cur:
        ft = loads(m["features_json"], {}) or {}
        w.writerow([m["method_id"], m["full_name"], m["commit_sha"], m["package"], m["class_name"], m["side"], m["selector"],
                    m["protocol"], m["source_path"], m["line"], m["container_format"], m["parse_status"],
                    "; ".join(loads(m["parse_errors_json"], []) or []), m["is_test"], m["is_extension"], m["in_trait"],
                    m["duplicate_of_method_id"], ft.get("n_ast_nodes"), ft.get("n_message_sends"), ft.get("n_blocks"),
                    ft.get("max_nesting_depth"), status.get(m["method_id"]), ";".join(sorted(labels.get(m["method_id"], []))),
                    m["source_hash"]])
        n += 1
    f.close()
    return n


def operation_evidence_csv(db: Database, path: Path) -> int:
    cols = ["evidence_id", "repository", "commit", "file", "class", "method", "line", "operation", "evidence_kind", "evidence",
            "strength", "classification", "taxonomy_version", "review_status"]
    f, w = _writer(path, cols)
    n = 0
    cur = db.execute(
        "SELECT e.*, r.full_name, s.commit_sha, m.source_path, m.class_name, m.side, m.selector FROM operation_evidence e"
        " JOIN methods m ON m.method_id=e.method_id JOIN repositories r ON r.repo_id=m.repo_id"
        " JOIN repository_snapshots s ON s.snapshot_id=m.snapshot_id ORDER BY e.evidence_id")
    for e in cur:
        ev = loads(e["evidence_json"], {}) or {}
        w.writerow([e["evidence_id"], e["full_name"], e["commit_sha"], e["source_path"],
                    e["class_name"] + (" class" if e["side"] == "class" else ""), e["selector"], ev.get("line"),
                    e["operation_id"], e["evidence_kind"], json.dumps({k: v for k, v in ev.items() if k not in ("line", "kind", "rule_strength")}),
                    e["strength"], e["classification"], e["taxonomy_version"], e["review_status"]])
        n += 1
    f.close()
    return n
