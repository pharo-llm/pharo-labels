"""Deduplication: forks, mirrors, exact duplicates and duplicated packages/code.

Repository identity (GitHub id, fork/parent links) and source-content
fingerprints (tree SHA, package fingerprints, method source hashes) are both
recorded so that frequency statistics can be computed with and without
duplicated code (sensitivity analysis), instead of silently dropping data.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib

from ..storage.database import Database, dumps, loads

TRIVIAL_NODE_LIMIT = 6  # tiny accessors are identical by chance; ignore them for similarity


def _priority(row) -> tuple:
    # canonical copy: not a fork, more stars, older
    return (int(row["is_fork"] or 0), -(row["stars"] or 0), row["created_at"] or "9999")


def metadata_duplicates(db: Database, run_id: str) -> dict:
    rows = db.query("SELECT repo_id, full_name, name, description, is_fork, parent_full_name, stars, created_at,"
                    " tree_summary_json FROM repositories")
    by_name = {r["full_name"].lower(): r for r in rows}
    counts = defaultdict(int)
    db.execute("DELETE FROM repository_duplicates WHERE kind IN ('fork','exact_tree_duplicate','mirror_candidate')")
    for r in rows:
        if r["is_fork"]:
            parent = by_name.get((r["parent_full_name"] or "").lower())
            db.upsert("repository_duplicates", {
                "repo_id": r["repo_id"], "other_repo_id": parent["repo_id"] if parent else -1, "kind": "fork",
                "similarity": None, "detail": f"fork of {r['parent_full_name'] or 'unknown parent'}",
                "canonical_repo_id": parent["repo_id"] if parent else None, "run_id": run_id}, ["repo_id", "other_repo_id", "kind"])
            counts["fork"] += 1
    groups = defaultdict(list)
    for r in rows:
        sha = (loads(r["tree_summary_json"], {}) or {}).get("tree_sha")
        if sha:
            groups[("tree", sha)].append(r)
        if r["name"] and r["description"]:
            groups[("name_desc", r["name"].lower(), r["description"].strip().lower())].append(r)
    for key, members in groups.items():
        if len(members) < 2:
            continue
        kind = "exact_tree_duplicate" if key[0] == "tree" else "mirror_candidate"
        canonical = sorted(members, key=_priority)[0]
        for m in members:
            if m["repo_id"] == canonical["repo_id"]:
                continue
            db.upsert("repository_duplicates", {
                "repo_id": m["repo_id"], "other_repo_id": canonical["repo_id"], "kind": kind,
                "similarity": 1.0 if kind == "exact_tree_duplicate" else None,
                "detail": f"same {'git tree SHA' if kind == 'exact_tree_duplicate' else 'name and description'} as {canonical['full_name']}",
                "canonical_repo_id": canonical["repo_id"], "run_id": run_id}, ["repo_id", "other_repo_id", "kind"])
            counts[kind] += 1
    db.commit()
    return dict(counts)


def package_fingerprint(hashes: list[str]) -> str:
    return hashlib.sha1("\n".join(sorted(hashes)).encode()).hexdigest()


def content_duplicates(db: Database, run_id: str, threshold: float = 0.8) -> dict:
    """Repository pairs sharing most non-trivial method bodies; duplicated packages;
    and per-method ``duplicate_of_method_id`` pointing to the canonical copy."""
    repos = {r["repo_id"]: r for r in db.query("SELECT repo_id, full_name, is_fork, stars, created_at FROM repositories")}
    snaps = db.query("SELECT snapshot_id, repo_id FROM repository_snapshots WHERE n_methods > 0")
    latest = {}
    for s in snaps:
        latest[s["repo_id"]] = max(latest.get(s["repo_id"], 0), s["snapshot_id"])
    order = sorted(latest, key=lambda rid: _priority(repos[rid]) if rid in repos else (9, 0, ""))
    rank = {rid: i for i, rid in enumerate(order)}
    hashes_by_repo: dict[int, set[str]] = defaultdict(set)
    first_owner: dict[str, tuple[int, int]] = {}
    updates = []
    rows = db.query("SELECT m.method_id, m.repo_id, m.source_hash, m.package, m.snapshot_id, m.features_json"
                    " FROM methods m WHERE m.parse_status='ok'")
    pkg_hashes = defaultdict(list)
    for m in rows:
        if latest.get(m["repo_id"]) != m["snapshot_id"]:
            continue
        pkg_hashes[(m["repo_id"], m["package"])].append(m["source_hash"])
        nodes = (loads(m["features_json"], {}) or {}).get("n_ast_nodes", TRIVIAL_NODE_LIMIT + 1)
        if nodes > TRIVIAL_NODE_LIMIT:
            hashes_by_repo[m["repo_id"]].add(m["source_hash"])
    # canonical copy of each non-trivial hash = highest-priority repository
    for m in sorted(rows, key=lambda m: (rank.get(m["repo_id"], 1 << 30), m["method_id"])):
        if latest.get(m["repo_id"]) != m["snapshot_id"]:
            continue
        if (loads(m["features_json"], {}) or {}).get("n_ast_nodes", TRIVIAL_NODE_LIMIT + 1) <= TRIVIAL_NODE_LIMIT:
            continue  # trivial bodies (accessors, constants) coincide without copying
        h = m["source_hash"]
        if h in first_owner and first_owner[h][0] != m["repo_id"]:
            updates.append((first_owner[h][1], m["method_id"]))
        elif h not in first_owner:
            first_owner[h] = (m["repo_id"], m["method_id"])
    db.execute("UPDATE methods SET duplicate_of_method_id=NULL")
    db.executemany("UPDATE methods SET duplicate_of_method_id=? WHERE method_id=?", updates)
    # package fingerprints
    fp_groups = defaultdict(list)
    for (rid, pkg), hs in pkg_hashes.items():
        if len(hs) >= 3:
            fp = package_fingerprint(hs)
            fp_groups[fp].append((rid, pkg))
            db.execute("UPDATE packages SET fingerprint=? WHERE snapshot_id=? AND name=?", (fp, latest[rid], pkg))
    db.execute("DELETE FROM repository_duplicates WHERE kind IN ('substantial_code_overlap','duplicated_package')")
    n_pkg = 0
    for fp, members in fp_groups.items():
        rids = sorted({rid for rid, _ in members}, key=lambda rid: rank.get(rid, 1 << 30))
        for rid in rids[1:]:
            pk = next(p for r, p in members if r == rid)
            db.upsert("repository_duplicates", {"repo_id": rid, "other_repo_id": rids[0], "kind": "duplicated_package",
                                                "similarity": 1.0, "detail": f"package {pk} identical (fingerprint {fp[:10]})",
                                                "canonical_repo_id": rids[0], "run_id": run_id},
                      ["repo_id", "other_repo_id", "kind"])
            n_pkg += 1
    # pairwise overlap via inverted index on hashes
    inverted = defaultdict(list)
    for rid, hs in hashes_by_repo.items():
        for h in hs:
            inverted[h].append(rid)
    shared = defaultdict(int)
    for h, rids in inverted.items():
        if 1 < len(rids) <= 50:
            for i, a in enumerate(rids):
                for b in rids[i + 1:]:
                    shared[(min(a, b), max(a, b))] += 1
    n_pairs = 0
    for (a, b), n in shared.items():
        smaller = min(len(hashes_by_repo[a]), len(hashes_by_repo[b]))
        containment = n / smaller if smaller else 0
        if containment >= threshold:
            later, canon = (a, b) if rank.get(a, 0) > rank.get(b, 0) else (b, a)
            union = len(hashes_by_repo[a] | hashes_by_repo[b])
            db.upsert("repository_duplicates", {
                "repo_id": later, "other_repo_id": canon, "kind": "substantial_code_overlap",
                "similarity": round(containment, 4),
                "detail": dumps({"shared_methods": n, "containment": round(containment, 4), "jaccard": round(n / union, 4)}),
                "canonical_repo_id": canon, "run_id": run_id}, ["repo_id", "other_repo_id", "kind"])
            n_pairs += 1
    db.commit()
    return {"duplicate_methods": len(updates), "duplicated_packages": n_pkg, "overlapping_repository_pairs": n_pairs}


def duplicate_repo_ids(db: Database, kinds=("fork", "exact_tree_duplicate", "substantial_code_overlap")) -> set[int]:
    q = "SELECT DISTINCT repo_id FROM repository_duplicates WHERE kind IN (%s)" % ",".join("?" for _ in kinds)
    return {r[0] for r in db.query(q, tuple(kinds))}
