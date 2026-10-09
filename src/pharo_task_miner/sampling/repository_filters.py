"""Module 3 — repository measurements, diversity attributes and eligibility.

Quality indicators are reported side by side; they are deliberately NOT
combined into a single quality score. Every exclusion is recorded with a reason.
"""
from __future__ import annotations

from datetime import datetime, timezone
import re

from ..storage.database import Database, loads

FRAMEWORK_RX = re.compile(r"\b(library|framework|binding|bindings|wrapper|api|toolkit|package|implementation of)\b", re.I)
APPLICATION_RX = re.compile(r"\b(application|app|tool|game|simulator|editor|browser|client|bot|website|demo)\b", re.I)
EDU_RX = re.compile(r"\b(tutorial|course|mooc|exercises?|kata|students?|teaching|lecture|homework|advent of code|book)\b", re.I)


def size_category(n_files: int | None, thresholds: dict) -> str:
    if not n_files:
        return "empty"
    for name in ("tiny", "small", "medium", "large"):
        if n_files < thresholds[name]:
            return name
    return "huge"


def repository_profiles(db: Database, cfg) -> list[dict]:
    rcfg, scfg = cfg["repositories"], cfg["sampling"]
    core = {o.lower() for o in rcfg.get("core_owners", [])}
    dom = {}
    for r in db.query("SELECT repo_id, label, score FROM domain_labels WHERE method='deterministic_rules'"
                      " ORDER BY repo_id, score DESC"):
        dom.setdefault(r["repo_id"], []).append(r["label"])
    dups = {}
    for r in db.query("SELECT repo_id, kind, canonical_repo_id FROM repository_duplicates"):
        dups.setdefault(r["repo_id"], []).append((r["kind"], r["canonical_repo_id"]))
    snaps = {r["repo_id"]: r for r in db.query(
        "SELECT * FROM repository_snapshots WHERE snapshot_id IN (SELECT MAX(snapshot_id) FROM repository_snapshots GROUP BY repo_id)")}
    now = datetime.now(timezone.utc)
    out = []
    for r in db.query("SELECT * FROM repositories ORDER BY repo_id"):
        tree = loads(r["tree_summary_json"], {}) or {}
        snap = snaps.get(r["repo_id"])
        n_files = tree.get("n_source_files")
        if snap is not None and snap["n_files"]:
            n_files = snap["n_files"]
        text = " ".join(filter(None, [r["description"], (r["readme_text"] or "")[:4000]]))
        labels = dom.get(r["repo_id"], [])
        kind = "framework_or_library" if FRAMEWORK_RX.search(r["description"] or "") else (
            "application" if APPLICATION_RX.search(r["description"] or "") else (
                "framework_or_library" if FRAMEWORK_RX.search(text) else (
                    "application" if APPLICATION_RX.search(text) else "unknown")))
        pushed = r["pushed_at"]
        created = r["created_at"]
        out.append({
            "repo_id": r["repo_id"], "full_name": r["full_name"], "owner": (r["owner"] or "").lower(),
            "pharo_status": r["pharo_status"], "archived": bool(r["archived"]), "is_fork": bool(r["is_fork"]),
            "stars": r["stars"] or 0, "forks": r["forks"] or 0, "contributors": r["contributors"],
            "commit_count": r["commit_count"], "license": r["license"],
            "n_source_files": n_files, "n_packages": tree.get("n_packages"),
            "n_classes": snap["n_classes"] if snap is not None else None,
            "n_methods": snap["n_methods"] if snap is not None else None,
            "lines_of_code": snap["n_lines"] if snap is not None else None,
            "has_tests": tree.get("has_tests"), "has_readme": bool(r["readme_text"]) or tree.get("has_readme"),
            "has_license": bool(r["license"] and r["license"] != "NOASSERTION") or tree.get("has_license_file"),
            "reproducible_revision": bool(r["head_commit_sha"] or (snap is not None and snap["commit_sha"])),
            "parseable": None if snap is None else (snap["n_methods"] or 0) > 0,
            "last_push_days": (now - datetime.fromisoformat(pushed.replace("Z", "+00:00"))).days if pushed else None,
            "age_days": (now - datetime.fromisoformat(created.replace("Z", "+00:00"))).days if created else None,
            "size_category": size_category(n_files, scfg["size_categories"]),
            "primary_domain": labels[0] if labels else "unknown", "domains": labels,
            "project_kind": kind,
            "educational": "education" in labels or bool(EDU_RX.search(r["description"] or "")),
            "core_infrastructure": (r["owner"] or "").lower() in core,
            "duplicates": dups.get(r["repo_id"], []),
        })
    return out


def eligibility(p: dict, cfg) -> tuple[bool, str]:
    rcfg = cfg["repositories"]
    accepted = set(rcfg.get("accepted_pharo_status", ["confirmed_pharo", "probable_pharo"]))
    if p["pharo_status"] is None:
        return False, "not verified (run `pharo-miner verify`)"
    if p["pharo_status"] not in accepted:
        if p["pharo_status"] == "uncertain":
            return False, "pharo_status=uncertain: queued for manual review"
        return False, f"pharo_status={p['pharo_status']}"
    if p["archived"] and not rcfg.get("include_archived", True):
        return False, "archived repository excluded by configuration"
    if p["is_fork"] and not rcfg.get("include_forks", False):
        return False, "fork excluded by configuration"
    for kind, canonical in p["duplicates"]:
        if kind in ("exact_tree_duplicate", "mirror_candidate", "substantial_code_overlap") and canonical and canonical != p["repo_id"]:
            return False, f"{kind} of repository {canonical}"
    if p["n_source_files"] == 0:
        return False, "no Smalltalk source files"
    return True, "eligible"
