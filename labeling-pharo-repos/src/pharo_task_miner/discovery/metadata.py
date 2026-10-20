"""Loading discovered repositories into the database, Pharo verification and
metadata collection (Modules 1–2). Raw API responses are preserved both in
the request cache and in ``data/raw/metadata/<repository_id>.json``."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

from .github_client import GitHubClient, GitHubError
from .pharo_detection import (local_listing, local_readme_and_ci, remote_ci, remote_listing, remote_readme,
                              summarize_listing, verify)
from ..storage.database import Database, dumps, loads
from ..storage.provenance import now_iso, read_jsonl, write_json


def clone_dir_name(full_name: str) -> str:
    return full_name.replace("/", "__")


def find_local_clone(cfg, full_name: str) -> Path | None:
    dirs = [cfg.resolve(cfg["repositories"]["clone_dir"])] + [cfg.resolve(d) for d in cfg["repositories"].get("existing_clone_dirs", [])]
    for d in dirs:
        p = d / clone_dir_name(full_name)
        if p.is_dir():
            return p
    return None


def row_from_raw(rec: dict) -> dict:
    raw = rec.get("raw") or {}
    owner = (raw.get("owner") or {}).get("login") or (rec.get("full_name") or "/").split("/")[0]
    lic = raw.get("license") or {}
    return {
        "repo_id": int(rec["repository_id"]), "full_name": raw.get("full_name") or rec.get("full_name"), "owner": owner,
        "name": raw.get("name"), "url": raw.get("html_url"), "description": raw.get("description"),
        "topics_json": dumps(raw.get("topics") or []), "language": raw.get("language"),
        "stars": raw.get("stargazers_count"), "forks": raw.get("forks_count"),
        "created_at": raw.get("created_at"), "updated_at": raw.get("updated_at"), "pushed_at": raw.get("pushed_at"),
        "archived": int(bool(raw.get("archived"))), "is_fork": int(bool(raw.get("fork"))),
        "parent_full_name": (raw.get("parent") or {}).get("full_name"),
        "license": lic.get("spdx_id") or lic.get("key"), "default_branch": raw.get("default_branch"),
        "size_kb": raw.get("size"), "discovery_json": dumps(rec.get("discovery")), "discovery_date": rec.get("discovery_date"),
    }


def load_discovered(db: Database, raw_dir: Path) -> int:
    n = 0
    for rec in read_jsonl(raw_dir / "repositories.jsonl"):
        row = row_from_raw(rec)
        existing = db.query("SELECT repo_id FROM repositories WHERE repo_id=?", (row["repo_id"],))
        if existing:
            # never overwrite richer metadata collected later with search-level fields
            db.execute("UPDATE repositories SET discovery_json=?, discovery_date=COALESCE(discovery_date, ?) WHERE repo_id=?",
                       (row["discovery_json"], row["discovery_date"], row["repo_id"]))
        else:
            row["metadata_source"] = "search_item"
            db.upsert("repositories", row, ["repo_id"])
        n += 1
    db.commit()
    return n


class _Progress:
    def __init__(self, total: int, label: str, log, every: int = 50):
        self.total, self.label, self.log, self.every = total, label, log, every
        self.start, self.done = time.time(), 0

    def step(self):
        self.done += 1
        if self.done % self.every == 0 or self.done == self.total:
            el = time.time() - self.start
            eta = el / self.done * (self.total - self.done)
            self.log(f"  {self.label} {self.done}/{self.total} — elapsed {el / 60:.0f} min, remaining ~{eta / 60:.0f} min")


def _verify_one(cfg, client, r, local, prefer_local: bool, max_ci: int):
    """Verify one repository with as few API requests as possible:
    1 request (tree) settles repositories without Smalltalk sources; the README
    is fetched only if present; CI/project files only when still undecided."""
    topics, desc, lang = loads(r["topics_json"], []), r["description"], r["language"]
    if client is not None and not (prefer_local and local):
        listing = remote_listing(client, r["full_name"], r["default_branch"])
        if listing is not None and not summarize_listing(listing)["n_source_files"]:
            return verify(listing, None, {}, topics, desc, lang), None, None, {}
        readme, readme_path = remote_readme(client, r["full_name"], listing)
        v = verify(listing, readme, {}, topics, desc, lang)
        ci = {}
        if v.status != "confirmed_pharo" and listing is not None:
            ci = remote_ci(client, r["full_name"], listing, max_ci)
            if ci:
                v = verify(listing, readme, ci, topics, desc, lang)
        return v, readme, readme_path, ci
    if local is not None:
        listing = local_listing(local)
        readme, ci = local_readme_and_ci(local, max_ci)
        return verify(listing, readme, ci, topics, desc, lang), readme, None, ci
    return verify(None, r["readme_text"], {}, topics, desc, lang), None, None, {}


def verify_repositories(cfg, db: Database, client: GitHubClient | None, only: list[str] | None = None,
                        prefer_local: bool = False, refresh: bool = False, log=print) -> dict:
    """Resumable: repositories already verified from a file listing are skipped
    unless ``refresh``. Errors on one repository never stop the stage."""
    counts: dict = {}
    max_ci = int(cfg["verification"].get("max_ci_files", 5))
    rows = db.query("SELECT * FROM repositories ORDER BY repo_id")
    if only:
        rows = [r for r in rows if r["full_name"] in only]
    todo = [r for r in rows if refresh or r["pharo_status"] is None or r["verification_source"] in (None, "metadata_only", "error")]
    counts["already_verified"] = len(rows) - len(todo)
    if client is not None and todo:
        log(f"  {len(todo)} repositories to verify (~1–3 API requests each; GitHub allows 5000/hour). "
            "Safe to interrupt: rerun to resume.")
    prog = _Progress(len(todo), "verified", log)
    for r in todo:
        local = find_local_clone(cfg, r["full_name"])
        try:
            v, readme, readme_path, ci = _verify_one(cfg, client, r, local, prefer_local, max_ci)
        except Exception as e:  # noqa: BLE001 — one bad repository must not stop a multi-hour stage
            counts["errors"] = counts.get("errors", 0) + 1
            log(f"  {r['full_name']}: {type(e).__name__}: {e}")
            db.execute("UPDATE repositories SET pharo_status='uncertain', verification_source='error',"
                       " pharo_counter_evidence_json=? WHERE repo_id=?",
                       (dumps([{"kind": "verification_error", "strength": "info", "detail": f"{type(e).__name__}: {e}"}]),
                        r["repo_id"]))
            prog.step()
            continue
        db.execute(
            "UPDATE repositories SET pharo_status=?, pharo_evidence_json=?, pharo_counter_evidence_json=?, verification_source=?,"
            " tree_summary_json=?, readme_text=COALESCE(?, readme_text), readme_path=COALESCE(?, readme_path) WHERE repo_id=?",
            (v.status, dumps(v.evidence), dumps(v.counter_evidence), v.source,
             dumps({**v.summary, "pharo_versions": v.pharo_versions, "ci_files": sorted(ci)}),
             readme, readme_path, r["repo_id"]))
        counts[v.status] = counts.get(v.status, 0) + 1
        prog.step()
        if prog.done % 50 == 0:
            db.commit()
    db.commit()
    return counts


def collect_metadata(cfg, db: Database, client: GitHubClient, raw_dir: Path, only: list[str] | None = None,
                     refresh: bool = False, log=print) -> dict:
    """Full metadata for Pharo candidates only (statuses in ``metadata.statuses``).
    3 requests per repository (+1 for forks, to find the parent). Resumable."""
    meta_dir = raw_dir / "metadata"
    statuses = set(cfg["metadata"]["statuses"])
    counts = {"collected": 0, "failed": 0}
    rows = [r for r in db.query("SELECT * FROM repositories ORDER BY repo_id") if r["pharo_status"] in statuses]
    if only:
        rows = [r for r in rows if r["full_name"] in only]
    todo = [r for r in rows if refresh or r["metadata_source"] != "github_api"]
    counts["already_collected"] = len(rows) - len(todo)
    log(f"  metadata for {len(todo)} Pharo candidates ({', '.join(sorted(statuses))})")
    prog = _Progress(len(todo), "metadata", log)
    for r in todo:
        full = r["full_name"]
        try:
            data = client.get(f"/repos/{full}").data if r["is_fork"] else None
            languages = client.get(f"/repos/{full}/languages").data or {}
            n_contrib = client.count_via_link(f"/repos/{full}/contributors", {"anon": "true"})
            commits = client.get(f"/repos/{full}/commits", {"per_page": 1, **({"sha": r["default_branch"]} if r["default_branch"] else {})})
            last = commits.last_page()
            commit_count = last if last is not None else len(commits.data or [])
            head = (commits.data or [{}])[0].get("sha") if isinstance(commits.data, list) and commits.data else None
        except Exception as e:  # noqa: BLE001
            counts["failed"] += 1
            log(f"  {full}: {type(e).__name__}: {e}")
            prog.step()
            continue
        write_json(meta_dir / f"{r['repo_id']}.json", {
            "repository_full_record": data, "languages": languages, "contributors_count": n_contrib,
            "commit_count": commit_count, "head_commit_sha": head, "collected_at": now_iso()})
        upd = {"languages_json": dumps(languages), "contributors": n_contrib, "commit_count": commit_count,
               "head_commit_sha": head or r["head_commit_sha"], "metadata_source": "github_api",
               "metadata_collected_at": now_iso()}
        if data:
            upd["parent_full_name"] = (data.get("parent") or {}).get("full_name")
        db.execute("UPDATE repositories SET " + ", ".join(f"{k}=?" for k in upd) + " WHERE repo_id=?",
                   (*upd.values(), r["repo_id"]))
        counts["collected"] += 1
        prog.step()
        if prog.done % 25 == 0:
            db.commit()
    db.commit()
    return counts


def repository_age_days(created_at: str | None) -> int | None:
    if not created_at:
        return None
    return (datetime.now(timezone.utc) - datetime.fromisoformat(created_at.replace("Z", "+00:00"))).days
