"""Module 1 — multi-strategy GitHub repository discovery.

Strategies (all recorded per repository, with the query that found it):

A. ``topics``        repository search ``topic:<t>``, partitioned by creation date
B. ``languages``     repository search ``language:<l>``, partitioned by creation date
C. ``organizations`` complete listing of ``/orgs/<org>/repos``
D. ``indicators``    repository/README searches and code searches for Pharo
                     indicators (BaselineOf, Metacello, Tonel, .smalltalk.ston).
                     These are discovery *signals*, never proof (see verify).
E. ``curated``       researcher-provided CSV/JSON lists of repositories

Outputs: ``data/raw/repositories.jsonl`` and ``data/raw/search_manifest.json``.
The manifest records each query/window, its total_count, whether GitHub
reported ``incomplete_results``, and whether the 1000-result window cap
truncated it. Completed tasks are skipped on resume.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

from .github_client import GitHubClient, GitHubError
from ..storage.provenance import now_iso, read_jsonl, write_json, write_jsonl

SEARCH_CAP = 1000


class DiscoveryStore:
    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.repos_path = self.raw_dir / "repositories.jsonl"
        self.manifest_path = self.raw_dir / "search_manifest.json"
        self.repos: dict[str, dict] = {}
        for rec in read_jsonl(self.repos_path):
            self.repos[str(rec["repository_id"])] = rec
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        else:
            self.manifest = {"created_at": now_iso(), "tasks": {}}

    def add(self, raw: dict, strategy: str, query: str, window: str | None = None, signal: bool = False):
        if not raw or "id" not in raw:
            return
        key = str(raw["id"])
        rec = self.repos.get(key)
        hit = {"strategy": strategy, "query": query, "window": window, "signal_only": signal,
               "discovered_at": now_iso()}
        if rec is None:
            rec = {"repository_id": raw["id"], "full_name": raw.get("full_name"), "discovery": [hit],
                   "discovery_date": hit["discovered_at"], "raw": raw}
            self.repos[key] = rec
        else:
            if not any(h["strategy"] == strategy and h["query"] == query for h in rec["discovery"]):
                rec["discovery"].append(hit)
            # keep the richest raw object (full /repos responses have more fields than search items)
            if len(json.dumps(raw)) > len(json.dumps(rec["raw"])):
                rec["raw"] = raw
            rec["full_name"] = raw.get("full_name") or rec.get("full_name")

    def task_done(self, task_id: str) -> bool:
        return self.manifest["tasks"].get(task_id, {}).get("status") == "completed"

    def record_task(self, task_id: str, info: dict):
        self.manifest["tasks"][task_id] = {**self.manifest["tasks"].get(task_id, {}), **info, "updated_at": now_iso()}

    def save(self):
        self.manifest["updated_at"] = now_iso()
        self.manifest["unique_repositories"] = len(self.repos)
        tasks = self.manifest["tasks"].values()
        self.manifest["incomplete_tasks"] = sum(1 for t in tasks if t.get("incomplete_results") or t.get("truncated"))
        self.manifest["failed_tasks"] = sum(1 for t in tasks if t.get("status") == "failed")
        write_jsonl(self.repos_path, sorted(self.repos.values(), key=lambda r: r["repository_id"]))
        write_json(self.manifest_path, self.manifest)


def _windows(start: date, end: date, days: int):
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=days - 1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def search_partitioned(client: GitHubClient, store: DiscoveryStore, strategy: str, base_query: str,
                       start: date, end: date, window_days: int, log=print):
    """Search ``base_query created:a..b`` splitting windows whose count exceeds the cap."""
    stack = list(reversed(list(_windows(start, end, window_days))))
    while stack:
        a, b = stack.pop()
        window = f"{a.isoformat()}..{b.isoformat()}"
        task_id = f"{strategy}|{base_query}|{window}"
        if store.task_done(task_id):
            continue
        q = f"{base_query} created:{window}"
        try:
            first = client.get("/search/repositories", {"q": q, "per_page": 100, "page": 1})
        except GitHubError as e:
            store.record_task(task_id, {"strategy": strategy, "query": q, "status": "failed", "error": str(e)})
            log(f"  FAILED {q}: {e}")
            continue
        total = (first.data or {}).get("total_count", 0)
        if total > SEARCH_CAP and (b - a).days >= 1:
            mid = a + timedelta(days=(b - a).days // 2)
            store.record_task(task_id, {"strategy": strategy, "query": q, "status": "split", "total_count": total})
            stack.append((mid + timedelta(days=1), b))
            stack.append((a, mid))
            continue
        incomplete = bool((first.data or {}).get("incomplete_results"))
        items = list((first.data or {}).get("items", []))
        pages = 1
        if len(items) == 100:
            for page in range(2, SEARCH_CAP // 100 + 1):
                try:
                    resp = client.get("/search/repositories", {"q": q, "per_page": 100, "page": page})
                except GitHubError as e:
                    store.record_task(task_id, {"page_error": str(e)})
                    incomplete = True
                    break
                pages += 1
                batch = (resp.data or {}).get("items", [])
                incomplete = incomplete or bool((resp.data or {}).get("incomplete_results"))
                items.extend(batch)
                if len(batch) < 100:
                    break
        for it in items:
            store.add(it, strategy, base_query, window)
        store.record_task(task_id, {"strategy": strategy, "query": q, "status": "completed", "total_count": total,
                                    "retrieved": len(items), "pages": pages, "incomplete_results": incomplete,
                                    "truncated": total > len(items)})
        log(f"  {q}: {len(items)}/{total}{' (incomplete)' if incomplete or total > len(items) else ''}")
        store.save()


def discover_organizations(client, store, orgs, log=print):
    for org in orgs:
        task_id = f"organizations|{org}"
        if store.task_done(task_id):
            continue
        n = 0
        try:
            for resp, items in client.paginate(f"/orgs/{org}/repos", {"type": "public"}):
                if resp.data is None:  # not an org: try as a user account
                    for _r2, items2 in client.paginate(f"/users/{org}/repos"):
                        for it in items2:
                            store.add(it, "organizations", f"owner:{org}")
                            n += 1
                    break
                for it in items:
                    store.add(it, "organizations", f"org:{org}")
                    n += 1
            store.record_task(task_id, {"strategy": "organizations", "query": org, "status": "completed", "retrieved": n})
        except GitHubError as e:
            store.record_task(task_id, {"strategy": "organizations", "query": org, "status": "failed", "error": str(e)})
        log(f"  org {org}: {n} repositories")
        store.save()


def discover_code_indicators(client, store, queries, log=print):
    """Code search finds repositories containing indicator files. Not date-partitionable."""
    for q in queries:
        task_id = f"indicators-code|{q}"
        if store.task_done(task_id):
            continue
        seen, total, incomplete, n = set(), 0, False, 0
        try:
            for resp, items in client.paginate("/search/code", {"q": q}, max_pages=10, item_key="items"):
                total = (resp.data or {}).get("total_count", total)
                incomplete = incomplete or bool((resp.data or {}).get("incomplete_results"))
                for it in items:
                    repo = it.get("repository") or {}
                    full = repo.get("full_name")
                    if full and full not in seen:
                        seen.add(full)
                        full_resp = client.get(f"/repos/{full}")
                        store.add(full_resp.data or repo, "indicators", f"code:{q}", signal=True)
                        n += 1
            store.record_task(task_id, {"strategy": "indicators", "query": q, "kind": "code_search", "status": "completed",
                                        "total_count": total, "unique_repositories": n, "incomplete_results": incomplete,
                                        "truncated": total > SEARCH_CAP})
        except GitHubError as e:
            store.record_task(task_id, {"strategy": "indicators", "query": q, "status": "failed", "error": str(e)})
        log(f"  code search {q!r}: {n} repositories (total hits {total})")
        store.save()


def _read_curated(path: Path) -> list[dict]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("repositories", [])
        return [r if isinstance(r, dict) else {"full_name": r} for r in rows]
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _full_name(row: dict) -> str | None:
    v = (row.get("full_name") or row.get("repository") or row.get("url") or "").strip()
    if "github.com/" in v:
        v = v.split("github.com/", 1)[1]
    v = v.strip("/").removesuffix(".git")
    return v if v.count("/") == 1 else None


def discover_curated(client, store, paths: list[Path], log=print):
    for path in paths:
        for row in _read_curated(Path(path)):
            full = _full_name(row)
            if not full:
                continue
            task_id = f"curated|{path.name}|{full}"
            if store.task_done(task_id):
                continue
            try:
                resp = client.get(f"/repos/{full}")
                if resp.data:
                    store.add(resp.data, "curated", f"{path.name}:{full}")
                    store.record_task(task_id, {"strategy": "curated", "query": full, "status": "completed",
                                                "note": row.get("note") or row.get("reason")})
                else:
                    store.record_task(task_id, {"strategy": "curated", "query": full, "status": "failed", "error": "not found"})
            except GitHubError as e:
                store.record_task(task_id, {"strategy": "curated", "query": full, "status": "failed", "error": str(e)})
        log(f"  curated list {path}: done")
        store.save()


def import_search_dump(store: DiscoveryStore, path: Path, label: str | None = None) -> int:
    """Import a previously saved list of GitHub repository objects (e.g. the
    prototype's ``output/pharo_raw.json``) as a discovery source."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("items", [])
    for it in items:
        store.add(it, "imported_dump", label or f"file:{Path(path).name}")
    store.record_task(f"imported_dump|{path}", {"strategy": "imported_dump", "query": str(path), "status": "completed",
                                                "retrieved": len(items),
                                                "note": "Imported raw search results; original queries not re-run"})
    store.save()
    return len(items)


def run_discovery(cfg, client: GitHubClient | None, store: DiscoveryStore, strategies: list[str] | None = None,
                  log=print) -> dict:
    d = cfg["github"]["discovery"]
    strategies = strategies or d["strategies"]
    start = datetime.fromisoformat(d.get("start_date") or "2008-01-01").date()
    end = datetime.fromisoformat(d["end_date"]).date() if d.get("end_date") else datetime.now(timezone.utc).date()
    wdays = int(d.get("initial_window_days", 365))
    if client is None:
        raise RuntimeError("A GitHub client is required for live discovery")
    if "topics" in strategies:
        for t in d.get("topics", []):
            log(f"[A] topic:{t}")
            search_partitioned(client, store, "topics", f"topic:{t}", start, end, wdays, log)
    if "languages" in strategies:
        for lang in d.get("languages", []):
            log(f"[B] language:{lang}")
            search_partitioned(client, store, "languages", f"language:{lang}", start, end, wdays, log)
    if "organizations" in strategies:
        log("[C] organizations")
        discover_organizations(client, store, d.get("organizations", []), log)
    if "indicators" in strategies:
        for q in d.get("indicator_repository_queries", []):
            log(f"[D] {q}")
            search_partitioned(client, store, "indicators", q, start, end, wdays, log)
        if client.token:
            discover_code_indicators(client, store, d.get("indicator_code_queries", []), log)
        else:
            store.record_task("indicators-code|*", {"strategy": "indicators", "status": "skipped",
                                                    "error": "code search requires an authenticated token"})
    if "curated" in strategies and d.get("curated_lists"):
        log("[E] curated lists")
        discover_curated(client, store, [cfg.resolve(p) for p in d["curated_lists"]], log)
    store.manifest["client_stats"] = client.stats
    store.save()
    return {"unique_repositories": len(store.repos), **client.stats}
