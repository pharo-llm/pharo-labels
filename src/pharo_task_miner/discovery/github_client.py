"""GitHub REST client with on-disk caching, rate-limit handling and retries.

Every response body is cached verbatim under ``github.cache_dir`` keyed by the
request, which (a) preserves raw API responses for auditing, (b) makes re-runs
reproducible and (c) lets interrupted collection resume without re-fetching.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

import requests


class GitHubError(RuntimeError):
    def __init__(self, status: int, url: str, message: str):
        super().__init__(f"GitHub {status} for {url}: {message}")
        self.status = status
        self.url = url


class Response:
    def __init__(self, status: int, data, headers: dict, url: str, from_cache: bool, fetched_at: str):
        self.status, self.data, self.headers, self.url = status, data, headers, url
        self.from_cache, self.fetched_at = from_cache, fetched_at

    def last_page(self) -> int | None:
        m = re.search(r'[?&]page=(\d+)[^>]*>;\s*rel="last"', self.headers.get("Link", ""))
        return int(m.group(1)) if m else None


_KEEP_HEADERS = ("Link", "X-RateLimit-Remaining", "X-RateLimit-Reset", "X-RateLimit-Resource", "ETag", "Date")


class GitHubClient:
    def __init__(self, cfg: dict, cache_dir: Path, token: str | None = None, offline: bool = False,
                 refresh: bool = False, log=print):
        self.api = cfg.get("api_url", "https://api.github.com").rstrip("/")
        self.token = token if token is not None else os.getenv(cfg.get("token_env", "GITHUB_TOKEN"))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ttl = cfg.get("cache_ttl_hours")
        self.cache_ttl = None if ttl in (None, "", 0) else float(ttl) * 3600
        self.max_retries = int(cfg.get("max_retries", 6))
        self.backoff = float(cfg.get("backoff_seconds", 2.0))
        self.search_interval = float(cfg.get("search_min_interval_seconds", 2.2))
        self.timeout = int(cfg.get("timeout_seconds", 45))
        self.offline, self.refresh, self.log = offline, refresh, log
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                                     "User-Agent": "PharoTaskMiner-research"})
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"
        self._last_search = 0.0
        self.stats = {"requests": 0, "cache_hits": 0, "retries": 0, "rate_limit_waits": 0}

    # -- caching ---------------------------------------------------------
    def _key(self, url: str, params: dict | None, accept: str | None) -> Path:
        raw = json.dumps([url, sorted((params or {}).items()), accept or ""], sort_keys=True, default=str)
        h = hashlib.sha256(raw.encode()).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.json"

    def _read_cache(self, path: Path):
        if not path.exists():
            return None
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if self.cache_ttl is not None:
            age = time.time() - datetime.fromisoformat(rec["fetched_at"]).timestamp()
            if age > self.cache_ttl:
                return None
        return rec

    # -- requests --------------------------------------------------------
    def get(self, path: str, params: dict | None = None, accept: str | None = None, allow_404: bool = True,
            use_cache: bool = True) -> Response:
        url = path if path.startswith("http") else self.api + path
        cpath = self._key(url, params, accept)
        if use_cache and not self.refresh:
            rec = self._read_cache(cpath)
            if rec is not None:
                self.stats["cache_hits"] += 1
                return Response(rec["status"], rec["data"], rec.get("headers", {}), url, True, rec["fetched_at"])
        if self.offline:
            raise GitHubError(0, url, "offline mode and no cached response")
        is_search = "/search/" in url
        headers = {"Accept": accept} if accept else {}
        attempt = 0
        while True:
            attempt += 1
            if is_search:
                wait = self.search_interval - (time.time() - self._last_search)
                if wait > 0:
                    time.sleep(wait)
            try:
                self.stats["requests"] += 1
                r = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt > self.max_retries:
                    raise GitHubError(-1, url, f"connection failed: {e}")
                self._sleep_backoff(attempt, f"connection error {type(e).__name__}")
                continue
            finally:
                if is_search:
                    self._last_search = time.time()
            if r.status_code in (403, 429) and self._rate_limited(r):
                self._wait_for_rate_limit(r)
                continue
            if r.status_code >= 500 or r.status_code == 202:  # 202: GitHub is still computing the answer
                if attempt > self.max_retries:
                    raise GitHubError(r.status_code, url, r.text[:300])
                self._sleep_backoff(attempt, f"HTTP {r.status_code}")
                continue
            break
        fetched_at = datetime.now(timezone.utc).isoformat()
        if r.status_code in (404, 409, 451) and allow_404:
            data = None
        elif r.status_code >= 400:
            raise GitHubError(r.status_code, url, r.text[:300])
        elif accept and ".raw" in accept:
            data = r.text  # raw file contents; GitHub still labels the media type "...raw+json"
        else:
            try:
                data = r.json()
            except ValueError:
                data = r.text
        keep = {k: r.headers[k] for k in _KEEP_HEADERS if k in r.headers}
        rec = {"url": url, "params": params, "accept": accept, "status": r.status_code, "headers": keep,
               "data": data, "fetched_at": fetched_at}
        cpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = cpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        tmp.replace(cpath)
        return Response(r.status_code, data, keep, url, False, fetched_at)

    def _rate_limited(self, r) -> bool:
        if r.status_code == 429:
            return True
        if r.headers.get("X-RateLimit-Remaining") == "0" or "Retry-After" in r.headers:
            return True
        return "rate limit" in r.text.lower()

    def _wait_for_rate_limit(self, r):
        self.stats["rate_limit_waits"] += 1
        retry_after = r.headers.get("Retry-After")
        if retry_after:
            wait = int(retry_after) + 1
        else:
            reset = int(r.headers.get("X-RateLimit-Reset", int(time.time()) + 60))
            wait = max(5, reset - int(time.time()) + 2)
        self.log(f"  rate limited; waiting {wait}s")
        time.sleep(wait)

    def _sleep_backoff(self, attempt: int, why: str):
        self.stats["retries"] += 1
        delay = min(120.0, self.backoff * (2 ** (attempt - 1))) + random.uniform(0, 1)
        self.log(f"  {why}; retry {attempt}/{self.max_retries} in {delay:.1f}s")
        time.sleep(delay)

    def paginate(self, path: str, params: dict | None = None, max_pages: int = 100, item_key: str | None = None):
        params = dict(params or {})
        params.setdefault("per_page", 100)
        for page in range(1, max_pages + 1):
            params["page"] = page
            resp = self.get(path, dict(params))
            data = resp.data or ([] if not item_key else {})
            items = data.get(item_key, []) if item_key else data
            yield resp, items
            if len(items) < params["per_page"]:
                break

    def count_via_link(self, path: str, params: dict | None = None) -> int | None:
        """Count list items (commits, contributors) with per_page=1 and the Link header."""
        p = dict(params or {})
        p["per_page"] = 1
        resp = self.get(path, p)
        if resp.status == 409:  # empty repository
            return 0
        if resp.data is None:
            return None
        last = resp.last_page()
        return last if last is not None else len(resp.data or [])
