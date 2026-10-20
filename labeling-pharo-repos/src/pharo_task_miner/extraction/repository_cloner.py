"""Shallow, safe cloning of selected repositories at a recorded revision.

Safety: git hooks are disabled, submodules and LFS smudging are skipped, no
interactive prompts are allowed, and no repository code is ever executed.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

SAFE_GIT = ["git", "-c", "core.hooksPath=/dev/null", "-c", "protocol.file.allow=never",
            "-c", "submodule.recurse=false", "-c", "advice.detachedHead=false"]


def _env():
    env = dict(os.environ)
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1", "GIT_ASKPASS": "true"})
    return env


def _git(args, cwd=None, timeout=900):
    return subprocess.run(SAFE_GIT + args, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=_env())


def head_sha(path: Path) -> str | None:
    r = _git(["rev-parse", "HEAD"], cwd=path, timeout=60)
    return r.stdout.strip() if r.returncode == 0 else None


def is_dirty(path: Path) -> bool:
    r = _git(["status", "--porcelain", "--untracked-files=no"], cwd=path, timeout=120)
    return bool(r.stdout.strip())


def reuse(dest: Path, commit: str | None) -> dict:
    """Record an existing checkout as-is (never fetches, never deletes)."""
    if not (dest / ".git").exists():
        return {"status": "reused_without_git", "commit_sha": None, "error": None,
                "note": "existing directory is not a git checkout; revision unknown (not reproducible)"}
    sha = head_sha(dest)
    out = {"status": "reused", "commit_sha": sha, "error": None, "dirty": is_dirty(dest)}
    notes = []
    if commit and sha and sha != commit:
        notes.append(f"existing clone at {sha[:12]} differs from recorded head {commit[:12]}")
    if out["dirty"]:
        notes.append("working tree has local modifications")
    if notes:
        out["note"] = "; ".join(notes)
    return out


def clone(url: str, dest: Path, commit: str | None, depth: int = 1, timeout: int = 900) -> dict:
    """Clone ``url`` into ``dest`` and check out ``commit`` (if given).

    Returns {status, commit_sha, error}. An existing checkout is reused
    without fetching; its HEAD is recorded as-is (and flagged if it differs).
    """
    if dest.exists():
        return reuse(dest, commit)
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = ["clone", "--no-recurse-submodules", "--single-branch"]
    if depth and depth > 0:
        args += ["--depth", str(depth)]
    r = _git(args + [url, str(dest)], timeout=timeout)
    if r.returncode != 0:
        return {"status": "failed", "commit_sha": None, "error": r.stderr.strip()[-1000:]}
    sha = head_sha(dest)
    if commit and sha != commit:
        f = _git(["fetch", "--depth", str(max(depth, 1)), "origin", commit], cwd=dest, timeout=timeout)
        c = _git(["checkout", "--quiet", commit], cwd=dest, timeout=300) if f.returncode == 0 else f
        if c.returncode != 0:
            return {"status": "cloned_at_different_revision", "commit_sha": sha,
                    "error": f"could not check out recorded commit {commit}: {c.stderr.strip()[-500:]}"}
        sha = head_sha(dest)
    return {"status": "cloned", "commit_sha": sha, "error": None}
