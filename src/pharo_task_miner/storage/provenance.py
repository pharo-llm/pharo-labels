"""Run provenance: every stage records configuration, versions and timestamps."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import uuid

from .database import Database, dumps

TOOL_VERSION = "0.1.0"
_PKG_ROOT = Path(__file__).resolve().parents[1]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def code_hash() -> str:
    """Hash of all package source files, so results can be tied to the exact code."""
    h = hashlib.sha256()
    for p in sorted(_PKG_ROOT.rglob("*.py")):
        h.update(p.relative_to(_PKG_ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


class Run:
    def __init__(self, db: Database, stage: str, config, arguments: dict | None = None):
        self.db, self.stage, self.config = db, stage, config
        self.run_id = f"{stage}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
        self.counts: dict = {}
        self.notes: list[str] = []
        self.arguments = arguments or {}

    def start(self):
        self.db.execute(
            "INSERT INTO analysis_runs (run_id, stage, started_at, status, config_hash, config_json, tool_version, code_hash,"
            " python_version, platform, arguments_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (self.run_id, self.stage, now_iso(), "running", self.config.digest(), dumps(dict(self.config)), TOOL_VERSION,
             code_hash(), sys.version.split()[0], platform.platform(), dumps(self.arguments)))
        self.db.commit()

    def finish(self, status: str = "completed"):
        self.db.execute("UPDATE analysis_runs SET finished_at=?, status=?, counts_json=?, notes=? WHERE run_id=?",
                        (now_iso(), status, dumps(self.counts), "\n".join(self.notes) or None, self.run_id))
        self.db.commit()

    def manifest(self) -> dict:
        return {"run_id": self.run_id, "stage": self.stage, "tool_version": TOOL_VERSION, "code_hash": code_hash(),
                "config_hash": self.config.digest(), "python": sys.version.split()[0], "platform": platform.platform(),
                "counts": self.counts, "notes": self.notes, "generated_at": now_iso()}


@contextmanager
def tracked_run(db: Database, stage: str, config, arguments: dict | None = None):
    run = Run(db, stage, config, arguments)
    run.start()
    try:
        yield run
    except BaseException as e:
        run.notes.append(f"failed: {type(e).__name__}: {e}")
        run.finish("failed")
        raise
    run.finish("completed")


def write_json(path: Path, data, indent: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=False, default=str), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path):
    if not Path(path).exists():
        return []
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out
