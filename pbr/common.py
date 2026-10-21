"""Shared plumbing of every phase: configuration, paths, logging, atomic file writing, phase state.

Every phase writes into its own folder ``results/NN_name/`` and finishes by writing ``_phase.json``:

    {"phase": 6, "status": "done", "run_id": "...", "fingerprint": "...", "started": ..., "finished": ...,
     "duration_s": ..., "metrics": {...}, "outputs": [...], "figures": [...], "warnings": [...]}

The fingerprint is a hash of the configuration the phase reads, the profile sizes, the run_id of every
upstream phase and the content of any human input (returned workbooks). ``./run.sh`` skips a phase whose
fingerprint is unchanged and redoes it (and everything after it) otherwise.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------------------------- phases
# number, key (folder results/NN_key), title, config sections read, upstream phases, human inputs (folders
# relative to the phase folder whose files change the fingerprint)
_PHASES = [   # dependencies are written by name; numbers exist only here (renumbering = editing this table)
    (1, "repositories", "Repository selection (weighted quality score)", ["repositories", "corpus"], [], []),
    (2, "literature", "Literature ingestion", ["literature"], [], []),
    (3, "corpus", "Pharo corpus preparation and method filters", ["corpus", "method_filters"], ["repositories"], []),
    (4, "sampling", "Representative sampling", ["sampling"], ["literature", "corpus"], []),
    (5, "discovery", "Taxonomy discovery (3 independent LLMs)", ["llm", "discovery"], ["literature", "corpus", "sampling"], []),
    (6, "taxonomy", "Two final taxonomies (Pharo methods, benchmark tasks)", ["llm", "taxonomy"], ["discovery"],
     ["pharo/review/returned", "literature/review/returned"]),
    (7, "annotation", "Independent structured annotation", ["llm", "annotation"],
     ["literature", "corpus", "sampling", "taxonomy"], []),
    (8, "validation", "AI agreement and expert validation", ["validation"], ["taxonomy", "annotation"],
     ["expert/returned", "adjudication/returned"]),
    (9, "profile", "Large-scale Pharo behaviour analysis", ["llm", "annotation", "profile"],
     ["corpus", "sampling", "taxonomy", "annotation", "validation"], []),
    (10, "coverage", "Literature versus Pharo comparison", ["coverage"], ["annotation", "validation", "profile"], []),
    (11, "benchmark", "Pharo benchmark construction", ["benchmark"], ["corpus", "taxonomy", "annotation", "coverage"],
     ["selected_behaviors.csv"]),
    (12, "verification", "Benchmark validation (execution, independent oracle, mutants, alignment, audit)", ["llm", "pharo", "benchmark"],
     ["benchmark"], ["expert/returned"]),
    (13, "evaluation", "Model evaluation", ["pharo", "evaluation"], ["taxonomy", "verification"], []),
]
BY_KEY = {p[1]: p[0] for p in _PHASES}
# Stage 1 (run by ./run.sh): from the repositories to the two final taxonomies. Stage 2 (phases 7-13: validation on
# new samples, profile, coverage, benchmark, evaluation) still expects ONE shared taxonomy: to be adapted to the two.
STAGE1_LAST = 6
PHASES = [(n, k, t, sec, [BY_KEY[d] for d in deps], human) for n, k, t, sec, deps, human in _PHASES]
BY_NUM = {p[0]: p for p in PHASES}


def num_of(phase) -> int:
    """A phase given by number or by name (``"annotation"``) → its number."""
    return phase if isinstance(phase, int) else BY_KEY[phase]


def phase_dirname(phase) -> str:
    num = num_of(phase)
    return f"{num:02d}_{BY_NUM[num][1]}"


# ---------------------------------------------------------------------------------------------- config
def load_config(path: Path | None = None, profile: str | None = None) -> dict:
    path = Path(path or ROOT / "config.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    profile = profile or os.getenv("PROFILE") or "pilot"
    if profile not in cfg["profiles"]:
        raise SystemExit(f"Unknown PROFILE {profile!r}: choose one of {', '.join(cfg['profiles'])}")
    cfg["profile_name"] = profile
    cfg["sizes"] = dict(cfg["profiles"][profile])
    if cfg["sizes"].get("results_dir"):          # a profile with its own results folder (e.g. smoke)
        cfg["project"]["results_dir"] = cfg["sizes"].pop("results_dir")
    return cfg


def resolve(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p).resolve()


def results_dir(cfg: dict) -> Path:
    return resolve(cfg["project"]["results_dir"])


def data_dir(cfg: dict) -> Path:
    return resolve(cfg["project"]["data_dir"])


def show(path: Path) -> str:
    """A path for messages: relative to the project when inside it, absolute otherwise."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def stable_hash(obj, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:n]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------- writing
def _atomic(path: Path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write(tmp)
    tmp.replace(path)
    return path


def write_json(path: Path, obj) -> Path:
    return _atomic(path, lambda t: t.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
                                                encoding="utf-8"))


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> Path:
    return _atomic(path, lambda t: t.write_text(text, encoding="utf-8"))


def write_jsonl(path: Path, rows) -> Path:
    def w(t):
        with t.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return _atomic(path, w)


def append_jsonl(path: Path, row: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:          # a line cut by an interrupted run
                    continue
    return out


def write_parquet(path: Path, df) -> Path:
    return _atomic(path, lambda t: df.to_parquet(t, index=False))


def write_csv(path: Path, df) -> Path:
    return _atomic(path, lambda t: df.to_csv(t, index=False))


# ---------------------------------------------------------------------------------------------- logging
CONSOLE_FMT = "%(asctime)s  %(levelname)-7s %(message)s"
FILE_FMT = "%(asctime)s | %(phase)-14s | %(levelname)-7s | %(message)s"


class _PhaseFilter(logging.Filter):
    def __init__(self, phase: str):
        super().__init__()
        self.phase = phase

    def filter(self, record):
        record.phase = getattr(record, "phase", self.phase)
        return True


def get_logger(cfg: dict, phase_label: str, phase_file: str | None) -> logging.Logger:
    """A logger writing to the console (short), results/logs/pipeline.log (every phase) and
    results/logs/<phase>.log (this phase only)."""
    logs = results_dir(cfg) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"pbr.{phase_label}")
    log.handlers.clear()
    log.setLevel(logging.DEBUG)
    log.propagate = False
    flt = _PhaseFilter(phase_label)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(CONSOLE_FMT, "%H:%M:%S"))
    handlers = [console, logging.FileHandler(logs / "pipeline.log", encoding="utf-8")]
    if phase_file:
        handlers.append(logging.FileHandler(logs / phase_file, encoding="utf-8"))
    for h in handlers[1:]:
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter(FILE_FMT, "%Y-%m-%d %H:%M:%S"))
    for h in handlers:
        h.addFilter(flt)
        log.addHandler(h)
    return log


def event(cfg: dict, **fields):
    """Structured event log (results/logs/events.jsonl): one line per phase start / end / warning."""
    append_jsonl(results_dir(cfg) / "logs" / "events.jsonl", {"time": now_iso(), **fields})


# ---------------------------------------------------------------------------------------------- phase context
class PhaseWaiting(Exception):
    """The phase cannot go on without a human decision (the message says which file to fill)."""


@dataclass
class Phase:
    num: int
    cfg: dict
    log: logging.Logger
    metrics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    provisional: list = field(default_factory=list)   # reasons the result will change once humans answer

    @property
    def key(self) -> str:
        return BY_NUM[self.num][1]

    @property
    def title(self) -> str:
        return BY_NUM[self.num][2]

    @property
    def dir(self) -> Path:
        d = results_dir(self.cfg) / phase_dirname(self.num)
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def sizes(self) -> dict:
        return self.cfg["sizes"]

    @property
    def seed(self) -> int:
        return int(self.cfg["project"]["seed"])

    def path(self, *parts) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def figure(self, name: str) -> Path:
        return self.path("figures", name)

    def upstream(self, phase, *parts) -> Path:
        """A file of an earlier phase (given by name or number); the phase must have been run."""
        num = num_of(phase)
        d = results_dir(self.cfg) / phase_dirname(num)
        state = read_json(d / "_phase.json") or {}
        if state.get("status") not in ("done", "provisional"):
            raise SystemExit(f"Phase {num} ({BY_NUM[num][2]}) has not been run: ./run.sh {num}")
        return d.joinpath(*parts)

    def upstream_state(self, phase) -> dict:
        return read_json(results_dir(self.cfg) / phase_dirname(phase) / "_phase.json") or {}

    def warn(self, msg: str):
        self.warnings.append(msg)
        self.log.warning(msg)

    def mark_provisional(self, reason: str):
        """The result is complete but rests on input humans have not given yet (it changes once they do)."""
        self.provisional.append(reason)
        self.log.warning("PROVISIONAL: " + reason)

    def metric(self, key: str, value):
        self.metrics[key] = value
        return value

    def progress(self, items, label: str, total: int | None = None, every_s: float = 30.0):
        """Iterate and log progress lines (count, %, rate, ETA) at most every ``every_s`` seconds and at
        the end, so the logs stay readable (no progress bars in log files)."""
        total = total if total is not None else (len(items) if hasattr(items, "__len__") else None)
        start = last = time.time()
        i = 0
        for i, x in enumerate(items, 1):
            yield x
            t = time.time()
            if t - last >= every_s or (total and i == total):
                last = t
                rate = i / max(t - start, 1e-9)
                if total:
                    eta = (total - i) / rate if rate else 0
                    self.log.info(f"{label}: {i}/{total} ({100 * i / total:.0f}%), {rate * 60:.1f}/min, "
                                  f"ETA {_hms(eta)}")
                else:
                    self.log.info(f"{label}: {i} done, {rate * 60:.1f}/min")


def _hms(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


# ---------------------------------------------------------------------------------------------- fingerprint
def human_inputs_digest(cfg: dict, num: int) -> str:
    d = results_dir(cfg) / phase_dirname(num)
    parts = []
    for rel in BY_NUM[num][5]:
        p = d / rel
        files = sorted(p.rglob("*")) if p.is_dir() else ([p] if p.exists() else [])
        for f in files:
            if f.is_file() and not f.name.startswith((".", "~$")):
                parts.append((str(f.relative_to(d)), file_sha256(f)))
    return stable_hash(parts)


def fingerprint(cfg: dict, num: int) -> str:
    _, _, _, sections, ups, _ = BY_NUM[num]
    used = {s: cfg.get(s) for s in sections}
    used["sizes"] = cfg["sizes"]
    used["seed"] = cfg["project"]["seed"]
    upstream = {u: (read_json(results_dir(cfg) / phase_dirname(u) / "_phase.json") or {}).get("run_id") for u in ups}
    return stable_hash({"config": used, "upstream": upstream, "human": human_inputs_digest(cfg, num)})


def phase_state(cfg: dict, num: int) -> dict:
    return read_json(results_dir(cfg) / phase_dirname(num) / "_phase.json") or {}


def is_up_to_date(cfg: dict, num: int) -> bool:
    st = phase_state(cfg, num)
    return st.get("status") in ("done", "provisional") and st.get("fingerprint") == fingerprint(cfg, num)


def list_outputs(d: Path) -> tuple[list, list]:
    outs, figs = [], []
    for f in sorted(d.rglob("*")):
        if not f.is_file() or f.name.startswith((".", "_phase")) or "returned" in f.parts:
            continue
        rel = str(f.relative_to(d))
        (figs if f.suffix in (".png", ".pdf", ".svg") else outs).append(rel)
    return outs, figs


@contextmanager
def running(cfg: dict, num: int):
    """Context of one phase run: logger, lock state, final _phase.json, SUMMARY.md, events."""
    label = f"{num:02d} {BY_NUM[num][1]}"
    log = get_logger(cfg, label, f"{phase_dirname(num)}.log")
    ph = Phase(num, cfg, log)
    st_path = ph.dir / "_phase.json"
    started, t0 = now_iso(), time.time()
    fp = fingerprint(cfg, num)
    log.info("=" * 78)
    log.info(f"PHASE {num:02d} — {ph.title}   (profile: {cfg['profile_name']})")
    log.info(f"folder: {show(ph.dir)}")
    write_json(st_path, {"phase": num, "key": ph.key, "title": ph.title, "status": "running", "started": started,
                         "fingerprint": fp, "pid": os.getpid()})
    event(cfg, phase=num, event="start", profile=cfg["profile_name"])
    try:
        yield ph
    except PhaseWaiting as e:
        dur = time.time() - t0
        write_json(st_path, {"phase": num, "key": ph.key, "title": ph.title, "status": "waiting",
                             "started": started, "finished": now_iso(), "duration_s": round(dur, 1),
                             "fingerprint": fp, "waiting_for": str(e), "metrics": ph.metrics,
                             "warnings": ph.warnings})
        log.warning(f"WAITING FOR A HUMAN DECISION: {e}")
        event(cfg, phase=num, event="waiting", detail=str(e))
        write_summary(cfg)
        raise
    except BaseException as e:
        dur = time.time() - t0
        write_json(st_path, {"phase": num, "key": ph.key, "title": ph.title, "status": "failed", "started": started,
                             "finished": now_iso(), "duration_s": round(dur, 1), "fingerprint": fp,
                             "error": f"{type(e).__name__}: {e}", "metrics": ph.metrics, "warnings": ph.warnings})
        log.error(f"FAILED after {_hms(dur)}: {type(e).__name__}: {e}")
        if not isinstance(e, (KeyboardInterrupt, SystemExit)):
            log.debug("traceback", exc_info=True)
            import traceback
            with (results_dir(cfg) / "logs" / f"{phase_dirname(num)}.log").open("a", encoding="utf-8") as f:
                traceback.print_exc(file=f)
        event(cfg, phase=num, event="failed", error=f"{type(e).__name__}: {e}")
        write_summary(cfg)
        raise
    dur = time.time() - t0
    outs, figs = list_outputs(ph.dir)
    status = "provisional" if ph.provisional else "done"
    write_json(st_path, {"phase": num, "key": ph.key, "title": ph.title, "status": status,
                         "run_id": uuid.uuid4().hex[:12], "fingerprint": fp,
                         "profile": cfg["profile_name"], "started": started, "finished": now_iso(),
                         "duration_s": round(dur, 1), "metrics": ph.metrics, "warnings": ph.warnings,
                         "provisional": ph.provisional, "outputs": outs, "figures": figs})
    log.info(f"PHASE {num:02d} {status.upper()} in {_hms(dur)} — {len(outs)} files, {len(figs)} figures"
             + (f", {len(ph.warnings)} warning(s)" if ph.warnings else ""))
    for k, v in ph.metrics.items():
        if not isinstance(v, (dict, list)):
            log.info(f"   {k}: {v}")
    event(cfg, phase=num, event="done", status=status, duration_s=round(dur, 1), metrics=ph.metrics)
    write_summary(cfg)


# ---------------------------------------------------------------------------------------------- summary
STATUS_ICON = {"done": "✓", "provisional": "◐", "running": "…", "failed": "✗", "waiting": "⏸", None: "·"}


def write_summary(cfg: dict):
    """results/SUMMARY.md: one table of every phase, its key numbers, and links to its figures."""
    rd = results_dir(cfg)
    lines = ["# Pharo benchmark research — pipeline summary", "",
             f"Profile: `{cfg['profile_name']}` · updated {now_iso()}", "",
             "| | Phase | Status | Finished | Duration | Key numbers |", "|---|---|---|---|---|---|"]
    details = []
    for num, key, title, *_ in PHASES:
        st = phase_state(cfg, num)
        s = st.get("status")
        stale = s in ("done", "provisional") and st.get("fingerprint") != fingerprint(cfg, num)
        nums = ", ".join(f"{k}={v}" for k, v in (st.get("metrics") or {}).items()
                         if isinstance(v, (int, float, str)) and len(str(v)) < 30)[:160]
        lines.append(f"| {STATUS_ICON.get(s, '?')} | [{num:02d} {title}]({phase_dirname(num)}/) | "
                     f"{(s or 'not run') + (' (stale)' if stale else '')} | {(st.get('finished') or '')[:16]} | "
                     f"{_hms(st['duration_s']) if st.get('duration_s') is not None else ''} | {nums} |")
        if st.get("figures") or st.get("warnings") or st.get("provisional") or st.get("waiting_for") or st.get("error"):
            details += [f"## {num:02d} {title}", ""]
            for p in st.get("provisional") or []:
                details.append(f"- ◐ provisional: {p}")
            if st.get("waiting_for"):
                details.append(f"- ⏸ waiting: {st['waiting_for']}")
            if st.get("error"):
                details.append(f"- ✗ error: `{st['error']}` (log: `logs/{phase_dirname(num)}.log`)")
            for w in (st.get("warnings") or [])[:8]:
                details.append(f"- ⚠ {w}")
            for f in st.get("figures") or []:
                if f.endswith(".png"):
                    details.append(f"\n![{f}]({phase_dirname(num)}/{f})")
            details.append("")
    lines += ["", "✓ done · ◐ provisional (waits for human input that will change it) · ⏸ waiting for a human decision · "
              "✗ failed · · not run", "", "Logs: `logs/pipeline.log` (all phases), `logs/NN_<phase>.log`, "
              "`logs/events.jsonl`; `./run.sh logs [phase]` prints them.", ""] + details
    write_text(rd / "SUMMARY.md", "\n".join(lines) + "\n")


def deep_get(d: dict, dotted: str, default=None):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def clone(obj):
    return copy.deepcopy(obj)
