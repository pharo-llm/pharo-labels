"""Snapshots of a run's results on the Hugging Face Hub (a dataset repository).

A snapshot is everything needed to continue without mining again: the
evidence database, the parsed methods, the GitHub discovery data, the
annotation sample, human and LLM annotations, tables, reports and logs.

Layout in the dataset repository::

    README.md                       dataset card (an auto-generated section is kept up to date)
    snapshot/manifest.json          what is inside: files, sha256, sizes, counts, provenance
    snapshot/database/…sqlite       consistent copy of the evidence database
    snapshot/data/…, validation/…, output/…   result files, same relative layout as locally
    snapshot/archives/*.tar.gz      directories with many small files (deterministic archives)

Guarantees:

* **Consistent.** The database is copied with SQLite's online backup API.
* **Verifiable.** Every file has a sha256 in the manifest; a pull checks all of them
  before anything is installed.
* **Portable.** Absolute clone paths are stored relative to the project root and
  restored for the machine that pulls.
* **Privacy.** E-mail addresses in raw GitHub API responses (commit authors) are
  replaced by ``<redacted>`` before upload. Clones of third-party repositories are
  excluded unless ``snapshot.include_clones`` is set.
* **Versioned.** Every push is one commit, tagged ``snapshot-YYYYMMDD-HHMMSS``; any
  tag or commit can be pulled. Pushing unchanged results uploads nothing.
* **Safe.** A pull never deletes local results: an existing results folder is
  renamed to ``<name>.before-pull-<timestamp>`` first.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import sys
import tarfile

from .storage.provenance import TOOL_VERSION, code_hash, now_iso

FORMAT_VERSION = 1
REMOTE_DIR = "snapshot"
CARD_START = "<!-- pharo-miner:snapshot:start -->"
CARD_END = "<!-- pharo-miner:snapshot:end -->"
ORIGIN_FILE = ".snapshot_origin.json"
_EMAIL_FIELD = re.compile(rb'("email"\s*:\s*)"[^"]*"')


class SnapshotError(RuntimeError):
    pass


def log(msg: str):
    print(msg, flush=True)


# ============================================================================ local layout
@dataclass
class Roots:
    """Logical snapshot roots -> local directories (from the configuration)."""
    project: Path
    data: Path
    validation: Path
    output: Path
    database: Path
    clones: Path

    @classmethod
    def from_cfg(cls, cfg) -> "Roots":
        return cls(project=Path(cfg.root).resolve(), data=cfg.data_dir, validation=cfg.validation_dir,
                   output=cfg.output_dir, database=cfg.database_path,
                   clones=cfg.resolve(cfg["repositories"]["clone_dir"]))


def local_state(cfg) -> str:
    """``complete`` (mining finished: database + annotation sample), ``partial`` (a run
    started but did not reach the sample) or ``none``."""
    r = Roots.from_cfg(cfg)
    has_db = r.database.exists() and r.database.stat().st_size > 0
    has_sample = (r.validation / "annotation_sample.csv").exists()
    if has_db and has_sample:
        return "complete"
    if has_db or (r.data / "raw" / "repositories.jsonl").exists():
        return "partial"
    return "none"


def _files_under(base: Path, pattern: str = "**/*") -> list[Path]:
    if not base.exists():
        return []
    return sorted(p for p in base.glob(pattern) if p.is_file() and p.name != ".DS_Store")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================ building
class Builder:
    def __init__(self, cfg, staging: Path, include_api_cache: bool, include_clones: bool):
        self.cfg = cfg
        self.r = Roots.from_cfg(cfg)
        self.staging = staging
        self.include_api_cache = include_api_cache
        self.include_clones = include_clones
        self.entries: list[dict] = []
        fp = staging / ".fingerprints.json"
        self.old_fp = json.loads(fp.read_text()) if fp.exists() else {}
        self.new_fp: dict = {}

    # -- plain files ---------------------------------------------------------------------------
    def add_file(self, src: Path, rel: str, root: str, local_rel: str):
        dst = self.staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        st = src.stat()
        if not (dst.exists() and dst.stat().st_size == st.st_size and dst.stat().st_mtime == st.st_mtime):
            shutil.copy2(src, dst)
        self.entries.append({"path": rel, "kind": "file", "root": root, "target": local_rel})

    def add_tree(self, base: Path, root: str, prefix: str, pattern: str = "**/*", exclude=lambda p: False):
        """Every file under `base`, uploaded under `prefix/` and restored under the same path below `root`."""
        base_rel = base.relative_to(self._root_dir(root)).as_posix()
        for p in _files_under(base, pattern):
            if exclude(p):
                continue
            rel = p.relative_to(base).as_posix()
            self.add_file(p, f"{prefix}/{rel}", root, rel if base_rel == "." else f"{base_rel}/{rel}")

    def _root_dir(self, root: str) -> Path:
        return {"data": self.r.data, "validation": self.r.validation, "output": self.r.output,
                "clones": self.r.clones, "database": self.r.database.parent}[root]

    # -- archives ------------------------------------------------------------------------------
    def add_archive(self, rel: str, root: str, base: Path, files: list[Path], scrub_emails: bool = False):
        """Deterministic .tar.gz of `files` (paths relative to `base`, extracted under root/base)."""
        if not files:
            return
        fingerprint = hashlib.sha256()
        for p in files:
            st = p.stat()
            fingerprint.update(f"{p.relative_to(base).as_posix()}|{st.st_size}|{int(st.st_mtime)}|{scrub_emails}\n".encode())
        fp = fingerprint.hexdigest()
        dst = self.staging / rel
        self.new_fp[rel] = fp
        if not (dst.exists() and self.old_fp.get(rel) == fp):
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".partial")
            with tmp.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=6) as gz, \
                    tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for p in files:
                    data = p.read_bytes()
                    if scrub_emails and b'"email"' in data:
                        data = _EMAIL_FIELD.sub(rb'\1"<redacted>"', data)
                    ti = tarfile.TarInfo(p.relative_to(base).as_posix())
                    ti.size, ti.mtime, ti.mode = len(data), int(p.stat().st_mtime), 0o644
                    ti.uid = ti.gid = 0
                    ti.uname = ti.gname = ""
                    tar.addfile(ti, io.BytesIO(data))
            tmp.replace(dst)
        target = base.relative_to(self._root_dir(root)).as_posix()
        self.entries.append({"path": rel, "kind": "archive", "root": root, "target": "" if target == "." else target,
                             "members": len(files), "emails_redacted": scrub_emails})

    # -- database ------------------------------------------------------------------------------
    def add_database(self):
        src = self.r.database
        if not src.exists():
            raise SnapshotError(f"No database at {src}: nothing to snapshot (run ./1-mining.sh first).")
        rel = f"database/{src.name}"
        dst = self.staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".partial")
        tmp.unlink(missing_ok=True)
        with sqlite3.connect(src) as s, sqlite3.connect(tmp) as d:
            s.backup(d)  # consistent even while WAL files exist
        with sqlite3.connect(tmp) as d:
            d.execute("PRAGMA journal_mode=DELETE")
            prefix = str(self.r.project) + os.sep
            d.execute("UPDATE repository_snapshots SET local_path = substr(local_path, ?) WHERE local_path LIKE ?",
                      (len(prefix) + 1, prefix + "%"))
            d.commit()
            d.execute("VACUUM")
        # replace only when the content changed, so an unchanged database is not re-uploaded
        if dst.exists() and _sha256(dst) == _sha256(tmp):
            tmp.unlink()
        else:
            tmp.replace(dst)
        self.entries.append({"path": rel, "kind": "database", "root": "database", "target": src.name,
                             "note": "repository_snapshots.local_path is relative to the project root"})

    # -- everything ----------------------------------------------------------------------------
    def build(self) -> dict:
        r = self.r
        self.add_database()
        proc = r.data / "processed"
        self.add_tree(proc, "data", "data/processed",
                      exclude=lambda p: p.name.startswith(r.database.name) or p.suffix in (".partial",))
        for name in ("repositories.jsonl", "search_manifest.json"):
            if (r.data / "raw" / name).exists():
                self.add_file(r.data / "raw" / name, f"data/raw/{name}", "data", f"raw/{name}")
        self.add_tree(r.data / "parsed", "data", "data/parsed")
        meta = r.data / "raw" / "metadata"
        self.add_archive("archives/raw_metadata.tar.gz", "data", meta, _files_under(meta), scrub_emails=True)
        if self.include_api_cache:
            cache = self.cfg.resolve(self.cfg["github"]["cache_dir"])
            groups: dict[str, list[Path]] = {}
            for p in _files_under(cache):
                key = p.relative_to(cache).parts[0][:1] if len(p.relative_to(cache).parts) > 1 else "_"
                groups.setdefault(key, []).append(p)
            for key, files in sorted(groups.items()):
                self.add_archive(f"archives/api_cache_{key}.tar.gz", "data", cache, files, scrub_emails=True)
        self.add_tree(r.validation, "validation", "validation")
        for sub in ("tables", "reports", "figures"):
            self.add_tree(r.output / sub, "output", f"output/{sub}")
        for p in sorted(r.output.glob("*.log")):
            self.add_file(p, f"output/{p.name}", "output", p.name)
        if self.include_clones and r.clones.exists():
            for d in sorted(x for x in r.clones.iterdir() if x.is_dir()):
                self.add_archive(f"archives/repositories/{d.name}.tar.gz", "clones", r.clones, _files_under(d))
        # remove stale staged files
        keep = {e["path"] for e in self.entries} | {"manifest.json", ".fingerprints.json"}
        for p in _files_under(self.staging):
            if p.relative_to(self.staging).as_posix() not in keep:
                p.unlink()
        for e in self.entries:
            p = self.staging / e["path"]
            e["size"], e["sha256"] = p.stat().st_size, _sha256(p)
        (self.staging / ".fingerprints.json").write_text(json.dumps(self.new_fp, indent=1))
        return self.manifest()

    def manifest(self) -> dict:
        db = self.staging / self.entries[0]["path"]
        summary = database_summary(db)
        origin = self.r.output / ORIGIN_FILE
        man = {
            "format_version": FORMAT_VERSION, "created_at": now_iso(), "tool_version": TOOL_VERSION,
            "code_hash": code_hash(), "python": sys.version.split()[0], "platform": platform.platform(),
            "config_digest": self.cfg.digest(), "derived_from": json.loads(origin.read_text()) if origin.exists() else None,
            "contents": {"api_cache": self.include_api_cache, "clones": self.include_clones,
                         "emails_redacted_in_raw_github_responses": True},
            "summary": summary, "total_bytes": sum(e["size"] for e in self.entries),
            "files": sorted(self.entries, key=lambda e: e["path"]),
        }
        for name in ("config.yaml", "models.yaml"):
            p = self.r.project / name
            if p.exists():
                man.setdefault("project_files", {})[name] = p.read_text(encoding="utf-8")
        (self.staging / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
        return man


def database_summary(db: Path) -> dict:
    out = {}
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        def q(sql):
            try:
                return c.execute(sql).fetchall()
            except sqlite3.Error:
                return []
        out["repositories_by_status"] = dict(q("SELECT COALESCE(pharo_status,'not_verified'), COUNT(*) FROM repositories GROUP BY 1"))
        out["analysed_repositories"] = (q("SELECT COUNT(DISTINCT repo_id) FROM repository_snapshots WHERE n_methods > 0") or [[0]])[0][0]
        out["methods"] = (q("SELECT COUNT(*) FROM methods") or [[0]])[0][0]
        out["labels_by_source"] = dict(q("SELECT label_source, COUNT(*) FROM method_labels GROUP BY 1"))
        out["annotation_decisions_by_kind"] = dict(q("SELECT kind, COUNT(*) FROM annotation_decisions GROUP BY 1"))
        out["stages_completed"] = sorted({r[0] for r in q("SELECT stage FROM analysis_runs WHERE status='completed'")})
    return out


# ============================================================================ Hugging Face
def _api(token: str | None = None):
    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover - declared dependency
        raise SnapshotError("huggingface_hub is missing: .venv/bin/pip install huggingface_hub") from e
    return HfApi(token=token or os.getenv("HF_TOKEN") or None)


def remote_manifest(repo: str, revision: str = "main", token: str | None = None) -> dict | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
    try:
        p = hf_hub_download(repo, f"{REMOTE_DIR}/manifest.json", repo_type="dataset", revision=revision,
                            token=token or os.getenv("HF_TOKEN") or None)
    except (EntryNotFoundError, RevisionNotFoundError):
        return None
    except RepositoryNotFoundError as e:
        raise SnapshotError(f"Dataset {repo} not found (or private and HF_TOKEN missing/without access).") from e
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _card(existing: str, man: dict, repo: str, tag: str) -> str:
    s = man["summary"]
    status = s.get("repositories_by_status", {})
    labels = s.get("labels_by_source", {})
    files = man["files"]
    section = f"""{CARD_START}
## PharoTaskMiner results snapshot

Latest snapshot: **`{tag}`** ({man['created_at']}), tool version {man['tool_version']} (code `{man['code_hash']}`).

| | |
|---|---|
| Repositories discovered | {sum(status.values())} ({', '.join(f'{k}: {v}' for k, v in sorted(status.items()))}) |
| Repositories analysed | {s.get('analysed_repositories')} |
| Methods parsed | {s.get('methods')} |
| Operation labels | {', '.join(f'{k}: {v}' for k, v in sorted(labels.items())) or '–'} |
| Files / size | {len(files)} files, {man['total_bytes'] / 1e9:.2f} GB |

**Use it.** With the PharoTaskMiner code, `./1-mining.sh` downloads this snapshot automatically
when there are no local results (instead of mining GitHub again). Manually:
`pharo-miner snapshot pull` (latest) or `pharo-miner snapshot pull --revision {tag}`.

**Contents.** `snapshot/manifest.json` lists every file with its sha256. `snapshot/database/` is the
SQLite evidence database (repositories, methods, message sends, labels, annotations, run provenance);
`snapshot/data/` parsed methods and discovery data; `snapshot/validation/` the annotation sample, human
and LLM annotations (`validation/llm/`: every model answer, prompts, agreement); `snapshot/output/`
tables, reports, figures and logs; `snapshot/archives/` raw GitHub API responses and repository metadata.

**Notes.** Method source code and README text come from public GitHub repositories and remain under
their authors' licenses. E-mail addresses in raw GitHub API responses were replaced by `<redacted>`.
Operation labels with source `llm_adjudicated` were produced by language models and are **not**
human-validated; `rule_based` labels are automatic.
{CARD_END}"""
    if CARD_START in existing and CARD_END in existing:
        before, rest = existing.split(CARD_START, 1)
        after = rest.split(CARD_END, 1)[1]
        return before + section + after
    if not existing.strip():
        existing = "---\nlicense: other\n---\n"
    if existing.startswith("---"):
        end = existing.find("\n---", 3)
        if end != -1:
            head, body = existing[:end + 4], existing[end + 4:]
            if "pretty_name:" not in head:
                head = head[:end] + "\npretty_name: PharoTaskMiner results" + head[end:]
            return head + "\n\n" + section + "\n" + body.lstrip("\n")
    return section + "\n\n" + existing


def push(cfg, repo: str, include_api_cache: bool = True, include_clones: bool = False, message: str | None = None,
         token: str | None = None, dry_run: bool = False) -> dict:
    token = token or os.getenv("HF_TOKEN")
    if not token and not dry_run:
        raise SnapshotError("HF_TOKEN is not set (a Hugging Face token with write access to the dataset).")
    r = Roots.from_cfg(cfg)
    staging = r.output / ".snapshot_staging"
    log(f"Preparing snapshot in {staging} …")
    man = Builder(cfg, staging, include_api_cache, include_clones).build()
    log(f"  {len(man['files'])} files, {man['total_bytes'] / 1e9:.2f} GB "
        f"(api cache: {'yes' if include_api_cache else 'no'}, clones: {'yes' if include_clones else 'no'})")
    if dry_run:
        return {"status": "dry_run", "staging": str(staging), "files": len(man["files"]), "bytes": man["total_bytes"],
                "summary": man["summary"]}
    api = _api(token)
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        info = api.dataset_info(repo)
    except RepositoryNotFoundError:
        log(f"  creating private dataset {repo}")
        api.create_repo(repo, repo_type="dataset", private=True)
        info = api.dataset_info(repo)
    old = remote_manifest(repo, "main", token)
    if old and {(f["path"], f["sha256"]) for f in old["files"]} == {(f["path"], f["sha256"]) for f in man["files"]}:
        log("  the dataset already holds exactly these files: nothing to upload.")
        return {"status": "up_to_date", "repo": repo, "commit": info.sha}
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    tag = f"snapshot-{ts}"
    s = man["summary"]
    msg = message or (f"Snapshot {ts}: {s.get('analysed_repositories')} repositories, {s.get('methods')} methods, "
                      f"labels {s.get('labels_by_source')}")
    log(f"  uploading to https://huggingface.co/datasets/{repo} (unchanged files are skipped) …")
    commit = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(staging), path_in_repo=REMOTE_DIR,
                               commit_message=msg, delete_patterns=["**"], ignore_patterns=[".fingerprints.json", "*.partial"])
    from huggingface_hub import hf_hub_download
    try:
        existing = Path(hf_hub_download(repo, "README.md", repo_type="dataset", token=token)).read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001 - no card yet
        existing = ""
    card = _card(existing, man, repo, tag)
    if card != existing:
        commit = api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                                 commit_message=f"Dataset card: {tag}")
    sha = getattr(commit, "oid", None) or api.dataset_info(repo).sha
    api.create_tag(repo, tag=tag, repo_type="dataset", revision=sha, tag_message=msg)
    return {"status": "pushed", "repo": repo, "tag": tag, "commit": sha, "files": len(man["files"]),
            "bytes": man["total_bytes"], "url": f"https://huggingface.co/datasets/{repo}/tree/{tag}"}


def pull(cfg, repo: str, revision: str = "main", force: bool = False, token: str | None = None) -> dict:
    from huggingface_hub import snapshot_download
    r = Roots.from_cfg(cfg)
    man = remote_manifest(repo, revision, token)
    if man is None:
        raise SnapshotError(f"No snapshot in {repo} at revision {revision!r}.")
    if man.get("format_version", 0) > FORMAT_VERSION:
        raise SnapshotError(f"Snapshot format {man['format_version']} is newer than this tool understands ({FORMAT_VERSION}).")
    state = local_state(cfg)
    if state != "none" and not force:
        raise SnapshotError(f"Local results already exist ({state}) in {r.output}. Use --force to replace them "
                            "(they are kept as a backup folder, not deleted).")
    download = r.project / ".snapshot_download"
    log(f"Downloading snapshot {repo}@{revision} ({man['total_bytes'] / 1e9:.2f} GB, {len(man['files'])} files) …")
    path = Path(snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=str(download),
                                  allow_patterns=[f"{REMOTE_DIR}/*", f"{REMOTE_DIR}/**"],
                                  token=token or os.getenv("HF_TOKEN") or None)) / REMOTE_DIR
    log("  verifying checksums …")
    bad = [f["path"] for f in man["files"] if not (path / f["path"]).exists() or _sha256(path / f["path"]) != f["sha256"]]
    if bad:
        raise SnapshotError(f"{len(bad)} file(s) missing or corrupted after download, e.g. {bad[:3]}; nothing was installed. "
                            "Run the pull again.")
    backup = None
    if state != "none":
        backup = r.output.with_name(f"{r.output.name}.before-pull-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        r.output.rename(backup)
        log(f"  previous results moved to {backup}")
    for d in (r.data, r.validation, r.output):
        d.mkdir(parents=True, exist_ok=True)
    dirs = {"data": r.data, "validation": r.validation, "output": r.output, "clones": r.clones, "database": r.database.parent}
    for f in man["files"]:
        src = path / f["path"]
        base = dirs[f["root"]]
        if f["kind"] == "database":
            r.database.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, r.database)
            with sqlite3.connect(r.database) as c:
                prefix = str(r.project) + os.sep
                c.execute("UPDATE repository_snapshots SET local_path = ? || local_path "
                          "WHERE local_path IS NOT NULL AND local_path NOT LIKE '/%'", (prefix,))
                c.commit()
        elif f["kind"] == "file":
            dst = base / f["target"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            dst = base / f["target"] if f["target"] else base
            dst.mkdir(parents=True, exist_ok=True)
            with tarfile.open(src, "r:gz") as tar:
                tar.extractall(dst, filter="data")
    origin = {"repo": repo, "revision": revision, "snapshot_created_at": man["created_at"], "pulled_at": now_iso(),
              "summary": man["summary"]}
    (r.output / ORIGIN_FILE).write_text(json.dumps(origin, indent=1), encoding="utf-8")
    shutil.rmtree(download, ignore_errors=True)
    return {"status": "pulled", "repo": repo, "revision": revision, "files": len(man["files"]),
            "summary": man["summary"], "backup_of_previous_results": str(backup) if backup else None}


def info(cfg, repo: str, revision: str = "main") -> dict:
    man = remote_manifest(repo, revision)
    tags = []
    try:
        tags = sorted((t.name for t in _api().list_repo_refs(repo, repo_type="dataset").tags), reverse=True)
    except Exception:  # noqa: BLE001 - listing refs is informative only
        pass
    return {"repo": repo, "revision": revision, "local_state": local_state(cfg), "remote_snapshot": man is not None,
            "remote_created_at": man and man["created_at"], "remote_summary": man and man["summary"],
            "remote_bytes": man and man["total_bytes"], "tags": tags[:20]}
