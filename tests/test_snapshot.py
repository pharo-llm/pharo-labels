"""Results snapshots: building, privacy redaction, determinism, and a push→pull
round trip against a fake Hugging Face Hub (no network)."""
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharo_task_miner import snapshot as snap  # noqa: E402
from pharo_task_miner.config import load_config  # noqa: E402
from pharo_task_miner.storage.database import Database  # noqa: E402


def make_project(root: Path, with_sample: bool = True):
    """A small but complete results folder, laid out like a real run."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text("snapshot: {repo: test/ds}\n")
    cfg = load_config(root / "config.yaml")
    with Database(cfg.database_path) as db:
        db.execute("INSERT INTO repositories (repo_id, full_name, pharo_status) VALUES (1, 'a/b', 'confirmed_pharo')")
        db.execute("INSERT INTO repository_snapshots (repo_id, commit_sha, local_path, n_methods) VALUES (1, 'abc', ?, 3)",
                   (str(cfg.resolve("results/data/repositories/a__b")),))
        db.commit()
    d = cfg.data_dir
    (d / "processed" / "statistics.json").write_text('{"x": 1}')
    (d / "raw").mkdir(parents=True, exist_ok=True)
    (d / "raw" / "repositories.jsonl").write_text('{"full_name": "a/b"}\n')
    (d / "raw" / "metadata").mkdir()
    (d / "raw" / "metadata" / "1.json").write_text('{"id": 1}')
    for shard in ("0a", "f3"):
        (d / "raw" / "api_cache" / shard).mkdir(parents=True)
    (d / "raw" / "api_cache" / "0a" / "x.json").write_text(
        '{"data": {"commit": {"author": {"name": "Ada", "email": "ada@example.org"}}}}')
    (d / "raw" / "api_cache" / "f3" / "y.json").write_text('{"data": [1, 2]}')
    (d / "parsed").mkdir()
    (d / "parsed" / "a__b.jsonl.gz").write_bytes(b"\x1f\x8bfake")
    (d / "repositories" / "a__b" / "src").mkdir(parents=True)
    (d / "repositories" / "a__b" / "src" / "A.class.st").write_text("Class { #name : #A }")
    lab = cfg.labelling_dir
    (lab / "rounds").mkdir(parents=True)
    (lab / "rounds" / "round_000.json").write_text('{"round": 0, "method_ids": [1]}')
    (lab / "labels.jsonl").write_text('{"method_id": 1, "raw_task": "Accessor"}\n')
    if with_sample:
        mark_analysed(cfg)
    (cfg.output_dir / "tables").mkdir(parents=True)
    (cfg.output_dir / "tables" / "t.csv").write_text("a\n1\n")
    (cfg.output_dir / "run.log").write_text("log\n")
    return cfg


def mark_analysed(cfg):
    with Database(cfg.database_path) as db:
        db.execute("INSERT INTO analysis_runs (run_id, stage, started_at, status) VALUES "
                   "('a1', 'analyze', '2026-10-10T00:00:00+00:00', 'completed')")
        db.commit()


class FakeHub:
    """Stands in for the Hub: 'uploads' are copies of the staging folder."""

    def __init__(self, store: Path):
        self.store = store

    def publish(self, staging: Path):
        shutil.rmtree(self.store, ignore_errors=True)
        shutil.copytree(staging, self.store / snap.REMOTE_DIR, ignore=shutil.ignore_patterns(".fingerprints.json"))

    def remote_manifest(self, repo, revision="main", token=None):
        p = self.store / snap.REMOTE_DIR / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else None

    def snapshot_download(self, repo, repo_type, revision, local_dir, allow_patterns, token):
        shutil.copytree(self.store, local_dir, dirs_exist_ok=True)
        return local_dir


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, cfg, **kw):
        staging = cfg.output_dir / ".snapshot_staging"
        man = snap.Builder(cfg, staging, kw.get("api_cache", True), kw.get("clones", False)).build()
        return staging, man

    def test_states(self):
        empty = load_config(root=self.tmp)
        self.assertEqual(snap.local_state(empty), "none")
        from pharo_task_miner.storage.database import Database
        Database(empty.database_path).close()   # e.g. `label` run before any mining
        self.assertEqual(snap.local_state(empty), "none")
        cfg = make_project(self.tmp / "p", with_sample=False)
        self.assertEqual(snap.local_state(cfg), "partial")
        mark_analysed(cfg)
        self.assertEqual(snap.local_state(cfg), "complete")

    def test_build_contents_privacy_and_portability(self):
        cfg = make_project(self.tmp / "p")
        staging, man = self.build(cfg)
        paths = {f["path"] for f in man["files"]}
        for expected in ("database/pharo_task_miner.sqlite", "data/processed/statistics.json", "data/raw/repositories.jsonl",
                         "data/parsed/a__b.jsonl.gz", "labelling/labels.jsonl", "labelling/rounds/round_000.json",
                         "output/tables/t.csv", "output/run.log", "archives/raw_metadata.tar.gz",
                         "archives/api_cache_0.tar.gz", "archives/api_cache_f.tar.gz"):
            self.assertIn(expected, paths)
        self.assertFalse(any("repositories/" in p and p.startswith("archives/") for p in paths))  # clones excluded
        self.assertFalse(any(".snapshot_staging" in p for p in paths))
        # e-mail redacted, name kept
        with tarfile.open(staging / "archives/api_cache_0.tar.gz") as t:
            body = t.extractfile("0a/x.json").read().decode()
        self.assertIn('"<redacted>"', body)
        self.assertNotIn("ada@example.org", body)
        self.assertIn("Ada", body)
        # absolute clone paths become relative to the project
        with sqlite3.connect(staging / "database/pharo_task_miner.sqlite") as c:
            self.assertEqual(c.execute("SELECT local_path FROM repository_snapshots").fetchone()[0],
                             "results/data/repositories/a__b")
        # the live database is untouched
        with sqlite3.connect(cfg.database_path) as c:
            self.assertTrue(c.execute("SELECT local_path FROM repository_snapshots").fetchone()[0].startswith("/"))
        self.assertEqual(man["summary"]["repositories_by_status"], {"confirmed_pharo": 1})
        self.assertTrue(man["contents"]["emails_redacted_in_raw_github_responses"])
        # every file is listed with its checksum
        for f in man["files"]:
            self.assertEqual(snap._sha256(staging / f["path"]), f["sha256"])

    def test_rebuild_is_deterministic(self):
        cfg = make_project(self.tmp / "p")
        _, m1 = self.build(cfg)
        shutil.rmtree(cfg.output_dir / ".snapshot_staging")
        _, m2 = self.build(cfg)
        self.assertEqual({(f["path"], f["sha256"]) for f in m1["files"]}, {(f["path"], f["sha256"]) for f in m2["files"]})

    def test_options(self):
        cfg = make_project(self.tmp / "p")
        _, man = self.build(cfg, api_cache=False, clones=True)
        paths = {f["path"] for f in man["files"]}
        self.assertFalse(any(p.startswith("archives/api_cache") for p in paths))
        self.assertIn("archives/repositories/a__b.tar.gz", paths)

    def test_push_pull_round_trip(self):
        src = make_project(self.tmp / "src")
        staging, man = self.build(src, clones=True)
        hub = FakeHub(self.tmp / "hub")
        hub.publish(staging)
        dst_root = self.tmp / "elsewhere"
        dst_root.mkdir()
        (dst_root / "config.yaml").write_text("snapshot: {repo: test/ds}\n")
        dst = load_config(dst_root / "config.yaml")
        with mock.patch.object(snap, "remote_manifest", hub.remote_manifest), \
                mock.patch("huggingface_hub.snapshot_download", hub.snapshot_download):
            res = snap.pull(dst, "test/ds")
            self.assertEqual(res["status"], "pulled")
            self.assertEqual(snap.local_state(dst), "complete")
            for rel in ("results/data/processed/statistics.json", "results/data/parsed/a__b.jsonl.gz",
                        "results/labelling/labels.jsonl", "results/tables/t.csv", "results/run.log",
                        "results/data/raw/metadata/1.json", "results/data/raw/api_cache/f3/y.json",
                        "results/data/repositories/a__b/src/A.class.st"):
                self.assertEqual((dst_root / rel).read_bytes(), (src.root / rel).read_bytes(), rel)
            redacted = (dst_root / "results/data/raw/api_cache/0a/x.json").read_text()
            self.assertIn("<redacted>", redacted)
            with sqlite3.connect(dst.database_path) as c:
                self.assertEqual(c.execute("SELECT local_path FROM repository_snapshots").fetchone()[0],
                                 str(dst_root.resolve() / "results/data/repositories/a__b"))
            origin = json.loads((dst.output_dir / snap.ORIGIN_FILE).read_text())
            self.assertEqual(origin["repo"], "test/ds")
            self.assertFalse((dst_root / ".snapshot_download").exists())
            # existing results are never overwritten silently ...
            with self.assertRaises(snap.SnapshotError):
                snap.pull(dst, "test/ds")
            # ... and --force keeps them as a backup
            (dst.output_dir / "mine.txt").write_text("keep me")
            res = snap.pull(dst, "test/ds", force=True)
            backup = Path(res["backup_of_previous_results"])
            self.assertEqual((backup / "mine.txt").read_text(), "keep me")
            self.assertFalse((dst.output_dir / "mine.txt").exists())

    def test_corrupted_download_installs_nothing(self):
        src = make_project(self.tmp / "src")
        staging, _ = self.build(src)
        hub = FakeHub(self.tmp / "hub")
        hub.publish(staging)
        (self.tmp / "hub" / snap.REMOTE_DIR / "output" / "run.log").write_text("tampered")
        dst_root = self.tmp / "elsewhere"
        dst_root.mkdir()
        dst = load_config(root=dst_root)
        with mock.patch.object(snap, "remote_manifest", hub.remote_manifest), \
                mock.patch("huggingface_hub.snapshot_download", hub.snapshot_download):
            with self.assertRaises(snap.SnapshotError):
                snap.pull(dst, "test/ds")
        self.assertEqual(snap.local_state(dst), "none")

    def test_card_keeps_existing_content(self):
        man = {"summary": {"repositories_by_status": {"confirmed_pharo": 2}, "analysed_repositories": 1, "methods": 5,
                           "labels_by_source": {"rule_based": 9}}, "files": [], "total_bytes": 0,
               "created_at": "t", "tool_version": "0.1.0", "code_hash": "c"}
        card = snap._card("---\nlicense: mit\n---\n\nMy own text.\n", man, "o/d", "snapshot-1")
        self.assertTrue(card.startswith("---\nlicense: mit"))
        self.assertIn("pretty_name:", card.split("---")[1])
        self.assertIn("My own text.", card)
        again = snap._card(card, {**man, "summary": {**man["summary"], "methods": 7}}, "o/d", "snapshot-2")
        self.assertEqual(again.count(snap.CARD_START), 1)
        self.assertIn("snapshot-2", again)
        self.assertNotIn("snapshot-1", again)
        self.assertIn("My own text.", again)


if __name__ == "__main__":
    unittest.main()
