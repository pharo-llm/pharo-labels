"""End-to-end test of the pharo-miner CLI on a synthetic offline corpus."""
import csv
import json
import warnings
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharo_task_miner.cli import main  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

CLASS_A = '''Class {
	#name : #{name},
	#superclass : #Object,
	#instVars : [ 'items' ],
	#category : #'{pkg}'
}

{ #category : #accessing }
{name} >> items [
	^ items ifNil: [ items := OrderedCollection new ]
]

{ #category : #operations }
{name} >> positives [
	^ self items select: [ :x | x > 0 ]
]

{ #category : #operations }
{name} >> total [
	^ self items inject: 0 into: [ :a :b | a + b ]
]

{ #category : #printing }
{name} >> printOn: aStream [
	aStream nextPutAll: 'Bag of '; print: self items size
]

{ #category : #operations }
{name} >> toJson [
	^ NeoJSONWriter toString: self items
]

{ #category : #'instance creation' }
{name} class >> withAll: aCollection [
	^ self new addAll: aCollection; yourself
]
'''

BASELINE = '''Class {
	#name : #BaselineOf{proj},
	#superclass : #BaselineOf,
	#category : #BaselineOf{proj}
}

{ #category : #baselines }
BaselineOf{proj} >> baseline: spec [
	<baseline>
	spec for: #common do: [
		spec baseline: 'NeoJSON' with: [ spec repository: 'github://svenvc/NeoJSON/repository' ].
		spec package: '{proj}' with: [ spec requires: #('NeoJSON') ] ]
]
'''


def make_repo(root: Path, owner: str, name: str, description: str):
    d = root / f"{owner}__{name}"
    (d / "src" / name).mkdir(parents=True)
    (d / "src" / f"BaselineOf{name}").mkdir(parents=True)
    (d / "src" / name / "package.st").write_text(f"Package {{ #name : #{name} }}\n")
    for cls in ("Alpha", "Beta"):
        (d / "src" / name / f"{name}{cls}.class.st").write_text(
            CLASS_A.replace("{name}", f"{name}{cls}").replace("{pkg}", name))
    (d / "src" / f"BaselineOf{name}" / f"BaselineOf{name}.class.st").write_text(BASELINE.replace("{proj}", name))
    (d / ".project").write_text("{ 'srcDirectory' : 'src' }\n")
    (d / "README.md").write_text(f"# {name}\n\n{description}. Load it in Pharo 12 with Metacello.\n")
    for args in (["init", "-q"], ["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(["git", *args], cwd=d, check=True, capture_output=True)


class PipelineTest(unittest.TestCase):
    def test_end_to_end(self):
        warnings.simplefilter("ignore", ResourceWarning)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            clones = tmp / "clones"
            repos = [("acme", "WebShop", "A web application framework for HTTP REST services", ["seaside"]),
                     ("labx", "ChartKit", "A visualization library for charts with Roassal", ["visualization"]),
                     ("solo", "Kata", "Exercises and tutorial for students learning Pharo", ["education"])]
            dump = []
            for i, (owner, name, desc, topics) in enumerate(repos, 1):
                make_repo(clones, owner, name, desc)
                dump.append({"id": 1000 + i, "full_name": f"{owner}/{name}", "name": name, "owner": {"login": owner},
                             "html_url": f"https://github.com/{owner}/{name}", "description": desc, "topics": topics,
                             "language": "Smalltalk", "stargazers_count": i, "forks_count": 0, "fork": False, "archived": False,
                             "created_at": "2020-01-01T00:00:00Z", "pushed_at": "2025-01-01T00:00:00Z", "default_branch": "main"})
            (tmp / "dump.json").write_text(json.dumps(dump))
            cfg = {
                "project": {"data_dir": str(tmp / "data"), "output_dir": str(tmp / "out"), "validation_dir": str(tmp / "val"),
                            "database": str(tmp / "data" / "db.sqlite")},
                "repositories": {"existing_clone_dirs": [str(clones)], "clone_dir": str(tmp / "data" / "repos")},
                "classification": {"domains": {"definitions": str(ROOT / "taxonomy" / "domains.yaml")},
                                   "operations": {"taxonomy": str(ROOT / "taxonomy" / "initial_categories.yaml")},
                                   "frameworks": str(ROOT / "taxonomy" / "frameworks.yaml")},
                "validation": {"annotation_sample_size": 12, "overlap_fraction": 1.0},
                "statistics": {"bootstrap_iterations": 50},
            }
            cfg_path = tmp / "config.json"
            cfg_path.write_text(json.dumps(cfg))
            c = ["--config", str(cfg_path)]

            def run(*a):
                return main([a[0], *c, *a[1:]])

            self.assertEqual(run("discover", "--import-dump", str(tmp / "dump.json")), 0)
            self.assertEqual(run("verify", "--local"), 0)
            self.assertEqual(run("classify-domains"), 0)
            self.assertEqual(run("sample", "--strategy", "all"), 0)
            self.assertEqual(run("clone", "--local-only"), 0)
            self.assertEqual(run("parse"), 0)
            self.assertEqual(run("analyze"), 0)
            self.assertEqual(run("classify-domains"), 0)
            self.assertEqual(run("discover-operations"), 0)
            # validation gate: blocked without human labels
            self.assertEqual(run("build-taxonomy"), 3)
            self.assertEqual(run("statistics"), 3)
            self.assertEqual(run("export-annotations"), 0)
            vdir = tmp / "val"
            sample = list(csv.DictReader((vdir / "annotation_sample.csv").open()))
            self.assertTrue(sample)
            self.assertNotIn("automatic_labels", sample[0])  # blind
            key = {r["sample_id"]: r["automatic_labels"] for r in csv.DictReader((vdir / "annotation_key.csv").open())}
            rows = list(csv.DictReader((vdir / "reviewer_decisions.csv").open()))
            for r in rows:  # annotators accept automatic labels; annotator_2 disagrees on the first item
                ops = key[r["sample_id"]] or "unknown"
                if r["annotator"] == "annotator_2" and r["sample_id"] == sample[0]["sample_id"]:
                    ops = "collection_sorting"
                r["operations"] = ops
            with (vdir / "reviewer_decisions.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            self.assertEqual(run("import-annotations"), 0)
            dis = list(csv.DictReader((vdir / "disagreements.csv").open()))
            self.assertEqual(len(dis), 1)
            self.assertEqual(run("build-taxonomy"), 0)
            tax = json.loads((vdir / "taxonomy_v1.json").read_text())
            self.assertEqual(tax["validation_basis"], "human_adjudicated_sample")
            self.assertEqual(run("statistics"), 0)
            stats = json.loads((tmp / "data" / "processed" / "statistics.json").read_text())
            ops = {r["operation"]: r for r in stats["operations_all_variants"]["all"]}
            self.assertEqual(ops["collection_filtering"]["repositories"], 3)
            # identical non-trivial methods copied across repositories count once in the deduplicated variant
            dedup = {r["operation"]: r for r in stats["operations"]}
            self.assertEqual(dedup["collection_filtering"]["repositories"], 1)
            self.assertEqual(stats["uncertainty"]["analysed_repositories"], 3)
            bench = tmp / "bench.json"
            bench.write_text(json.dumps({"tasks": [
                {"benchmark": "toy", "task_id": "t1", "operations": ["collection_filtering", "not_a_category"],
                 "transfer_status": "portable"},
                {"benchmark": "toy", "task_id": "t2", "description": "filter list", "operations": []}]}))
            self.assertEqual(run("benchmark-coverage", "--input", str(bench)), 0)
            cov = json.loads((tmp / "data" / "processed" / "benchmark_coverage.json").read_text())
            byop = {r["operation"]: r for r in cov["coverage"]}
            self.assertEqual(byop["collection_filtering"]["covering_tasks"], 1)
            self.assertEqual(byop["collection_sorting"]["existing_benchmark_coverage"], "none")
            self.assertEqual(run("report"), 0)
            for f in ("ecosystem_summary.md", "repository_selection.md", "operation_analysis.md", "taxonomy_report.md",
                      "benchmark_coverage.md"):
                self.assertTrue((tmp / "out" / "reports" / f).exists(), f)
            for f in ("repositories.csv", "repository_domains.csv", "methods.csv", "operation_evidence.csv",
                      "operation_statistics.csv", "taxonomy.csv"):
                self.assertTrue((tmp / "out" / "tables" / f).exists(), f)
            for f in ("operation_prevalence.png", "pharo_capabilities.png", "benchmark_coverage.png", "domain_distribution.png"):
                self.assertTrue((tmp / "out" / "figures" / f).exists(), f)
            ev = list(csv.DictReader((tmp / "out" / "tables" / "operation_evidence.csv").open()))
            self.assertTrue(all(r["commit"] and r["file"] and r["review_status"] == "pending" for r in ev))


if __name__ == "__main__":
    unittest.main()
