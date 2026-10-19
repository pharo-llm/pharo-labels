"""End-to-end test of the pharo-miner CLI on a synthetic offline corpus: mining, the two filters,
labelling (with a stand-in model) and the report."""
import csv
import json
import warnings
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

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
                "project": {"data_dir": str(tmp / "data"), "output_dir": str(tmp / "out"),
                            "labelling_dir": str(tmp / "out" / "labelling"), "database": str(tmp / "data" / "db.sqlite")},
                "repositories": {"existing_clone_dirs": [str(clones)], "clone_dir": str(tmp / "data" / "repos")},
                "classification": {"domains": {"definitions": str(ROOT / "taxonomy" / "domains.yaml")},
                                   "frameworks": str(ROOT / "taxonomy" / "frameworks.yaml")},
                "corpus_filter": {"require_license": False, "min_contributors": 0, "min_commits": 0,
                                  "min_development_days": 0, "max_years_since_last_push": 0, "require_baseline": True,
                                  "require_tonel": True, "min_non_test_methods": 0, "max_methods": 0, "min_test_methods": 0,
                                  "min_test_method_share": 0, "max_duplicated_method_share": 1.0, "max_parse_failure_share": 1.0},
                "labelling": {"embedding_model": None, "sample_size": 4, "batch_size": 2, "patience": 2, "max_rounds": 5,
                              "classify_sample_size": 10, "selfcheck_size": 2, "gold_min_per_stratum": 1},
            }
            cfg_path = tmp / "config.json"
            cfg_path.write_text(json.dumps(cfg))
            c = ["--config", str(cfg_path)]

            def run(*a):
                return main([a[0], *c, *a[1:]])

            # mining and filter 1
            self.assertEqual(run("discover", "--import-dump", str(tmp / "dump.json")), 0)
            self.assertEqual(run("verify", "--local"), 0)
            self.assertEqual(run("classify-domains"), 0)
            self.assertEqual(run("sample", "--strategy", "all"), 0)
            self.assertEqual(run("clone", "--local-only"), 0)
            self.assertEqual(run("parse"), 0)
            self.assertEqual(run("analyze"), 0)
            self.assertEqual(run("classify-domains"), 0)
            # labelling before filter 2 is blocked
            self.assertEqual(run("label"), 3)
            # filter 2
            self.assertEqual(run("filter-corpus"), 0)
            corpus = json.loads((tmp / "data" / "processed" / "corpus.json").read_text())
            self.assertEqual(len(corpus["selected"]), 3)
            # labelling with a stand-in model (no Ollama needed)
            with mock.patch("pharo_task_miner.labelling.models.make_model",
                            lambda spec, s: FakeModel()):
                self.assertEqual(run("label"), 0)
            lab = tmp / "out" / "labelling"
            plan = json.loads((lab / "plan.json").read_text())
            self.assertEqual(plan["initial_sample_size"], 4)
            labels = list(csv.DictReader((lab / "labels.csv").open()))
            self.assertTrue(labels and all(r["url"].startswith("https://github.com/") for r in labels))
            self.assertLessEqual({r["category"] for r in labels}, {"Accessor", "Collection filtering", "(rare name: not a category)"})
            draft = list(csv.DictReader((lab / "codebook_draft.csv").open()))
            self.assertEqual({r["category"] for r in draft} - {"Collection filtering"}, {"Accessor"})
            self.assertTrue(all(r["definition"].startswith("Methods that") for r in draft))
            # deductive phase: three stand-in models that always agree
            with mock.patch("pharo_task_miner.labelling.models.make_model",
                            lambda spec, s: FakeModel(spec)):
                self.assertEqual(run("classify", "--models", "a b c"), 0)
            agreement = json.loads((lab / "classify" / "agreement.json").read_text())
            self.assertTrue(agreement["reliable"])
            self.assertEqual(agreement["selfcheck"]["methods"], 2)
            inductive = {json.loads(x)["method_id"] for x in (lab / "labels.jsonl").read_text().splitlines()}
            self.assertFalse(inductive & set(json.loads((lab / "classify" / "sample.json").read_text())["method_ids"]))
            for f in ("README.md", "codebook_review.csv", "method_review.csv", "examples.md"):
                self.assertTrue((lab / "expert_validation" / f).exists(), f)
            review = list(csv.DictReader((lab / "expert_validation" / "method_review.csv").open()))
            self.assertNotIn("majority", review[0])                     # blind: no model answer shown
            # two experts return their files; validation
            for e in ("ann", "bob"):
                d = lab / "expert_validation" / "returned" / e
                d.mkdir(parents=True)
                with (d / "method_review.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(review[0]))
                    w.writeheader()
                    w.writerows({**r, "expert_category": "Collection filtering" if "select:" in r["source"] else "Accessor"}
                                for r in review)
            self.assertEqual(run("validate"), 0)
            v = json.loads((lab / "validated" / "validation.json").read_text())
            self.assertEqual(v["accuracy"]["majority"]["estimate"], 1.0)
            self.assertAlmostEqual(sum(t["share"] for t in v["final_taxonomy"]), 1.0, places=2)
            for f in ("saturation_curve.png", "category_distribution.png"):
                self.assertTrue((tmp / "out" / "figures" / f).exists(), f)
            # the report covers every step
            self.assertEqual(run("report"), 0)
            report = (tmp / "out" / "REPORT.md").read_text()
            for h in ("## 1. GitHub mining", "## 2. Filter 2", "## 3. Sample size", "## 4. The saturation loop",
                      "## 5. The codebook, classified by three models", "**Krippendorff's α**",
                      "## 6. The experts' answers", "**Accuracy of the majority label**",
                      "## 7. What the labelling would cost", "**Whole corpus**",
                      "**How to reduce it**"):
                self.assertIn(h, report)


class FakeModel:
    """Labels a method from its own code, so several instances always agree."""

    def __init__(self, name="fake"):
        self.name = name

    def describe(self):
        return {"backend": "fake"}

    def complete(self, prompt, max_tokens, stop):
        if "one-sentence definition" in prompt:
            return "Methods that do this kind of task."
        task = "Collection filtering" if "select:" in prompt.rsplit("### Method of", 1)[1] else "Accessor"
        if "Codebook:" in prompt:
            return task
        return f" Does something.\n### Task: {task}"


if __name__ == "__main__":
    unittest.main()
