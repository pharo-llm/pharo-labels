"""Unit tests for PharoTaskMiner core components (parser, readers, features, detection, repository sampling)."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharo_task_miner.extraction.smalltalk_parser import ParseError, count_nodes, iter_nodes, parse_method  # noqa: E402
from pharo_task_miner.extraction.tonel_reader import classify_path, read_chunk, read_filetree_method, read_tonel  # noqa: E402
from pharo_task_miner.extraction.method_extractor import ast_facts, baseline_dependencies, extract_repository  # noqa: E402
from pharo_task_miner.extraction.pharo_parser_bridge import PythonBackend  # noqa: E402
from pharo_task_miner.analysis.structural_features import compute_features  # noqa: E402
from pharo_task_miner.analysis.api_usage import RepositoryIndex, resolve_send  # noqa: E402
from pharo_task_miner.analysis.domain_classifier import DomainClassifier  # noqa: E402
from pharo_task_miner.discovery.pharo_detection import Listing, verify  # noqa: E402
from pharo_task_miner.sampling.diversity_sampling import sample  # noqa: E402
from pharo_task_miner.sampling.repository_filters import is_coursework, quality  # noqa: E402
from pharo_task_miner.config import load_yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def sends(src):
    return [n["selector"] for n, *_ in iter_nodes(parse_method(src)) if n.get("type") == "Message"]


class ParserTests(unittest.TestCase):
    def test_precedence_keywords_cascade(self):
        self.assertCountEqual(sends("foo ^ Dictionary new at: (items collect: [ :x | x size + 1 ]) put: 3; yourself"),
                              ["new", "size", "+", "collect:", "at:put:", "yourself"])

    def test_cascade_structure(self):
        ast = parse_method("foo OrderedCollection new add: 1; add: 2 + 3; yourself")
        c = ast["body"]["statements"][0]
        self.assertEqual(c["type"], "Cascade")
        self.assertEqual(c["receiver"]["selector"], "new")
        self.assertEqual([m["selector"] for m in c["messages"]], ["add:", "add:", "yourself"])
        self.assertTrue(all(m["receiver"]["type"] == "CascadeReceiver" for m in c["messages"]))

    def test_literals_are_not_sends(self):
        src = "foo <primitive: 60> | r | \"ZnClient new\" r := #(a b: #c (1 2) $) #[1 2]) , 'x y: z'. ^ #at:put: size"
        self.assertCountEqual(sends(src), [",", "size"])

    def test_negative_numbers_and_binary_minus(self):
        self.assertCountEqual(sends("foo ^ 3@-4 - -1.5e-3 + 16rFF"), ["@", "-", "+"])
        lits = [n["value"] for n, *_ in iter_nodes(parse_method("foo ^ 3@-4 - -1.5e-3")) if n.get("type") == "Literal"]
        self.assertCountEqual(lits, ["3", "-4", "-1.5e-3"])

    def test_block_args_and_temps(self):
        ast = parse_method("foo ^ [:x || t | t := x. t] value: 1")
        blk = ast["body"]["statements"][0]["value"]["receiver"]
        self.assertEqual(blk["arguments"], ["x"])
        self.assertEqual(blk["body"]["temporaries"], ["t"])

    def test_double_hash_literal_array_and_underscore_identifier(self):
        parse_method("foo ##(A B) do: [ :_ :each | each ]")

    def test_malformed_reports_error(self):
        with self.assertRaises(ParseError):
            parse_method("foo ^ (1 + ")
        with self.assertRaises(ParseError):
            parse_method("foo ^ 1 2")


TONEL = '''"
A comment with ""quotes""
"
Class {
	#name : #Foo,
	#superclass : #Bar,
	#instVars : [ 'items', 'count' ],
	#classVars : [ 'Default' ],
	#category : #'Pkg-Core',
	#package : 'Pkg'
}

{ #category : #accessing }
Foo >> items [
	^ items
]

{ #category : 'private' }
Foo >> at: i put: v [
	"a ] bracket in a comment and in a string ']' and char $]"
	items at: i put: v.
	^ v
]

{ #category : #'instance creation' }
Foo class >> new [
	^ super new initialize
]
'''


class ReaderTests(unittest.TestCase):
    def test_tonel(self):
        r = read_tonel(TONEL, "src/Pkg/Foo.class.st")
        self.assertEqual(r.status, "ok", r.warnings)
        self.assertEqual(r.classes[0].instance_variables, ["items", "count"])
        self.assertEqual(r.classes[0].superclass, "Bar")
        self.assertEqual([m.selector_hint for m in r.methods], ["items", "at:put:", "new"])
        self.assertEqual(r.methods[2].side, "class")
        self.assertEqual(r.methods[1].protocol, "private")
        self.assertEqual(r.methods[1].line, 19)
        self.assertTrue(parse_method(r.methods[1].source))

    def test_filetree(self):
        r = read_filetree_method("accessing\nitems\n\t^ items", "repo/Pkg.package/Foo.class/instance/items.st")
        m = r.methods[0]
        self.assertEqual((m.class_name, m.protocol, m.package), ("Foo", "accessing", "Pkg"))
        self.assertEqual(parse_method(m.source)["selector"], "items")

    def test_chunk(self):
        src = "Object subclass: #Foo\n\tinstanceVariableNames: 'a'\n\tclassVariableNames: ''\n\tpackage: 'P'!\n\n!Foo methodsFor: 'x'!\nbar\n\t^ a!\nbaz: y\n\t^ 'it''s!!'! !\n"
        r = read_chunk(src, "foo.st")
        self.assertEqual([m.selector_hint for m in r.methods], ["bar", "baz:"])
        self.assertIn("!", r.methods[1].source)
        self.assertEqual(r.classes[0].name, "Foo")

    def test_classify_path(self):
        self.assertEqual(classify_path("src/A/B.class.st"), "tonel")
        self.assertEqual(classify_path("r/P.package/C.class/class/x.st"), "filetree_method")
        self.assertEqual(classify_path("x.gs"), "gemstone_topaz")


class FactsAndFeaturesTests(unittest.TestCase):
    def setUp(self):
        r = read_tonel(TONEL, "src/Pkg/Foo.class.st")
        self.table = {c.name: c for c in r.classes}

    def facts(self, src, side="instance"):
        ast = parse_method(src)
        m = {"selector": ast["selector"], "side": side, "arguments": ast["arguments"], "source": src, "ast": ast,
             "is_test": False}
        m.update(ast_facts(ast, "Foo", side, self.table))
        return m

    def test_scopes_and_getter_setter(self):
        g = self.facts("items ^ items")
        self.assertEqual(g["variables_read"]["instance_variable"], ["items"])
        self.assertTrue(compute_features(g)["is_getter"])
        s = self.facts("items: x items := x")
        self.assertTrue(compute_features(s)["is_setter"])

    def test_inlined_blocks_early_vs_nonlocal_return(self):
        f = compute_features(self.facts("foo: x x ifNil: [ ^ 0 ]. ^ x"))
        self.assertTrue(f["has_early_return"])
        self.assertFalse(f["has_nonlocal_return"])
        f2 = compute_features(self.facts("foo ^ items detect: [ :e | e > 1 ] ifNone: [ ^ nil ]"))
        self.assertTrue(f2["has_nonlocal_return"])

    def test_recursion_and_reflection(self):
        f = compute_features(self.facts("walk: n n children do: [ :c | self walk: c ]. ^ n perform: #foo"))
        self.assertTrue(f["has_recursion"] and f["recursion_in_iteration"] and f["has_dynamic_send"])

    def test_receiver_resolution(self):
        methods = [{"class_name": "Foo", "side": "instance", "selector": "items", "parse_status": "ok"},
                   {"class_name": "Foo", "side": "class", "selector": "new", "parse_status": "ok"}]
        idx = RepositoryIndex([{"name": "Foo", "kind": "class", "superclass": "Bar"}], methods)
        m = {"class_name": "Foo", "side": "instance"}
        self.assertEqual(resolve_send({"receiver_kind": "self", "selector": "items"}, m, idx)["resolved_target"], "Foo>>items")
        self.assertEqual(resolve_send({"receiver_kind": "global", "receiver_name": "Foo", "selector": "new"}, m, idx)["resolved_target"], "Foo class>>new")
        r = resolve_send({"receiver_kind": "temporary", "receiver_name": "x", "selector": "add:"}, m, idx)
        self.assertIsNone(r["resolved_target"])
        self.assertEqual(r["receiver_resolution"], "unknown")

    def test_baseline_dependencies(self):
        src = "baseline: spec <baseline> spec for: #common do: [ spec baseline: 'NeoJSON' with: [ spec repository: 'x' ]. spec package: 'A' with: [ spec requires: #('NeoJSON' 'B') ] ]"
        ast = parse_method(src)
        deps = baseline_dependencies([{"class_name": "BaselineOfFoo", "ast": ast}])
        kinds = {(d["kind"], d["name"]) for d in deps}
        self.assertIn(("external_project", "NeoJSON"), kinds)
        self.assertIn(("requires", "B"), kinds)


class DomainAndDetectionTests(unittest.TestCase):
    def test_domain_multilabel_with_evidence_and_no_name_only(self):
        clf = DomainClassifier(load_yaml(ROOT / "taxonomy" / "domains.yaml"))
        res = clf.classify({"full_name": "a/b", "name": "b", "description": "A web framework for REST APIs",
                            "readme": "Store data in a PostgreSQL database with Glorp.", "topics": ["seaside"],
                            "dependencies": ["P3"], "packages": [], "framework_shares": {}})
        labels = {l["label"] for l in res["domain_labels"]}
        self.assertTrue({"web_development", "data_persistence"} <= labels)
        self.assertTrue(all(l["score_kind"] == "uncalibrated_weight_sum" for l in res["domain_labels"]))
        res2 = clf.classify({"full_name": "a/web-chess-game", "name": "web-chess-game", "description": "", "readme": "",
                             "topics": [], "dependencies": [], "packages": [], "framework_shares": {}})
        self.assertEqual(res2["domain_labels"][0]["label"], "unknown")

    def test_pharo_detection(self):
        v = verify(Listing(["src/BaselineOfX/BaselineOfX.class.st", "src/X/Foo.class.st", ".project"], {}, False, None, "test"),
                   "Load in Pharo 12", {".smalltalk.ston": "SmalltalkCISpec { #smalltalk : 'Pharo64-12' }"}, [], None, "Smalltalk")
        self.assertEqual(v.status, "confirmed_pharo")
        self.assertTrue(v.evidence)
        v2 = verify(Listing(["src/Foo.pck.st"], {}, False, None, "test"), "A Cuis package", {}, [], None, "Smalltalk")
        self.assertEqual(v2.status, "non_pharo")
        v3 = verify(Listing(["README.md"], {}, False, None, "test"), "Pharo notes", {}, [], None, None)
        self.assertEqual(v3.status, "non_pharo")
        self.assertEqual(verify(None, None, {}, ["pharo"], None, "Smalltalk").status, "uncertain")


class SamplingTests(unittest.TestCase):
    def test_sampling_reproducible_and_owner_cap(self):
        pool = [{"repo_id": i, "full_name": f"o{i % 3}/r{i}", "owner": f"o{i % 3}", "primary_domain": ["web", "viz"][i % 2],
                 "size_category": "small", "project_kind": "app", "educational": False, "core_infrastructure": False}
                for i in range(30)]
        cfg = {"max_per_owner": 3, "strata": ["primary_domain", "size_category"],
               "diversity_attributes": ["primary_domain", "owner"]}
        for strat in ("random", "stratified", "diversity"):
            a, _, _ = sample(pool, strat, 6, 7, cfg)
            b, _, _ = sample(pool, strat, 6, 7, cfg)
            self.assertEqual([p["repo_id"] for p in a], [p["repo_id"] for p in b])
            self.assertLessEqual(max(sum(1 for p in a if p["owner"] == o) for o in ("o0", "o1", "o2")), 3)

    def test_quality_criteria_give_the_first_failing_reason(self):
        q = {"require_pharo_source": True, "exclude_coursework": True, "min_commits": 10,
             "min_pharo_source_files": 10, "require_tests": True, "require_external_interest": True}
        good = {"n_pharo_source_files": 40, "coursework": False, "commit_count": 120, "has_tests": True,
                "contributors": 1, "stars": 3, "forks": 0}
        self.assertEqual(quality(good, q), (True, "eligible"))
        for change, reason in (({"n_pharo_source_files": 0}, "no Pharo package source"), ({"coursework": True}, "coursework"),
                               ({"commit_count": 9}, "fewer than 10 commits"),
                               ({"n_pharo_source_files": 9}, "fewer than 10 Pharo source files"),
                               ({"has_tests": False}, "no tests"), ({"stars": 0}, "single author")):
            ok, why = quality({**good, **change}, q)
            self.assertFalse(ok)
            self.assertIn(reason, why)
        # criteria that are off, and values that were never measured, do not exclude
        self.assertTrue(quality({**good, "commit_count": 1, "has_tests": False}, {})[0])
        self.assertTrue(quality({**good, "commit_count": None, "n_pharo_source_files": None, "has_tests": None}, q)[0])

    def test_coursework_detection(self):
        for owner, name, desc in (("dvmason", "AdventOfCode2021", None), ("uca-argentina", "2025-tdd-lovers", None),
                                  ("hpi-swa-teaching", "PixelEditor", None), ("x", "OO2-TP1", None), ("x", "CuOOraTp7", None),
                                  ("x", "Thing", "Homework for the Pharo MOOC")):
            self.assertTrue(is_coursework(owner, name, desc), (owner, name))
        for owner, name, desc in (("SeasideSt", "Seaside", "The framework for developing web applications"),
                                  ("hpi-swa", "smalltalkCI", "Framework for testing Smalltalk projects"),
                                  ("pharo-project", "pharo-launcher", None), ("astares", "Pharo-Examples-Lab", None)):
            self.assertFalse(is_coursework(owner, name, desc), (owner, name))


class ExtractionTests(unittest.TestCase):
    def test_extract_repository(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "src" / "Pkg").mkdir(parents=True)
            (root / "src" / "Pkg" / "Foo.class.st").write_text(TONEL)
            (root / "src" / "Pkg" / "package.st").write_text("Package { #name : 'Pkg' }")
            (root / "src" / "Pkg" / "Bad.class.st").write_text("Class { #name : #Bad }\n\nBad >> broken [\n\t^ (1 +\n]\n")
            (root / "scripts").mkdir()
            (root / "scripts" / "load.st").write_text("Metacello new load.")
            ex = extract_repository(root, "o/r", "abc", PythonBackend())
            statuses = {m["selector"]: m["parse_status"] for m in ex.methods}
            self.assertEqual(statuses["at:put:"], "ok")
            self.assertIn("failed", statuses.values())
            self.assertTrue(any(f["format"] == "script" and f["status"] == "unsupported" for f in ex.files))
            ok = next(m for m in ex.methods if m["selector"] == "at:put:")
            self.assertEqual(ok["package"], "Pkg")
            self.assertEqual(ok["commit_sha"], "abc")
            self.assertTrue(ok["message_sends"])


if __name__ == "__main__":
    unittest.main()
