"""Unit tests of the parts every result depends on: python -m unittest discover -s tests"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from pbr import agreement as A
from pbr import taxonomy as T
from pbr import workbook as W
from pbr.phases.p02_corpus import method_id, strip_comments
from pbr.phases.p03_sampling import allocate
from pbr.pharo import clean_method, gates, method_selector, split_methods


def small_tax():
    t = T.empty("v1")
    t["categories"] = [{"id": "C001", "name": "Collections", "definition": "work with collections", "status": "active"}]
    t["subcategories"] = [{"id": "S001", "name": "Indexed access", "definition": "access by position", "category_id": "C001",
                           "status": "active"}]
    t["behaviors"] = [
        {"id": "B001", "name": "Retrieve an element by index", "definition": "get the element at a position",
         "subcategory_id": "S001", "category_id": "C001", "status": "active"},
        {"id": "B002", "name": "Element retrieval by position", "definition": "same thing",
         "subcategory_id": "S001", "category_id": "C001", "status": "active"}]
    return t


class TaxonomyTest(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(T.validate(small_tax()), [])

    def test_invalid_parent(self):
        t = small_tax()
        t["behaviors"][0]["subcategory_id"] = "S999"
        self.assertTrue(any("unknown subcategory" in e for e in T.validate(t)))

    def test_merge_keeps_ids_and_maps(self):
        t = small_tax()
        v2, problems = T.apply_decisions(t, [{"id": "B002", "action": "merge", "merge_into": "B001", "reviewer": "ex1"}], "v2", "test")
        self.assertEqual(problems, [])
        self.assertEqual(T.behavior_ids(v2), ["B001"])
        self.assertEqual(v2["id_map"]["B002"], ["B001"])
        self.assertIn("Element retrieval by position", T.index(v2)["B001"]["aliases"])
        self.assertEqual(T.translate(["B002"], t, v2), ["B001"])
        self.assertEqual(T.behavior_ids(t), ["B001", "B002"], "the earlier version is never modified")

    def test_split_rename_reject_add(self):
        t = small_tax()
        v2, problems = T.apply_decisions(t, [
            {"id": "B001", "action": "split", "split_into": "Read by index | Read first element", "reviewer": "e"},
            {"id": "B002", "action": "rename", "new_name": "Positional read", "reviewer": "e"},
            {"id": "", "action": "add", "new_name": "Write by index", "parent_id": "S001", "new_definition": "set at a position",
             "reviewer": "e"},
            {"id": "B999", "action": "reject", "reviewer": "e"}], "v2", "test")
        self.assertTrue(any("B999" in p for p in problems))
        names = {b["name"] for b in T.active(v2, "behavior")}
        self.assertEqual(names, {"Read by index", "Read first element", "Positional read", "Write by index"})
        self.assertEqual(len(T.translate(["B001"], t, v2)), 2)

    def test_next_id_respects_registry(self):
        self.assertEqual(T.next_id(small_tax(), "behavior", {"behavior": ["B007"]}), "B008")


class AgreementTest(unittest.TestCase):
    def test_alpha_perfect_and_masi(self):
        self.assertEqual(A.krippendorff_alpha([["a", "a", "a"], ["b", "b", "b"]]), 1.0)
        sets = [[frozenset("a"), frozenset("a")], [frozenset("ab"), frozenset("ab")], [frozenset("b"), frozenset("b")]]
        self.assertEqual(A.krippendorff_alpha(sets, A.masi), 1.0)
        self.assertAlmostEqual(A.masi({"a"}, {"a", "b"}), 1 - 0.5 * 2 / 3)
        self.assertEqual(A.masi({"a"}, {"b"}), 1.0)

    def test_alpha_known_value(self):
        # Krippendorff (2011) nominal example, 4 coders, missing values
        data = [[1, 1, None, 1], [2, 2, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2], [1, 2, 3, 4], [4, 4, 4, 4],
                [1, 1, 2, 1], [2, 2, 2, 2], [None, 5, 5, 5], [None, None, 1, 1], [None, None, 3, None]]
        self.assertAlmostEqual(A.krippendorff_alpha(data), 0.743, places=3)

    def test_levels(self):
        self.assertEqual(A.agreement_level([{"a"}, {"a"}, {"a"}]), "full")
        self.assertEqual(A.agreement_level([{"a"}, {"a", "b"}, {"c"}]), "partial")
        self.assertEqual(A.agreement_level([{"a"}, {"b"}, {"c"}]), "none")
        self.assertEqual(A.majority_set([{"a", "b"}, {"a"}, {"c"}]), frozenset({"a"}))

    def test_prf(self):
        r = A.multilabel_prf([{"a"}, {"b"}], [{"a"}, {"a"}])
        self.assertEqual(r["micro"]["precision"], 0.5)
        self.assertEqual(r["exact_match"], 0.5)

    def test_weighted_share(self):
        r = A.weighted_share([1, 0, 1, 0], [1, 1, 3, 3], ["r1", "r1", "r2", "r2"], ["g", "g", "g", "g"])
        self.assertAlmostEqual(r["share"], 0.5)
        self.assertLessEqual(r["ci_low"], 0.5)

    def test_ppi_unbiased_when_model_perfect(self):
        r = A.ppi_mean([1, 0, 1, 0], [1, 0], [1, 0])
        self.assertEqual(r["share"], 0.5)
        self.assertEqual(r["correction"], 0.0)


class CorpusTest(unittest.TestCase):
    def test_strip_comments_keeps_strings(self):
        src = 'foo\n\t"a comment"\n\t^ \'say "hi"\' , $" asString'
        self.assertEqual(strip_comments(src), 'foo ^ \'say "hi"\' , $" asString')

    def test_method_id_stable(self):
        self.assertEqual(method_id("o/r", "p.st", "C", "instance", "x"), method_id("o/r", "p.st", "C", "instance", "x"))
        self.assertNotEqual(method_id("o/r", "p.st", "C", "class", "x"), method_id("o/r", "p.st", "C", "instance", "x"))

    def test_allocate(self):
        g = pd.DataFrame({"repos": [10, 2, 5], "methods": [1000, 10, 90]}, index=["a", "b", "c"])
        self.assertEqual(sum(allocate(10, g, "equal").values()), 10)
        self.assertEqual(allocate(10, g, "proportional")["a"], 9)


class WorkbookTest(unittest.TestCase):
    def test_roundtrip_keeps_reviewer(self):
        wb = W.new_workbook("t", ["x"])
        W.add_sheet(wb, "S", [{"row_key": "k1", "v": "a"}], [("row_key", "row_key", 10, False), ("answer", "answer", 10, True)],
                    choices={"answer": ["B001 x", "UNKNOWN"]})
        wb["README"]["B3"] = "expert-1"
        with tempfile.TemporaryDirectory() as d:
            p = W.save(wb, Path(d) / "e.xlsx")
            reviewer, rows, prov = W.read_sheet(p, "S")
        self.assertEqual(reviewer, "expert-1")
        self.assertEqual(rows[0]["row_key"], "k1")
        self.assertEqual(len(prov["sha256"]), 64)


class PharoTest(unittest.TestCase):
    def test_selector_and_cleaning(self):
        self.assertEqual(method_selector("at: i put: v\n\t^ x"), "at:put:")
        self.assertEqual(method_selector("+ other\n\t^ 1"), "+")
        self.assertEqual(clean_method("```smalltalk\nFoo >> bar [\n\t^ 1\n]\n```"), "bar\n\t^ 1")
        self.assertEqual(len(split_methods("```smalltalk\nFoo >> a\n\t^1\nFoo >> b\n\t^2\n```")), 2)

    def test_gates_order(self):
        cfg = {"pharo": {"min_tests": 2, "repeat": 1, "min_mutation_score": 0.5}}
        r = {"stage": "done", "facts": [], "tests": ["t1", "t2"], "reference_runs": [{"t1": {"outcome": "pass"}, "t2": {"outcome": "pass"}}],
             "stub_runs": [{"tests": {"t1": {"outcome": "error"}, "t2": {"outcome": "pass"}}}],
             "mutation": {"evaluated": 10, "killed": 4, "terminated": 0}}
        g, first = gates(r, cfg)
        self.assertTrue(g["fail_to_pass"])
        self.assertEqual(first, "mutation")


if __name__ == "__main__":
    unittest.main()


class ValidationAnalysisTest(unittest.TestCase):
    def test_item_table_metrics_and_human_vs_ai(self):
        from pbr.phases import p07_validation as V
        tax = small_tax()
        tax["behaviors"].append({"id": "B003", "name": "Write by index", "definition": "set at a position",
                                 "subcategory_id": "S001", "category_id": "C001", "status": "active"})
        models = ["m1", "m2", "m3"]
        answers = {"i1": [["B001"], ["B001"], ["B001"]], "i2": [["B001", "B003"], ["B003"], ["B002"]],
                   "i3": [["B002"], ["B003"], ["B001"]], "i4": [["INSUFFICIENT"], ["INSUFFICIENT"], ["B001"]]}
        rows = []
        for item, sets in answers.items():
            for m, s in zip(models, sets):
                rows.append({"item_id": item, "source": "pharo", "group": "g", "purpose": "random", "model": m,
                             "behavior_ids": s, "primary_behavior": s[0], "primary_category": T.behavior_path(tax, s[0])[0]})
        items = V.item_table(pd.DataFrame(rows), models)
        lv = dict(zip(items["item_id"], items["level"]))
        self.assertEqual(lv, {"i1": "full", "i2": "partial", "i3": "none", "i4": "insufficient"})
        self.assertEqual(dict(zip(items["item_id"], items["majority_primary"]))["i1"], "B001")
        met = V.ai_metrics(items, models, tax)
        self.assertEqual(met["overall"]["items"], 4)
        sel = items.assign(stratum=items["level"].map(lambda l: l if l in ("full", "partial") else "none"))
        sel["stratum_N"] = sel["stratum"].map(sel["stratum"].value_counts())
        sel["stratum_m"] = sel["stratum_N"]
        ref = pd.DataFrame({"item_id": ["i1", "i2", "i3"], "reference": [["B001"], ["B003"], ["B002"]]})
        hv = V.human_vs_ai(ref, items, sel, models, tax)
        self.assertEqual(hv["m1"]["exact_match"], round(2 / 3, 4))
        self.assertEqual(hv["m1"]["primary_behavior_in_reference"], round(2 / 3, 4))
        self.assertEqual(hv["ai_majority"]["primary_behavior_correct_by_ai_agreement_level"]["full"], 1.0)

    def test_parse_ids(self):
        from pbr.phases.p07_validation import parse_ids
        ids, bad = parse_ids(["B001 Retrieve", "unknown", "B999 x", None], {"B001"})
        self.assertEqual(ids, ["B001", "UNKNOWN"])
        self.assertEqual(bad, ["B999 x"])
