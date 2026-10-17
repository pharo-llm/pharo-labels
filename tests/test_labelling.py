"""Unit tests for the labelling: sample size, draws, answer parsing, codebook, saturation loop."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharo_task_miner.labelling import coding, sampling, saturation  # noqa: E402


class SampleSizeTests(unittest.TestCase):
    def test_cochran_with_finite_population(self):
        c = sampling.cochran(278043, 0.95, 0.05, 0.5)
        self.assertAlmostEqual(c["n0"], 384.15, places=1)
        self.assertEqual(c["n"], 384)
        self.assertEqual(sampling.cochran(1000, 0.95, 0.05)["n"], 278)   # correction matters for small N
        self.assertEqual(sampling.cochran(100, 0.95, 0.5)["n"], 4)

    def test_margin_is_the_inverse(self):
        self.assertAlmostEqual(sampling.margin_for(384, 278043), 0.05, places=3)
        self.assertGreater(sampling.margin_for(300, 278043), 0.05)

    def test_quiet_run_length(self):
        L = sampling.quiet_run_length(0.005, 0.95)
        self.assertEqual(L, 598)
        self.assertLessEqual((1 - 0.005) ** L, 0.05)                       # a 0.5% task shows with >= 95%
        self.assertGreater((1 - 0.005) ** (L - 1), 0.05)
        self.assertEqual(sampling.quiet_run_length(0, 0.95), 0)

    def test_wilson(self):
        lo, hi = sampling.wilson(5, 10)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertEqual(sampling.wilson(0, 0), (0.0, 0.0))
        lo2, hi2 = sampling.wilson(5, 10, population=12)              # almost a census: narrower
        self.assertLess(hi2 - lo2, hi - lo)

    def test_draw_is_proportional_reproducible_and_excludes(self):
        pop = {1: list(range(0, 600)), 2: list(range(600, 900)), 3: list(range(900, 1000))}
        a = sampling.draw(pop, 100, seed=7)
        self.assertEqual(a, sampling.draw(pop, 100, seed=7))
        self.assertEqual(len(a), 100)
        self.assertEqual(sum(1 for m in a if m < 600), 60)
        self.assertEqual(sum(1 for m in a if 600 <= m < 900), 30)
        b = sampling.draw(pop, 100, seed=8, exclude=set(a))
        self.assertFalse(set(a) & set(b))
        self.assertEqual(len(sampling.draw(pop, 5000, seed=1)), 1000)


class AnswerTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(coding.parse_answer(" Returns the x coordinate.\n### Task: Accessor"),
                         ("Returns the x coordinate.", "Accessor"))
        self.assertEqual(coding.parse_answer("Does things."), ("Does things.", None))
        self.assertEqual(coding.parse_answer("Sets it.\n### Task: x := 3"), ("Sets it.", None))
        self.assertEqual(coding.parse_answer("Parses.\n### Task: parsing."), ("Parses.", "Parsing"))

    def test_clean_name_rejects_code_and_sentences(self):
        self.assertIsNone(coding.clean_name("self assert: x"))
        self.assertIsNone(coding.clean_name("this method returns the value of the field"))
        self.assertEqual(coding.clean_name('"event handling"'), "Event handling")

    def test_prompt_has_examples_then_the_method(self):
        p = coding.build_prompt({"class_name": "Foo", "side": "class", "source": "bar\n\t^ 1"})
        self.assertTrue(p.endswith("### Method of Foo class\nbar\n\t^ 1\n### Summary:"))
        self.assertEqual(p.count("### Task:"), len(coding.EXAMPLES))
        long = coding.build_prompt({"class_name": "A", "source": "\n".join(["x"] * 100)}, max_lines=5)
        self.assertIn('"..."', long)


class CodebookTests(unittest.TestCase):
    def test_merging_and_support(self):
        cb = coding.Codebook(min_support=2, seeds=["Accessor"])
        self.assertEqual(cb.observe("Accessors", 0), "Accessor")             # plural: same category
        self.assertEqual(cb.observe("Accessor method", 0), "Accessor")       # generic last word
        self.assertEqual(cb.observe("Serialisation", 0), "Serialisation")
        self.assertEqual(cb.admit(0, 3), ["Accessor"])
        self.assertEqual(cb.observe("Serialization", 1), "Serialisation")    # spelling variant
        self.assertEqual(cb.observe("Collection modification", 1), "Collection modification")
        self.assertEqual(cb.observe("Object modification", 1), "Object modification")  # not merged
        self.assertEqual(cb.admit(1, 6), ["Serialisation"])
        self.assertEqual(cb.taxonomy(), ["Accessor", "Serialisation"])
        self.assertEqual(cb.taxonomy(0), ["Accessor"])
        self.assertIsNone(cb.final("Object modification"))
        self.assertEqual(cb.final("serialization"), "Serialisation")
        self.assertIsNone(cb.observe(None, 2))

    def test_share_threshold(self):
        cb = coding.Codebook(min_support=2, seeds=[], min_share=0.1)
        for _ in range(3):
            cb.observe("Parsing", 0)
        self.assertEqual(cb.admit(0, 40), [])                                # 3 < 10% of 40
        self.assertEqual(cb.admit(1, 30), ["Parsing"])

    def test_semantic_merge_is_recorded(self):
        vecs = {"Computation": [1.0, 0.0], "Calculation": [0.96, 0.28], "Parsing": [0.0, 1.0]}
        cb = coding.Codebook(min_support=1, seeds=[], embed=lambda xs: [vecs[x] for x in xs], semantic_threshold=0.9)
        cb.observe("Computation", 0)
        self.assertEqual(cb.observe("Calculation", 0), "Computation")
        self.assertEqual(cb.cats["Computation"]["merged"], {"Calculation": 0.96})
        self.assertEqual(cb.observe("Parsing", 1), "Parsing")                # 0.0 similarity: a new name
        cb.admit(1, 3)
        self.assertEqual(cb.final("calculation"), "Computation")

    def test_opposites_never_merge_by_meaning(self):
        vecs = {"Deserialization": [1.0, 0.0], "Serialization": [0.99, 0.14]}
        cb = coding.Codebook(min_support=1, seeds=[], embed=lambda xs: [vecs[x] for x in xs], semantic_threshold=0.9)
        cb.observe("Deserialization", 0)
        self.assertEqual(cb.observe("Serialization", 0), "Serialization")
        self.assertTrue(coding.opposite("Unit installation", "Unit uninstallation") is False)
        self.assertTrue(coding.opposite("Locking", "Unlocking"))

    def test_unused_seed_is_not_in_the_taxonomy(self):
        cb = coding.Codebook(min_support=2, seeds=["Printing"])
        cb.observe("Printing", 0)
        cb.admit(0, 1)
        self.assertEqual(cb.taxonomy(), [])


class FakeModel:
    """Answers from a fixed list of task names, one per call (cycling)."""
    name = "fake"

    def __init__(self, tasks):
        self.tasks, self.calls = tasks, 0

    def describe(self):
        return {"backend": "fake"}

    def complete(self, prompt, max_tokens, stop):
        t = self.tasks[self.calls % len(self.tasks)]
        self.calls += 1
        return f" Does something.\n### Task: {t}"


def records(ids):
    return {m: {"method_id": m, "full_name": f"o/r{m % 3}", "signature": f"C>>m{m}", "is_test": m % 4 == 0,
                "class_name": "C", "side": "instance", "source": f"m{m}\n\t^ {m}", "url": None} for m in ids}


class SaturationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pop = {1: list(range(1, 301)), 2: list(range(301, 401))}
        self.s = {"model": "fake", "random_seed": 1, "sample_size": 20, "batch_size": 10, "patience": 2,
                  "max_rounds": 10, "min_support": 2, "merge_similarity": 0.88, "include_tests": True,
                  "max_method_lines": 40, "max_tokens": 40, "method_review_size": 5, "prompt_version": "t",
                  "corpus_run_id": "r"}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_loop_stops_at_saturation_and_resumes(self):
        out = self.tmp / "lab"
        saturation.prepare(out, dict(self.s), lambda m: None)
        model = FakeModel(["Accessor", "Parsing", "Accessor", "Rendering"])
        res = saturation.Run(out, self.s, self.pop, records, lambda m: None).run(model)
        self.assertEqual(res["stop_reason"], "saturated")
        self.assertEqual(res["rounds"], 2)                              # rounds 1 and 2 add nothing
        self.assertEqual(res["methods_labelled"], 40)
        self.assertEqual(res["categories"], 3)
        drawn = [json.loads((out / "rounds" / f"round_{r:03d}.json").read_text())["method_ids"] for r in range(3)]
        self.assertEqual(len(set(sum(drawn, []))), 40)                  # never the same method twice
        # resume: nothing is sent to the model again, the curve is the same
        again = FakeModel(["Other"])
        run2 = saturation.Run(out, self.s, self.pop, records, lambda m: None)
        self.assertEqual(run2.run(again), res)
        self.assertEqual(again.calls, 0)

    def test_new_category_late_resets_patience(self):
        out = self.tmp / "lab2"
        saturation.prepare(out, dict(self.s), lambda m: None)
        tasks = ["Accessor"] * 28 + ["Visitor", "Visitor"]                # calls 29-30: the end of round 1
        res = saturation.Run(out, self.s, self.pop, records, lambda m: None).run(FakeModel(tasks + ["Accessor"] * 200))
        curve = (out / "curve.csv").read_text().splitlines()
        self.assertTrue(curve[2].startswith("1,10,30,1,Visitor,2,"))
        self.assertEqual(res["rounds"], 3)                              # round 1 adds one, 2 and 3 are quiet

    def test_changed_settings_start_fresh(self):
        out = self.tmp / "lab3"
        s = dict(self.s)
        saturation.prepare(out, s, lambda m: None)
        (out / "plan.json").write_text(json.dumps({"settings_digest": s["settings_digest"]}))
        (out / "labels.jsonl").write_text("")
        saturation.prepare(out, dict(self.s), lambda m: None)            # same settings: kept
        self.assertTrue((out / "labels.jsonl").exists())
        saturation.prepare(out, {**self.s, "min_support": 5}, lambda m: None)   # codebook only: replayed
        self.assertTrue((out / "labels.jsonl").exists())
        saturation.prepare(out, {**self.s, "model": "other"}, lambda m: None)
        self.assertFalse((out / "labels.jsonl").exists())
        self.assertEqual(len(list(self.tmp.glob("lab3.before-*"))), 1)


if __name__ == "__main__":
    unittest.main()
