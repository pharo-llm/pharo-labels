"""Unit tests for the labelling: sample size, draws, answer parsing, codebook, saturation loop."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharo_task_miner.labelling import coding, cost, sampling, saturation  # noqa: E402


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

    def test_character_cap_cuts_long_lines(self):
        m = {"class_name": "A", "source": "data\n\t^ #(" + "1 " * 10000 + ")"}
        self.assertEqual(coding.build_prompt(m, 40), coding.prefix() + coding.method_block(m, 40))
        cut = coding.method_source(m, 40, 100)
        self.assertLessEqual(len(cut), 100 + len('\n\t"..."'))
        self.assertTrue(cut.endswith('\t"..."'))
        short = {"class_name": "A", "source": "x\n\t^ x"}
        self.assertEqual(coding.method_source(short, 40, 100), coding.method_source(short, 40))   # unchanged


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

    def test_answers_made_with_another_prompt_are_redone_in_place(self):
        out = self.tmp / "lab4"
        s = dict(self.s)
        saturation.prepare(out, s, lambda m: None)
        recs = lambda ids: {m: {**r, "source": r["source"] + ("x" * 500 if m % 7 == 0 else "")}
                            for m, r in records(ids).items()}
        first = saturation.Run(out, s, self.pop, recs, lambda m: None)
        res = first.run(FakeModel(["Accessor", "Parsing"]))
        long = sorted(m for m in first.labels if m % 7 == 0)
        self.assertTrue(long)
        for name in ("labels.jsonl",):                             # as stored before prompts had a hash
            rows = [json.loads(x) for x in (out / name).read_text().splitlines()]
            (out / name).write_text("".join(json.dumps({k: v for k, v in r.items() if k != "prompt_sha"}) + "\n"
                                            for r in rows))
        capped = {**s, "max_method_chars": 100}
        again = FakeModel(["Accessor", "Parsing"])
        run2 = saturation.Run(out, capped, self.pop, recs, lambda m: None)
        self.assertEqual(run2.run(again), res)                     # same rounds, same taxonomy
        self.assertEqual(again.calls, len(long))                   # only the methods whose prompt changed
        rows = [json.loads(x) for x in (out / "labels.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), res["methods_labelled"])
        self.assertEqual(sorted(r["method_id"] for r in rows if "prompt_sha" in r), long)
        run3 = saturation.Run(out, capped, self.pop, recs, lambda m: None)
        none = FakeModel(["Other"])
        run3.run(none)
        self.assertEqual(none.calls, 0)                            # nothing stale any more

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


class CostTests(unittest.TestCase):
    c = {"chat_overhead_tokens": 20, "batch_discount": 0.5, "reasoning_tokens_per_call": 250,
         "cache_min_prefix_tokens": 1024, "budget_factor": [1.5, 2.0]}

    def test_price_of_a_scenario(self):
        sc = cost.scenario("x", 1000, 100_000, 16_000, 700, self.c)    # 1,000 calls of 720 + 100 tokens
        m = {"name": "M", "input": 10.0, "cached_input": 1.0, "output": 50.0}
        p = cost.price(sc, m, self.c)
        self.assertAlmostEqual(p["input_mtok"], 0.82)
        self.assertAlmostEqual(p["standard"], 0.82 * 10 + 0.016 * 50)
        self.assertAlmostEqual(p["batch"], p["standard"] / 2)
        self.assertAlmostEqual(p["reasoning"], p["standard"] + 0.25 * 50)
        self.assertAlmostEqual(p["cached"], 1.024 * 1 + 0.1 * 10 + 0.016 * 50)   # prefix padded to 1,024
        self.assertAlmostEqual(p["budget_low"], p["batch"] * 1.5)
        self.assertAlmostEqual(p["budget_high"], p["reasoning_batch"] * 2)
        claude = cost.price(sc, {**m, "token_factor": 1.25}, self.c)
        self.assertAlmostEqual(claude["standard"], p["standard"] * 1.25)
        self.assertIsNone(cost.price(sc, {**m, "cached_input": None}, self.c)["cached"])

    def test_tokenizer_counts_something(self):
        count, name = cost.tokenizer()
        self.assertTrue(name)
        self.assertGreater(count(coding.prefix()), 500)
        self.assertEqual(count(""), 0)



class AgreementTests(unittest.TestCase):
    """The statistics of the deductive phase, against published values and a simulation."""

    def test_krippendorff_alpha_reference_example(self):
        from pharo_task_miner.labelling import agreement as ag
        # Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability": 4 coders, 12 units, missing data
        A = [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None]
        B = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3]
        C = [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None]
        D = [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None]
        self.assertAlmostEqual(ag.krippendorff_alpha([list(u) for u in zip(A, B, C, D)]), 0.743, places=3)

    def test_fleiss_kappa_reference_example(self):
        from pharo_task_miner.labelling import agreement as ag
        # Fleiss (1971) as worked on Wikipedia: 10 items, 14 raters, 5 categories, kappa = 0.210
        counts = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
                  [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
        units = [[c for c, k in enumerate(row) for _ in range(k)] for row in counts]
        self.assertAlmostEqual(ag.fleiss_kappa(units), 0.210, places=3)

    def test_ac1_stays_high_when_one_category_dominates(self):
        from pharo_task_miner.labelling import agreement as ag
        units = [["A", "A", "A"]] * 95 + [["A", "B", "A"]] * 4 + [["B", "B", "B"]]
        self.assertLess(ag.fleiss_kappa(units), 0.5)                  # the kappa paradox
        self.assertGreater(ag.gwet_ac1(units), 0.9)

    def test_majority_levels(self):
        from pharo_task_miner.labelling import agreement as ag
        self.assertEqual(ag.majority(["A", "A", "A"]), ("A", "3/3"))
        self.assertEqual(ag.majority(["A", "B", "A"]), ("A", "2/3"))
        self.assertEqual(ag.majority(["A", "B", "C"]), (None, "none"))
        self.assertEqual(ag.majority(["A", None, "A"]), ("A", "2/3"))

    def test_ppi_share_is_unbiased_and_covers(self):
        """Models that over-label category A (true share 30%, models say ~41%): the corrected share
        is unbiased and its 95% interval covers the truth about 95% of the time; the models'
        share alone is biased."""
        from pharo_task_miner.labelling import agreement as ag
        import random as rnd
        rng = rnd.Random(7)
        n, reps, hits, ests, naive = 400, 400, 0, [], []
        for _ in range(reps):
            truth = [rng.random() < 0.30 for _ in range(n)]
            pred = [t if rng.random() < 0.85 else (rng.random() < 0.7) for t in truth]
            level = ["3/3" if rng.random() < 0.7 else "2/3" for _ in range(n)]
            strata = {h: {"N": level.count(h), "m": 0} for h in ("3/3", "2/3")}
            diffs = {}
            for h, m in (("3/3", 40), ("2/3", 60)):
                idx = rng.sample([i for i in range(n) if level[i] == h], m)
                strata[h]["m"] = m
                diffs[h] = [int(truth[i]) - int(pred[i]) for i in idx]
            r = ag.ppi_share(sum(pred) / n, n, strata, diffs, 0.95)
            ests.append(r["share"])
            naive.append(r["model_share"])
            hits += r["ci_low"] <= 0.30 <= r["ci_high"]
        self.assertAlmostEqual(sum(ests) / reps, 0.30, delta=0.01)
        self.assertGreater(abs(sum(naive) / reps - 0.30), 0.05)
        self.assertGreater(hits / reps, 0.92)

    def test_stratified_accuracy(self):
        from pharo_task_miner.labelling import agreement as ag
        strata = {"3/3": {"N": 300, "m": 50}, "2/3": {"N": 100, "m": 50}}
        r = ag.stratified_mean(strata, {"3/3": [1.0] * 45 + [0.0] * 5, "2/3": [1.0] * 30 + [0.0] * 20})
        self.assertAlmostEqual(r["estimate"], 0.75 * 0.9 + 0.25 * 0.6, places=4)   # not the raw 75/100
        self.assertLess(r["ci_low"], r["estimate"])


if __name__ == "__main__":
    unittest.main()
