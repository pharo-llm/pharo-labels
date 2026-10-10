"""Benchmark generation: the acceptance gates (pure), and the whole pipeline in a real Pharo
image. The image tests run only when the validation image exists (it is built by
`pharo-miner benchmark-validate` or ./4-benchmark.sh) and are skipped otherwise."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from pharo_task_miner.benchgen import gates as G
from pharo_task_miner.benchgen.pipeline import Generator, _unique, base_of, finalize, run_validation
from pharo_task_miner.benchgen.synthesis import Synthesizer, pattern_of, render_prompt
from pharo_task_miner.config import DEFAULTS, load_config
from pharo_task_miner.taxonomy.category_manager import load_taxonomy

ROOT = Path(__file__).resolve().parents[1]
BCFG = DEFAULTS["benchmark_generation"]
IMAGE = ROOT / "results" / "benchmark" / "pharo" / "PTMBench.image"

CONTEXT = ["initialize\n\tsuper initialize.\n\tcounts := Dictionary new",
           "add: aWord\n\tcounts at: aWord put: (counts at: aWord ifAbsent: [ 0 ]) + 1"]
TARGET = ("wordsAtLeast: aNumber\n\t^ ((counts associations select: [ :each | each value >= aNumber ])"
          " collect: [ :each | each key ]) sorted")
SETUP = ("setUp\n\tsuper setUp.\n\tcounter := PtbTally new.\n"
         "\t#('pear' 'apple' 'pear' 'fig' 'pear' 'apple') do: [ :w | counter add: w ]")
WEAK_TESTS = ["testNone\n\tself assert: (counter wordsAtLeast: 9) isEmpty",
              "testIsCollection\n\tself assert: (counter wordsAtLeast: 1) isCollection",
              "testNotEmpty\n\tself assert: (counter wordsAtLeast: 1) notEmpty",
              "testEmpty\n\tself assert: (PtbTally new wordsAtLeast: 1) isEmpty"]
STRONG_TESTS = ["testAtLeastTwo\n\tself assert: (counter wordsAtLeast: 2) equals: #('apple' 'pear')",
                "testAtLeastThree\n\tself assert: (counter wordsAtLeast: 3) equals: #('pear')",
                "testAtLeastOneIsSorted\n\tself assert: (counter wordsAtLeast: 1) equals: #('apple' 'fig' 'pear')"]


def task(tests=None):
    return {"task_id": "collection_filtering-01", "operation": "collection_filtering", "package": "PharoTaskBench-X",
            "title": "Tally", "class": {"name": "PtbTally", "superclass": "Object", "instance_variables": ["counts"], "comment": ""},
            "test_class": {"name": "PtbTallyTest", "superclass": "TestCase", "instance_variables": ["counter"]},
            "context_methods": [{"side": "instance", "source": s} for s in CONTEXT],
            "target_methods": [{"side": "instance", "source": TARGET}],
            "tests": [{"source": s} for s in (tests or [SETUP] + WEAK_TESTS)]}


def harness_result(**over):
    names = ["testA", "testB", "testC", "testD"]
    ok = {n: {"outcome": "pass", "detail": ""} for n in names}
    bad = {n: {"outcome": "error", "detail": "Error: not implemented"} for n in names}
    mutants = [{"status": "killed", "operator": "Op", "selector": "wordsAtLeast:", "source": "x"}] * 9 + [
        {"status": "survived", "operator": "Op", "selector": "wordsAtLeast:", "source": "x"}]
    r = {"stage": "done", "problems": [], "image": "Pharo-13", "tests": names, "facts": [],
         "reference_runs": [ok, ok, ok], "all_stubbed_run": bad,
         "stub_runs": [{"class": "PtbTally", "selector": "wordsAtLeast:", "tests": bad}],
         "mutation": {"evaluated": 10, "killed": 9, "terminated": 0, "alive": 1, "mutants": mutants},
         "after_mutation_run": ok, "exported": True}
    r.update(over)
    return r


class GateTests(unittest.TestCase):
    def failed(self, result, t=None):
        ff = G.first_failure(G.evaluate(t or task(), result, BCFG))
        return ff["gate"] if ff else None

    def test_a_clean_run_passes_every_execution_gate(self):
        self.assertIsNone(self.failed(harness_result()))

    def test_each_observation_fails_its_own_gate(self):
        one_fail = dict(harness_result()["reference_runs"][0], testB={"outcome": "failure", "detail": "Got 1 instead of 2."})
        ok = harness_result()["reference_runs"][0]
        self.assertEqual(self.failed(harness_result(stage="install", problems=[
            {"kind": "unimplemented_selector", "class": "PtbTally", "selector": "x", "detail": "frobnicate:"}])), "installs")
        self.assertEqual(self.failed(harness_result(stage="timeout")), "installs")
        self.assertEqual(self.failed(harness_result(facts=[
            {"class": "PtbTally", "selector": "x", "globals": ["DateAndTime"], "sends": ["now"]}])), "hermetic")
        self.assertEqual(self.failed(harness_result(reference_runs=[one_fail])), "reference_passes")
        self.assertEqual(self.failed(harness_result(reference_runs=[ok, one_fail])), "reference_passes")
        self.assertEqual(self.failed(harness_result(stub_runs=[
            {"class": "PtbTally", "selector": "wordsAtLeast:", "tests": ok}])), "fail_to_pass")
        self.assertEqual(self.failed(harness_result(mutation={"evaluated": 10, "killed": 7, "terminated": 0, "alive": 3})), "mutation")
        self.assertEqual(self.failed(harness_result(mutation={"evaluated": 2, "killed": 2, "terminated": 0, "alive": 0})), "mutation")
        self.assertEqual(self.failed(harness_result(mutation={"error": "boom"})), "mutation")
        self.assertEqual(self.failed(harness_result(after_mutation_run=one_fail)), "mutation")

    def test_a_flaky_test_is_reported_as_not_deterministic(self):
        ok = harness_result()["reference_runs"][0]
        flaky = dict(ok, testB={"outcome": "failure", "detail": ""})
        gates = G.evaluate(task(), harness_result(reference_runs=[ok, flaky]), BCFG)
        self.assertIn("not deterministic", G.first_failure(gates)["detail"])

    def test_seeded_random_is_allowed_unseeded_is_not(self):
        fact = {"class": "C", "selector": "s", "globals": ["Random"], "sends": ["next"]}
        self.assertEqual(self.failed(harness_result(facts=[fact])), "hermetic")
        self.assertIsNone(self.failed(harness_result(facts=[dict(fact, sends=["next", "seed:"])])))

    def test_well_formed(self):
        t = task()
        t["tests"] = t["tests"][:3]  # setUp + 2 tests
        self.assertEqual(self.failed(harness_result(), t), "well_formed")
        t = task()
        t["target_methods"] = []
        self.assertEqual(self.failed(harness_result(), t), "well_formed")

    def test_terminated_mutants_count_as_detected(self):
        ms = G.mutation_summary({"mutation": {"evaluated": 10, "killed": 6, "terminated": 2, "alive": 2}})
        self.assertEqual((ms["detected"], ms["score"]), (8, 0.8))

    def test_duplicates_and_seed_copies_are_rejected(self):
        a = {**task(), "status": "accepted", "gates": []}
        b = {**task(), "task_id": "collection_filtering-02", "status": "accepted", "gates": []}
        c = {**task(), "task_id": "collection_filtering-03", "status": "accepted", "gates": []}
        c["class"] = dict(c["class"], name="PtbOther")
        c["test_class"] = dict(c["test_class"], name="PtbOtherTest")
        c["target_methods"] = [{"side": "instance", "source": "wordsAtLeast: n\n\t^ (counts keys select: [ :k | (counts at: k) >= n ]) sorted"}]
        _unique([a, b, c], BCFG, {"collection_filtering-03": c["target_methods"][0]["source"]})
        self.assertEqual([r["status"] for r in (a, b, c)], ["accepted", "rejected", "rejected"])
        self.assertIn("class name", b["reason"])
        self.assertIn("seed", c["reason"])

    def test_prompt_shows_signatures_but_not_the_solution(self):
        t = {**task(), "description": "Return the words seen at least a number of times, sorted."}
        prompt = render_prompt(t)
        self.assertIn("PtbTally >> wordsAtLeast: aNumber", prompt)
        self.assertIn("add: aWord", prompt)
        self.assertNotIn("select:", prompt)
        self.assertEqual(pattern_of("at: k put: v\n\t^ 1"), "at: k put: v")
        self.assertEqual(pattern_of("size\n\t^ 1"), "size")
        self.assertEqual(pattern_of("+ other\n\t^ 1"), "+ other")


class _Answer:
    def __init__(self, parsed):
        self.parsed, self.error, self.served_model, self.usage, self.latency_seconds = parsed, None, "scripted", {}, 0.0


class Scripted:
    """A stand-in model that replays prepared answers per role and records what it was asked."""

    def __init__(self, name, answers):
        self.name = self.model = name
        self.answers, self.asked = {k: list(v) for k, v in answers.items()}, []

    def complete(self, system, user, schema, role="x"):
        self.asked.append((role, user))
        queue = self.answers.get(role) or []
        return _Answer(queue.pop(0) if queue else None)


def author_answer(tests):
    return {"title": "Words seen often", "class_name": "PtbTally", "superclass": "Object", "instance_variables": ["counts"],
            "class_comment": "I count words.", "context_methods": [{"side": "instance", "source": s} for s in CONTEXT],
            "target_methods": [{"side": "instance", "source": TARGET}], "test_class_name": "PtbTallyTest",
            "test_instance_variables": ["counter"], "tests": [{"source": s} for s in tests]}


@unittest.skipUnless(IMAGE.exists(), "validation image not built (run: pharo-miner benchmark-validate --input tests/fixtures/benchmark_tasks.json)")
class ImageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ptm-benchgen-"))
        self.cfg = load_config(ROOT / "config.yaml")
        self.cfg["benchmark_generation"]["dir"] = str(self.tmp / "benchmark")
        self.cfg["project"]["output_dir"] = str(self.tmp)
        (self.tmp / "benchmark").mkdir()
        os.symlink(IMAGE.parent, self.tmp / "benchmark" / "pharo")
        self.tax = load_taxonomy(self.cfg.resolve(self.cfg["classification"]["operations"]["taxonomy"]))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_every_fixture_is_rejected_by_the_gate_it_was_written_for(self):
        tasks = json.loads((ROOT / "tests" / "fixtures" / "benchmark_tasks.json").read_text())["tasks"]
        self.cfg["benchmark_generation"]["concurrency"] = 4
        run_validation(self.cfg, self.tax, tasks, log=lambda m: None)
        home = self.tmp / "benchmark" / "verification"
        got = {json.loads(line)["task_id"]: json.loads(line) for f in ("tasks.jsonl", "rejected.jsonl")
               for line in (home / f).read_text().splitlines()}
        for t in tasks:
            r = got[t["task_id"]]
            self.assertEqual(r["rejected_by"] or "accepted", t["expect"], f"{t['task_id']}: {r['reason']}")
        self.assertTrue((home / "PharoTaskBench" / "src" / "BaselineOfPharoTaskBench").is_dir())

    def test_generation_repairs_adds_tests_and_checks_the_statement(self):
        wrong = [SETUP] + WEAK_TESTS + ["testWrong\n\tself assert: (counter wordsAtLeast: 2) equals: #('pear')"]
        author = Scripted("author", {
            "author": [author_answer(wrong)],                      # a wrong expected value -> reference_passes fails
            "repair": [author_answer([SETUP] + WEAK_TESTS)],       # correct but weak tests -> mutation fails
            "amplify": [{"tests": [{"source": s} for s in STRONG_TESTS]}],
            "describe": [{"description": "wordsAtLeast: returns, as a sorted Array, the words added at least the given number of times."}]})
        solver = Scripted("solver", {"solve": [
            {"methods": [{"side": "instance", "source": "wordsAtLeast: n\n\t^ (counts keys select: [ :k | (counts at: k) > n ]) sorted"}]},
            {"methods": [{"side": "instance", "source": "wordsAtLeast: n\n\t^ (counts keys select: [ :k | (counts at: k) >= n ]) sorted asArray"}]}]})
        bcfg = self.cfg["benchmark_generation"]
        home = self.tmp / "benchmark"
        synth = Synthesizer(author, [solver], bcfg, home / "llm_cache", lambda op: "Selecting elements by a condition.")
        from pharo_task_miner.benchgen.image import ensure_image
        pharo = ensure_image(self.cfg, log=lambda m: None)
        gen = Generator(self.cfg, pharo, synth, self.tax, home, log=lambda m: None)
        seed = {"operation": "collection_filtering", "index": 1, "method_id": 1, "repository": "o/r", "license": "MIT",
                "commit_sha": "abc", "path": "src/A.class.st", "line": 3, "class_name": "A", "side": "instance",
                "selector": "evens", "source": "evens\n\t^ items select: [ :e | e even ]"}
        rec = gen.generate(seed)
        self.assertEqual(rec["status"], "accepted", rec["reason"])
        self.assertEqual([h["failed"] for h in rec["history"]], ["reference_passes", "mutation", None, None])
        self.assertEqual((rec["repairs"], rec["amplifications"]), (1, 1))
        self.assertEqual(rec["validation"]["mutation"]["score"], 1.0)
        self.assertEqual(rec["independent_solution"]["model"], "solver")
        self.assertNotIn("select:", [u for r, u in solver.asked][0])  # the solver never saw the reference
        m = finalize(self.cfg, pharo, [rec], {base_of(seed, bcfg)["task_id"]: seed["source"]}, {"author": "author", "solvers": ["solver"]},
                     home, self.tmp / "report.md", log=lambda m: None)
        self.assertEqual((m["accepted"], m["rejected"]), (1, 0))
        self.assertTrue(all(g["passed"] for g in rec["gates"]))
        self.assertEqual([g["gate"] for g in rec["gates"]], G.GATE_IDS)


if __name__ == "__main__":
    unittest.main()
