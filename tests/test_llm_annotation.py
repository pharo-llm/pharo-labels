"""LLM annotation panel: answer interpretation, aggregation, providers (against
local mock servers) and an end-to-end run with fake models."""
import csv
import http.server
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pharo_task_miner.cli import main  # noqa: E402
from pharo_task_miner.validation import llm_providers as lp  # noqa: E402
from pharo_task_miner.validation.llm_annotation import (Prompt, aggregate, compare, guideline_sections,  # noqa: E402
                                                        interpret)
from pharo_task_miner.taxonomy.category_manager import load_taxonomy  # noqa: E402
from test_ptm_pipeline import ROOT, make_repo  # noqa: E402

TAX = load_taxonomy(ROOT / "taxonomy" / "initial_categories.yaml")
GUIDE = (ROOT / "taxonomy" / "annotation_guidelines.md").read_text()
ITEM = {"sample_id": "S00001", "method_id": "1", "repository": "a/b", "repository_domains": "", "class": "Bag",
        "selector": "positives", "protocol": "operations", "source": "positives\n\t^ self items select: [ :x | x > 0 ]"}


def answer(ops=(), special="none", **kw):
    base = {"rationale": "r", "operations": [{"id": i, "evidence": e} for i, e in ops], "special_status": special,
            "new_category": "", "new_category_description": "", "too_broad": False, "too_narrow": False,
            "multiple_operations": False, "evidence_sufficient": "yes", "missing_operations": "", "notes": ""}
    base.update(kw)
    return base


class PromptTest(unittest.TestCase):
    def test_prompt_uses_human_guidelines_and_all_categories(self):
        p = Prompt(GUIDE, TAX)
        level3 = [c["id"] for c in TAX["categories"] if c["level"] == 3]
        self.assertEqual(sorted(p.schema["properties"]["operations"]["items"]["properties"]["id"]["enum"]), sorted(level3))
        self.assertIn("## Decision rules", p.system)
        self.assertIn("## For every method", p.system)
        self.assertNotIn("## Files", p.system)          # file-handling instructions are for humans only
        for cid in level3:
            self.assertIn(f"`{cid}`", p.system)
        self.assertIn("Excludes: Searching for one matching element", p.system)
        self.assertIn("select: [ :x | x > 0 ]", p.user(ITEM))
        self.assertEqual(len(p.sha256), 64)
        self.assertEqual(p.sha256, Prompt(GUIDE, TAX).sha256)  # stable -> cache keys are stable

    def test_guideline_sections_fallback(self):
        self.assertEqual(guideline_sections("no headings"), "no headings")


class InterpretTest(unittest.TestCase):
    ids = {"collection_filtering", "collection_sorting"}

    def test_evidence_is_enforced(self):
        d = interpret(answer([("collection_filtering", ["select:"]), ("collection_sorting", ["sortBlock"])]), ITEM, self.ids)
        self.assertEqual(d["labels"], ["collection_filtering"])
        self.assertEqual(d["rejected"][0]["id"], "collection_sorting")

    def test_record_policy_keeps_labels(self):
        d = interpret(answer([("collection_sorting", ["sortBlock"])]), ITEM, self.ids, "record")
        self.assertEqual(d["labels"], ["collection_sorting"])

    def test_whitespace_insensitive_evidence(self):
        d = interpret(answer([("collection_filtering", ["self items   select:"])]), ITEM, self.ids)
        self.assertEqual(d["labels"], ["collection_filtering"])

    def test_special_statuses(self):
        self.assertEqual(interpret(answer(special="insufficient_context"), ITEM, self.ids)["labels"], ["insufficient_context"])
        d = interpret(answer([("collection_filtering", ["select:"])], special="unknown"), ITEM, self.ids)
        self.assertEqual(d["labels"], ["collection_filtering"])
        d = interpret(answer([("collection_sorting", ["nope"])]), ITEM, self.ids)
        self.assertEqual(d["labels"], ["unknown"])
        self.assertTrue(d["interpretation_notes"])

    def test_unknown_id_rejected(self):
        d = interpret(answer([("made_up", ["select:"])]), ITEM, self.ids)
        self.assertEqual(d["rejected"][0]["reason"], "not a taxonomy category")


class AggregateTest(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(aggregate({"a": {"x", "y"}, "b": {"y", "x"}}), ({"x", "y"}, "agreed", {"x": 2, "y": 2}))
        labels, status, _ = aggregate({"a": {"x", "y"}, "b": {"x"}, "c": {"x", "z"}})
        self.assertEqual((labels, status), ({"x"}, "majority"))
        labels, status, _ = aggregate({"a": {"x"}, "b": {"y"}})
        self.assertEqual((labels, status), (set(), "no_consensus"))
        labels, status, _ = aggregate({"a": {"x", "y"}, "b": {"x"}, "c": {"x", "y"}}, "unanimous")
        self.assertEqual(labels, {"x"})
        self.assertEqual(aggregate({"a": {"x"}})[1], "single_annotation")
        self.assertEqual(aggregate({})[1], "incomplete")
        # a special status loses against a real operation chosen by the majority
        labels, _, _ = aggregate({"a": {"x"}, "b": {"x", "unknown"}, "c": {"unknown"}})
        self.assertEqual(labels, {"x"})

    def test_compare(self):
        ref = {"1": {"a"}, "2": {"a", "b"}, "3": {"c"}}
        res = compare(ref, {"1": {"a"}, "2": {"a"}, "3": {"c"}}, ["a", "b", "c"])
        self.assertEqual(res["items"], 3)
        self.assertEqual(res["micro_precision"], 1.0)
        self.assertEqual(res["micro_recall"], 0.75)
        self.assertAlmostEqual(res["mean_jaccard"], round((1 + 0.5 + 1) / 3, 3))


# --------------------------------------------------------------------------- providers vs mock HTTP servers
class _Handler(http.server.BaseHTTPRequestHandler):
    script: list = []
    seen: list = []

    def log_message(self, *a):
        pass

    def _reply(self):
        status, body = self.script.pop(0)
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._reply()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.seen.append({"path": self.path, "auth": self.headers.get("Authorization"),
                          "body": json.loads(self.rfile.read(n) or b"{}")})
        self._reply()


class ProviderTest(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        _Handler.script, _Handler.seen = [], []
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self._sleep = lp._Base._sleep
        lp._Base._sleep = staticmethod(lambda attempt: None)

    def tearDown(self):
        lp._Base._sleep = self._sleep
        self.srv.shutdown()
        self.srv.server_close()

    def openai_reply(self, content, finish="stop"):
        return 200, {"id": "r1", "model": "served-x", "choices": [{"message": {"content": content}, "finish_reason": finish}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    def test_openai_compatible_retries_then_succeeds(self):
        os.environ["PTM_TEST_KEY"] = "k"
        _Handler.script = [(429, {"error": "slow down"}), self.openai_reply("not json"), self.openai_reply('{"a": 1}')]
        p = lp.make_provider({"provider": "openai_compatible", "base_url": self.url + "/v1", "model": "m",
                              "api_key_env": "PTM_TEST_KEY", "profile": "t"})
        c = p.complete("sys", "user", {"type": "object"})
        self.assertIsNone(c.error)
        self.assertEqual(c.parsed, {"a": 1})
        self.assertEqual(c.attempts, 3)
        self.assertEqual(c.served_model, "served-x")
        self.assertEqual(_Handler.seen[-1]["auth"], "Bearer k")
        self.assertEqual(_Handler.seen[-1]["path"], "/v1/chat/completions")
        self.assertEqual(_Handler.seen[-1]["body"]["response_format"]["type"], "json_schema")

    def test_openai_compatible_bad_key_aborts(self):
        os.environ["PTM_TEST_KEY"] = "k"
        _Handler.script = [(401, {"error": "bad key"})]
        p = lp.make_provider({"provider": "openai_compatible", "base_url": self.url, "model": "m",
                              "api_key_env": "PTM_TEST_KEY", "profile": "t"})
        with self.assertRaises(lp.ProviderConfigError):
            p.complete("sys", "user", {"type": "object"})

    def test_missing_key_is_reported_before_any_request(self):
        os.environ.pop("PTM_ABSENT_KEY", None)
        with self.assertRaises(lp.ProviderConfigError):
            lp.make_provider({"provider": "openai_compatible", "base_url": self.url, "model": "m",
                              "api_key_env": "PTM_ABSENT_KEY", "profile": "t"})

    def test_truncated_output_is_an_error(self):
        _Handler.script = [self.openai_reply('{"a":', "length")] * 2
        p = lp.make_provider({"provider": "openai_compatible", "base_url": self.url, "model": "m", "profile": "t",
                              "max_attempts": 2})
        c = p.complete("sys", "user", {"type": "object"})
        self.assertIn("truncated", c.error)

    def test_ollama(self):
        _Handler.script = [(200, {"models": [{"name": "qwen:7b"}]}),
                           (200, {"model": "qwen:7b", "message": {"content": '{"ok": true}'}, "done_reason": "stop",
                                  "prompt_eval_count": 50, "eval_count": 5})]
        p = lp.make_provider({"provider": "ollama", "base_url": self.url, "model": "qwen:7b", "profile": "q", "num_ctx": 1024})
        c = p.complete("s" * 6000, "u", {"type": "object"})
        self.assertEqual(c.parsed, {"ok": True})
        opts = _Handler.seen[-1]["body"]["options"]
        self.assertGreater(opts["num_ctx"], 1024)      # enlarged so the prompt is not silently truncated
        self.assertEqual(opts["temperature"], 0)

    def test_ollama_model_not_installed(self):
        _Handler.script = [(200, {"models": [{"name": "other:1b"}]})]
        with self.assertRaises(lp.ProviderConfigError):
            lp.make_provider({"provider": "ollama", "base_url": self.url, "model": "qwen:7b", "profile": "q"})


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class AnthropicProviderTest(unittest.TestCase):
    def make(self, responses, **prof):
        os.environ["PTM_ANTHROPIC_TEST"] = "sk-test"
        p = lp.make_provider({"provider": "anthropic", "model": "claude-opus-5-5", "profile": "c",
                              "api_key_env": "PTM_ANTHROPIC_TEST", "effort": "high", **prof})
        calls = []

        def create(**kw):
            calls.append(kw)
            return responses.pop(0)
        p.client = _Obj(messages=_Obj(create=create), beta=_Obj(messages=_Obj(create=create)))
        return p, calls

    def resp(self, text, stop="end_turn"):
        return _Obj(content=[_Obj(type="thinking", thinking=""), _Obj(type="text", text=text)], model="claude-opus-5-5",
                    stop_reason=stop, stop_details=None,
                    usage=_Obj(input_tokens=10, output_tokens=3, cache_read_input_tokens=9, cache_creation_input_tokens=0))

    def test_structured_output_request(self):
        p, calls = self.make([self.resp('{"x": 1}')])
        c = p.complete("system prompt", "user", {"type": "object"})
        self.assertEqual(c.parsed, {"x": 1})
        kw = calls[0]
        self.assertEqual(kw["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(kw["output_config"]["effort"], "high")
        self.assertEqual(kw["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("temperature", kw)           # current Claude models reject sampling parameters
        self.assertEqual(c.usage["cache_read_input_tokens"], 9)

    def test_fallbacks_use_beta_endpoint(self):
        p, calls = self.make([self.resp('{"x": 1}')], fallbacks="default")
        p.complete("s", "u", {"type": "object"})
        self.assertEqual(calls[0]["fallbacks"], "default")
        self.assertEqual(calls[0]["betas"], ["server-side-fallback-2026-07-01"])

    def test_refusal_is_not_retried(self):
        p, calls = self.make([self.resp("", "refusal")])
        c = p.complete("s", "u", {"type": "object"})
        self.assertIn("refusal", c.error)
        self.assertEqual(len(calls), 1)


# --------------------------------------------------------------------------- end to end with fake models
class FakeProvider(lp._Base):
    """Deterministic 'model': labels from selectors present in the user prompt."""
    provider = "fake"

    def _call(self, system, user, schema, schema_name):
        src = user.split("```smalltalk", 1)[1]
        ops = []
        if "select:" in src:
            ops.append({"id": "collection_filtering", "evidence": ["select:"]})
        if "inject:" in src and self.profile.get("sees_aggregation", True):
            ops.append({"id": "collection_aggregation", "evidence": ["inject:into:" if "inject:into:" in src else "inject:"]})
        if "printOn:" in src:
            ops.append({"id": "object_printing", "evidence": ["printOn:"]})
        if self.profile.get("hallucinate"):
            ops.append({"id": "http_networking", "evidence": ["ZnClient get"]})  # not in the source -> rejected
        if schema_name == "adjudication":
            ops = [o for o in ops if o["id"] != "collection_aggregation"] or ops
        body = {"rationale": f"{self.name} rationale", "operations": ops,
                "special_status": "none" if ops else "insufficient_context", "new_category": "",
                "new_category_description": "", "too_broad": False, "too_narrow": False, "multiple_operations": len(ops) > 1,
                "evidence_sufficient": "yes", "missing_operations": "", "notes": ""}
        return lp.Completion(parsed=None, raw_text=json.dumps(body), served_model=self.model, stop_reason="stop",
                             usage={"input_tokens": len(system + user) // 3, "output_tokens": 50})


class EndToEndTest(unittest.TestCase):
    def test_panel_to_report(self):
        warnings.simplefilter("ignore", ResourceWarning)
        lp.PROVIDERS["fake"] = FakeProvider
        try:
            self._run()
        finally:
            lp.PROVIDERS.pop("fake", None)

    def _run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            clones = tmp / "clones"
            repos = [("acme", "WebShop", "A web application framework for HTTP REST services", ["seaside"]),
                     ("labx", "ChartKit", "A visualization library for charts with Roassal", ["visualization"])]
            dump = []
            for i, (owner, name, desc, topics) in enumerate(repos, 1):
                make_repo(clones, owner, name, desc)
                dump.append({"id": 1000 + i, "full_name": f"{owner}/{name}", "name": name, "owner": {"login": owner},
                             "html_url": f"https://github.com/{owner}/{name}", "description": desc, "topics": topics,
                             "language": "Smalltalk", "stargazers_count": i, "forks_count": 0, "fork": False,
                             "archived": False, "created_at": "2020-01-01T00:00:00Z", "pushed_at": "2025-01-01T00:00:00Z",
                             "default_branch": "main"})
            (tmp / "dump.json").write_text(json.dumps(dump))
            (tmp / "models.yaml").write_text(json.dumps({"default": "m1", "models": {
                "m1": {"provider": "fake", "model": "fake-1"},
                "m2": {"provider": "fake", "model": "fake-2", "sees_aggregation": False},
                "m3": {"provider": "fake", "model": "fake-3", "hallucinate": True},
                "judge": {"provider": "fake", "model": "fake-judge"}}}))
            cfg = {
                "project": {"data_dir": str(tmp / "data"), "output_dir": str(tmp / "out"), "validation_dir": str(tmp / "val"),
                            "database": str(tmp / "data" / "db.sqlite")},
                "repositories": {"existing_clone_dirs": [str(clones)], "clone_dir": str(tmp / "data" / "repos")},
                "classification": {"domains": {"definitions": str(ROOT / "taxonomy" / "domains.yaml")},
                                   "operations": {"taxonomy": str(ROOT / "taxonomy" / "initial_categories.yaml")},
                                   "frameworks": str(ROOT / "taxonomy" / "frameworks.yaml")},
                "validation": {"annotation_sample_size": 10, "overlap_fraction": 1.0},
                "semantic": {"models_file": str(tmp / "models.yaml")},
                "llm_annotation": {"guidelines": str(ROOT / "taxonomy" / "annotation_guidelines.md"), "spotcheck_size": 4},
                "statistics": {"bootstrap_iterations": 20},
            }
            (tmp / "config.json").write_text(json.dumps(cfg))
            c = ["--config", str(tmp / "config.json")]

            def run(*a):
                return main([a[0], *c, *a[1:]])

            for step in (("discover", "--import-dump", str(tmp / "dump.json")), ("verify", "--local"), ("classify-domains",),
                         ("sample", "--strategy", "all"), ("clone", "--local-only"), ("parse",), ("analyze",),
                         ("discover-operations",), ("export-annotations",)):
                self.assertEqual(run(*step), 0, step)
            self.assertEqual(run("build-taxonomy"), 3)                  # gate closed before labelling
            self.assertEqual(run("llm-annotate"), 3)                    # no models chosen -> clear message
            self.assertEqual(run("llm-annotate", "--models", "nope"), 3)  # unknown model

            self.assertEqual(run("llm-annotate", "--models", "m1 m2 m3", "--adjudicator", "judge"), 0)
            vdir, odir = tmp / "val", tmp / "val" / "llm"
            n_items = len(list(csv.DictReader((vdir / "annotation_sample.csv").open())))
            for f in ("llm_decisions.csv", "llm_adjudicated_labels.csv", "run_manifest.json", "agreement_report.json",
                      "prompt_system.md", "output_schema.json", "human_spotcheck.csv", "failures.csv"):
                self.assertTrue((odir / f).exists(), f)
            decisions = list(csv.DictReader((odir / "llm_decisions.csv").open()))
            self.assertEqual(len(decisions), 3 * n_items)
            self.assertEqual({d["annotator"] for d in decisions}, {"llm:m1", "llm:m2", "llm:m3"})
            self.assertFalse(any("http_networking" in d["operations"] for d in decisions))  # invented label rejected
            man = json.loads((odir / "run_manifest.json").read_text())
            self.assertGreater(man["per_model"]["llm:m3"]["labels_rejected"], 0)
            self.assertEqual(man["aggregation"], "adjudicator")
            final = {r["sample_id"]: r for r in csv.DictReader((odir / "llm_adjudicated_labels.csv").open())}
            self.assertEqual(len(final), n_items)
            statuses = {r["status"] for r in final.values()}
            self.assertIn("agreed", statuses)
            self.assertIn("adjudicated", statuses)                     # m2 never sees inject: -> disagreement
            spot = list(csv.DictReader((odir / "human_spotcheck.csv").open()))
            self.assertEqual(len(spot), 4)
            self.assertNotIn("llm", ",".join(spot[0].keys()))          # blind: no AI labels in the human file

            # rerun: everything comes from the cache (fake provider would be called otherwise)
            calls = []
            orig = FakeProvider._call
            FakeProvider._call = lambda self, *a: calls.append(1) or orig(self, *a)
            try:
                self.assertEqual(run("llm-annotate", "--models", "m1 m2 m3", "--adjudicator", "judge"), 0)
            finally:
                FakeProvider._call = orig
            self.assertEqual(calls, [])

            # the gate opens on LLM labels, which are reported as such
            self.assertEqual(run("build-taxonomy"), 0)
            tax = json.loads((vdir / "taxonomy_v1-llm.json").read_text())
            self.assertEqual(tax["validation_basis"], "llm_panel_annotated_sample_NOT_human_validated")
            self.assertFalse(any(cat["status"] == "validated" for cat in tax["categories"]))
            self.assertTrue(any(cat["status"] == "llm_supported" for cat in tax["categories"]))
            self.assertFalse((vdir / "taxonomy_v1.json").exists())
            self.assertEqual(run("statistics"), 0)
            stats = json.loads((tmp / "data" / "processed" / "statistics.json").read_text())
            self.assertEqual(stats["uncertainty"]["label_basis"], "llm_panel_annotated")

            # human spot-check: a person agrees with the final labels on all but one item
            for i, r in enumerate(spot):
                ops = final[r["sample_id"]]["operations"]
                r["operations"] = "object_printing" if i == 0 else ops
            with (odir / "human_spotcheck.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(spot[0].keys()))
                w.writeheader()
                w.writerows(spot)
            self.assertEqual(run("llm-spotcheck"), 0)
            rep = json.loads((odir / "spotcheck_report.json").read_text())
            fin = rep["comparisons"]["human_1 vs llm:final_labels"]
            self.assertEqual(fin["items"], 4)
            self.assertLess(fin["exact_set_agreement"], 1.0)
            self.assertEqual(run("report"), 0)
            summary = (tmp / "out" / "reports" / "ecosystem_summary.md").read_text()
            self.assertIn("LLM-annotated sample — not human-validated", summary)
            self.assertIn("Human spot-check", summary)

            # label_source: human -> LLM labels are ignored and the gate closes again
            cfg["validation"]["label_source"] = "human"
            (tmp / "config.json").write_text(json.dumps(cfg))
            self.assertEqual(run("build-taxonomy"), 3)


if __name__ == "__main__":
    unittest.main()
