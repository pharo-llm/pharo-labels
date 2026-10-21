"""Phase 10 — Pharo benchmark construction.

1. **Behaviours to cover**: ``selected_behaviors.csv`` in this folder when the research team wrote one
   (column ``behavior_id``, optional ``reason``); otherwise chosen from the coverage matrix (phase 9):
   statuses ``benchmark.statuses`` first (default: Pharo-only, then both), most frequent in Pharo first
   (with the experts' relevance as tie-break when available) → ``auto_selected_behaviors.csv`` with the
   reason of each choice.
2. **Seeds**: real Pharo methods labelled with the behaviour (phases 6 and 8), not tests, 3–30 statements,
   one per repository; and a literature task with the same behaviour when one exists (provenance: what
   existing benchmarks ask for this behaviour).
3. **Authoring**: the author model writes a NEW self-contained task in the same spirit: class, context
   methods, reference implementation (target methods), SUnit tests (≥ pharo.min_tests), and the
   natural-language specification last (from the code; reverse-order generation).
4. **Execute and repair**: each candidate is installed and its tests run in the isolated image; when it
   fails, the author sees the image's report and rewrites it (``benchmark.repair_rounds``).

Each candidate: task ID, behaviour IDs (+ category, subcategory), evaluation task types (code generation,
plus code completion and bug repair variants derived in phase 11), specification, provenance (seed method
with repository, commit and link; literature example), reference implementation, tests, target Pharo
version, validation status.

Outputs: benchmark_candidates.jsonl, tasks/<task_id>/ (task.json, statement.md, reference.st, tests.st),
auto_selected_behaviors.csv, figures: funnel.png, candidates_per_category.png.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import pandas as pd

from .. import figures as F
from .. import pharo as P
from .. import taxonomy as T
from ..common import read_jsonl, write_csv, write_json, write_jsonl, write_text
from ..llm import Client, ledger_summary
from .p08_profile import latest_taxonomy

PROMPT_VERSION = "author-v3"

# a method is an array of lines: inside one JSON string, models squeeze code onto one line (with "|" as separator)
_M = {"type": "object", "properties": {"lines": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
      "required": ["lines"]}


def src(m: dict) -> str:
    """The method source of an answer item: its lines joined (an old-style "source" string is accepted too)."""
    if m.get("lines"):
        return "\n".join(str(x).rstrip() for x in m["lines"]).strip()
    return str(m.get("source", "")).strip()
SCHEMA = {"type": "object", "properties": {
    "class_name": {"type": "string"}, "instance_variables": {"type": "array", "items": {"type": "string"}},
    "class_comment": {"type": "string"}, "context_methods": {"type": "array", "items": _M},
    "target_methods": {"type": "array", "items": _M, "minItems": 1, "maxItems": 2},
    "tests": {"type": "array", "items": _M, "minItems": 3, "maxItems": 10},
    "statement": {"type": "string", "minLength": 40}},
    "required": ["class_name", "instance_variables", "class_comment", "context_methods", "target_methods", "tests", "statement"]}

AUTHOR = """You write programming tasks for a Pharo 13 benchmark. Each task exercises ONE programming behaviour that real Pharo projects use, inspired by a real method (the seed), but it must be NEW and self-contained.

A task is one small class, one or two methods a model will have to implement (target methods), context methods given to the solver, and SUnit tests that decide whether an implementation is correct. Your answer is loaded into a clean Pharo 13 image and run; it is discarded if anything fails.
Rules:
- Only classes and messages of a standard Pharo 13 image. No file, network, clock (now, today), random value, Smalltalk globals, and never open a window. Do NOT reuse the seed's project classes.
- EXACTLY ONE new class (yours). Never refer to any other class you would have to create (no PBRNode, no helper class): represent nested data with Dictionary, Array, OrderedCollection, Set, Association, String, Symbol, Number.
- Pharo syntax: temporaries are declared once as `| a b |` on the line after the message pattern; statements end with a period; `^` returns; blocks are `[ :x | ... ]`; cascades use `;`.
- The class name starts with "PBR". The test class is generated: a subclass of PBTestCase whose instance variable `task` holds a fresh instance of your class (task := YourClass new). Tests use `task`, assert:, assert:equals:, deny:, should:raise:.
- Every method is given as "lines": one array element per line of code. The first line is the message pattern, then the temporaries line, then one statement per line. Never put several lines in one element, never use "|" to separate lines. No class name, no ">>", no Tonel brackets.
- target_methods: your reference solution (the solver sees only their message pattern). Real logic (several statements, a condition, a loop or a collection operation), never a bare accessor.
- context_methods: initialize, accessors, helpers the solver is given as they are.
- tests: MINT to MAXT test methods named test..., covering normal cases, edge cases and errors; each must fail if the target method is missing or wrong.
- statement: written last, from your code: what the target method(s) must do, inputs, outputs, edge cases, in plain English, without code of the solution."""


def select_behaviors(ph, tax) -> pd.DataFrame:
    team = ph.path("selected_behaviors.csv")
    if team.exists():
        sel = pd.read_csv(team)
        sel["reason"] = sel.get("reason", "selected by the research team")
        ph.log.info(f"behaviours selected by the team: {len(sel)} (selected_behaviors.csv)")
        return sel
    cov = pd.read_csv(ph.upstream(9, "coverage_matrix.csv"))
    rel_p = ph.upstream(7, "behavior_relevance.csv")
    if rel_p.exists():
        r = pd.read_csv(rel_p)
        cov = cov.merge(r[["behavior_id", "relevance_benchmark"]], on="behavior_id", how="left")
    else:
        cov["relevance_benchmark"] = None
    statuses = list(ph.cfg["benchmark"]["statuses"])
    # rank by the status from the counts (support_status); the agreement-aware status is reported with it
    scol = "support_status" if "support_status" in cov else "status"
    cov["rank_status"] = cov[scol].map(lambda s: statuses.index(s) if s in statuses else len(statuses))
    cov["observed_pharo"] = cov["pharo_profile_methods"] + cov["pharo_annotation_methods"]
    cov = cov[cov["observed_pharo"] > 0].sort_values(
        ["rank_status", "pharo_weighted_share", "relevance_benchmark", "observed_pharo"], ascending=[True, False, False, False])
    k = math.ceil(int(ph.sizes["benchmark_candidates"]) / int(ph.cfg["benchmark"]["seeds_per_behavior"]))
    sel = cov.head(k).copy()
    sel["reason"] = sel.apply(lambda r: (f"coverage {scol} {r[scol]}" + ("" if r[scol] in statuses else " (filler: not enough behaviours in "
                                         f"{statuses})")) + f" (agreement-aware status: {r.status}); Pharo weighted share "
                              f"{r.pharo_weighted_share}", axis=1)
    write_csv(ph.path("auto_selected_behaviors.csv"), sel[["behavior_id", "behavior", "category", "status", "pharo_weighted_share",
                                                           "literature_tasks", "relevance_benchmark", "reason"]])
    return sel


def seeds_for(ph, bid: str, methods: pd.DataFrame, labels: pd.DataFrame, k: int, used_repos: set) -> list[dict]:
    ids = labels[labels["behavior_ids"].map(lambda xs: bid in xs)]["method_id"]
    cand = methods[methods["method_id"].isin(set(ids)) & ~methods["is_test"] & methods["n_statements"].between(3, 30)]
    cand = cand.sort_values(["method_id"])
    out = []
    for r in cand.itertuples():
        if r.repository in used_repos:
            continue
        out.append(r._asdict())
        used_repos.add(r.repository)
        if len(out) >= k:
            break
    if len(out) < k:   # relax: allow shorter / longer methods
        for r in methods[methods["method_id"].isin(set(ids)) & ~methods["is_test"]].sort_values("method_id").itertuples():
            if r.method_id not in {o["method_id"] for o in out}:
                out.append(r._asdict())
            if len(out) >= k:
                break
    return out


def to_task(ans: dict, task_id: str) -> dict:
    cls = re.sub(r"\W", "", ans.get("class_name") or "") or "PBRTask"
    if not cls.startswith("PBR"):
        cls = "PBR" + cls
    cls = cls[:60] + task_id.split("-")[-1][:6].upper()    # unique across tasks
    tests = [{"source": f"setUp\n\tsuper setUp.\n\ttask := {cls} new"}]
    tests += [{"source": src(t)} for t in ans.get("tests", []) if P.method_selector(src(t)) != "setUp"]
    targets = [{"source": src(m), "selector": P.method_selector(src(m)), "signature": P.method_signature(src(m))}
               for m in ans.get("target_methods", [])]
    return {"class": {"name": cls, "superclass": "Object", "instance_variables": [re.sub(r"\W", "", v) for v in ans.get("instance_variables", []) if re.sub(r"\W", "", v)],
                      "comment": ans.get("class_comment", "")},
            "test_class": {"name": cls + "Test", "superclass": "PBTestCase"}, "package": "PharoBenchResearch-Tasks",
            "target_methods": targets, "context_methods": [{"source": src(m)} for m in ans.get("context_methods", [])],
            "tests": tests, "statement": ans.get("statement", "")}


def report_for_author(result: dict) -> str:
    lines = [f"stage: {result.get('stage')}"]
    for p in (result.get("problems") or [])[:8]:
        lines.append(f"- {p.get('kind')}: {p.get('class', '')} {p.get('selector', '')} {str(p.get('detail', ''))[:200]}")
    for run in (result.get("reference_runs") or [])[:1]:
        for t, v in run.items():
            if v.get("outcome") != "pass":
                lines.append(f"- test {t}: {v.get('outcome')} {str(v.get('detail', ''))[:200]}")
    if result.get("stage") in ("crash", "timeout"):
        lines.append(str(result.get("detail", ""))[:400])
    return "\n".join(lines)


def tonel_like(task: dict) -> tuple[str, str]:
    c = task["class"]
    ref = [f'"{c["comment"]}"', f"Object subclass: #{c['name']}", f"\tinstanceVariableNames: '{' '.join(c['instance_variables'])}'", ""]
    for m in task["context_methods"]:
        ref += [f"{c['name']} >> {m['source']}", ""]
    for m in task["target_methods"]:
        ref += [f'"target (reference implementation)"', f"{c['name']} >> {m['source']}", ""]
    tests = [f"PBTestCase subclass: #{task['test_class']['name']}", ""]
    tests += [f"{task['test_class']['name']} >> {t['source']}\n" for t in task["tests"]]
    return "\n".join(ref), "\n".join(tests)


def run(ph):
    tax = latest_taxonomy(ph)
    bcfg = ph.cfg["benchmark"]
    sel = select_behaviors(ph, tax)
    if sel.empty:
        raise RuntimeError("no behaviour to build tasks for (coverage matrix empty?)")
    for u in (7, 8, 9):
        if ph.upstream_state(u).get("status") == "provisional":
            ph.mark_provisional(f"behaviours chosen from provisional results (phase {u})")
            break
    # labels of Pharo methods: phase 8 profile (model labels) + phase 6 annotation (AI majority)
    prof = pd.read_parquet(ph.upstream(8, "pharo_behavior_profiles.parquet"), columns=["method_id", "behavior_ids"])
    import ast
    agr = pd.read_csv(ph.upstream(7, "item_agreement.csv"), converters={"majority": ast.literal_eval})
    agr = agr[agr["source"] == "pharo"]
    labels = pd.concat([prof.assign(behavior_ids=prof["behavior_ids"].map(list)),
                        pd.DataFrame({"method_id": agr["item_id"].str[2:], "behavior_ids": agr["majority"].map(list)})])
    methods = pd.read_parquet(ph.upstream(2, "methods.parquet"),
                              columns=["method_id", "repository", "commit_sha", "class_name", "selector", "source", "url",
                                       "is_test", "n_statements", "superclass", "package"],
                              filters=[("method_id", "in", list(set(labels["method_id"])))])
    cov = pd.read_csv(ph.upstream(9, "coverage_matrix.csv")).set_index("behavior_id")
    lit = pd.read_parquet(ph.upstream(1, "literature_tasks.parquet"), columns=["task_uid", "card", "benchmark_name"]).set_index("task_uid")
    author = Client(bcfg["author"], ph.cfg)
    runner = P.Runner(ph.cfg, ph.log.info)
    ph.log.info(f"author {author.info().get('model')}; Pharo image {runner.info.get('system_version', '?')} "
                f"(sandbox: {runner.sandbox})")
    mint = int(ph.cfg["pharo"]["min_tests"])
    system = AUTHOR.replace("MINT", str(mint)).replace("MAXT", "8")
    target_n = int(ph.sizes["benchmark_candidates"])
    funnel = {"behaviors": len(sel), "seeds": 0, "authored": 0, "installs": 0, "reference_passes": 0}
    cands, used_repos = [], set()
    plan = []
    for b in sel.itertuples():
        for s in seeds_for(ph, b.behavior_id, methods, labels, int(bcfg["seeds_per_behavior"]), used_repos):
            plan.append((b, s))
    plan = plan[:target_n]
    funnel["seeds"] = len(plan)
    for b, seed in ph.progress(plan, "candidates authored", every_s=60):
        beh = T.index(tax).get(b.behavior_id) or {}
        task_id = "PBR-" + b.behavior_id + "-" + hashlib.sha1(seed["method_id"].encode()).hexdigest()[:6]
        lit_ex = str(cov.loc[b.behavior_id, "literature_examples"]) if b.behavior_id in cov.index else ""
        lit_id = next((x.strip()[2:] for x in lit_ex.split(";") if x.strip().startswith("L:")), None)
        lit_card = lit.loc[lit_id, "card"][:900] if lit_id in lit.index else None
        user = (f"BEHAVIOUR {b.behavior_id} {beh.get('name')}: {beh.get('definition')}\n"
                f"Use when: {'; '.join(beh.get('include') or [])}\n\n"
                f"SEED (real Pharo method from {seed['repository']}, class {seed['class_name']} subclass of {seed['superclass']}):\n"
                f"```smalltalk\n{seed['source'][:2500]}\n```\n"
                + (f"\nHOW AN EXISTING BENCHMARK TESTS THIS BEHAVIOUR (another language, for inspiration only):\n{lit_card}\n" if lit_card else "")
                + "\nWrite the task as JSON.")
        task, result, history = None, None, []
        for rnd in range(int(bcfg.get("repair_rounds", 1)) + 1):
            ans = author.ask(system=system, user=user, schema=SCHEMA, max_tokens=4000, prompt_version=PROMPT_VERSION,
                             phase="10_benchmark", item_id=f"{task_id}#r{rnd}")
            if not ans.valid:
                history.append({"round": rnd, "outcome": "invalid author answer", "errors": ans.errors})
                break
            task = to_task(ans.data, task_id)
            result = runner.run(P.job(task, mutation=False, repeat=1), ph.path("work", task_id, f"r{rnd}"))
            ok_install = result.get("stage") == "done"
            runs = result.get("reference_runs") or []
            ok_tests = bool(runs) and all(t.get("outcome") == "pass" for t in runs[0].values())
            history.append({"round": rnd, "stage": result.get("stage"), "installs": ok_install, "tests_pass": ok_tests,
                            "problems": (result.get("problems") or [])[:5]})
            if ok_install and ok_tests:
                break
            user = user + "\n\nYOUR PREVIOUS ANSWER:\n" + json.dumps(ans.data)[:6000] + \
                "\n\nIT FAILED IN THE PHARO IMAGE:\n" + report_for_author(result) + "\nFix it and answer with the complete corrected JSON."
        if task is None:
            status = "rejected: author answer invalid"
        else:
            funnel["authored"] += 1
            last = history[-1]
            if last.get("installs"):
                funnel["installs"] += 1
            if last.get("tests_pass"):
                funnel["reference_passes"] += 1
            status = "executes" if last.get("tests_pass") else f"rejected: {last.get('stage')}" + ("" if last.get("installs") else " (does not install)")
        rec = {"task_id": task_id, "behavior_ids": [b.behavior_id], "category_id": beh.get("category_id"),
               "subcategory_id": beh.get("subcategory_id"), "behavior": beh.get("name"),
               "taxonomy_version": tax["version"], "task_types": list(bcfg["task_types"]),
               "statement": (task or {}).get("statement"), "task": task, "target_pharo_version": str(bcfg["pharo_version"]),
               "provenance": {"selection_reason": b.reason, "seed_method_id": seed["method_id"], "seed_repository": seed["repository"],
                              "seed_commit": seed["commit_sha"], "seed_url": seed["url"],
                              "seed_selector": f"{seed['class_name']}>>{seed['selector']}",
                              "literature_example": lit_id, "coverage_status": cov.loc[b.behavior_id, "status"] if b.behavior_id in cov.index else None},
               "author": author.info(), "prompt_version": PROMPT_VERSION, "rounds": history, "validation_status": status}
        cands.append(rec)
        if task:
            d = ph.path("tasks", task_id)
            write_json(d / "task.json", rec)
            write_text(d / "statement.md", f"# {task_id}: {beh.get('name')}\n\n{task['statement']}\n\n"
                       f"Behaviour {b.behavior_id} · seed {seed['url']}\n")
            ref, tests = tonel_like(task)
            write_text(d / "reference.st", ref)
            write_text(d / "tests.st", tests)
    write_jsonl(ph.path("benchmark_candidates.jsonl"), cands)
    ok = [c for c in cands if c["validation_status"] == "executes"]
    ph.metric("behaviors_selected", len(sel))
    ph.metric("candidates", len(cands))
    ph.metric("candidates_executing", len(ok))
    ph.metric("funnel", funnel)
    ph.metric("llm", ledger_summary(ph.cfg, "10_benchmark"))
    stages = ["behaviors", "seeds", "authored", "installs", "reference_passes"]
    F.barh(["behaviours selected", "seed methods", "valid author answers", "install in Pharo", "reference passes its tests"],
           [funnel[s] for s in stages], "Benchmark construction funnel", "count", ph.figure("funnel.png"),
           colors=[F.SERIES[0]] * 2 + [F.SERIES[2]] * 3, note=f"after ≤ {bcfg.get('repair_rounds', 1)} repair round(s) with the image's error report")
    if cands:
        cc = pd.DataFrame([{"category": T.label(tax, c["category_id"]) if c["category_id"] else "?", "ok": c["validation_status"] == "executes"}
                           for c in cands])
        tab = cc.groupby(["category", "ok"]).size().unstack(fill_value=0).rename(columns={True: "executes", False: "rejected"})
        F.stacked_barh(list(tab.index), {k: tab[k].tolist() for k in ("executes", "rejected") if k in tab},
                       "Benchmark candidates per category", "candidates", ph.figure("candidates_per_category.png"),
                       colors={"executes": F.STATUS["good"], "rejected": F.STATUS["critical"]})
