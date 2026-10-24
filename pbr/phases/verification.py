"""Phase 12 — Benchmark validation, in five levels (pbr/oracle.py for the independent checks).

**Level 1 — Executability** (isolated image, resource limits, no network: pbr/pharo.py). Each candidate that
executed in phase 11 is run again; every check is a fact observed in the image, in this order (the first
failure is the rejection reason): installs · hermetic · enough tests · reference passes ``pharo.repeat`` times
(once in reverse order) · fail-to-pass (each target replaced by a stub fails a test) · mutation score ≥
``pharo.min_mutation_score`` (MuTalk).

**Level 2 — Functional correctness against an independent oracle.** The author wrote the reference AND its
tests, so they can share one misunderstanding. Oracle models of other families (``benchmark.oracles``) see only
the solver's view and the behaviour contract (phase 11) and write their own tests; run on the reference in a
clean image, each test agrees, is a counterexample, is flaky, vacuous or invalid. A task needs at least
``benchmark.min_oracle_tests`` agreeing tests and no counterexample.

**Level 3 — Test-suite effectiveness: valid mutants.** Every MuTalk mutant of the target methods is re-run in
a clean image on the task's tests and on the independent tests: killed, survived (detectable: a test gap),
possibly equivalent, unstable, terminated, invalid. Mutation score bounds and one record per mutant.

**Level 4 — Behaviour alignment.** Each contract property is covered, violated or untested by the independent
tests; a task needs no violated property and a coverage ≥ ``benchmark.min_property_coverage``.

A task failing level 2 or 4 is **quarantined** (the reference or the oracle is wrong: only a person can tell),
not rejected; it is never evaluated unless an expert accepts it.

**Variants** of every task (evaluation task types, §2): ``code_generation`` (statement + class context + stubs);
``code_completion`` (the same with the first lines of the reference body); ``bug_repair`` (a mutant that compiles,
is killed reproducibly by a failing assertion or error, and violates the contract). The three variants share one
base task and its hidden tests: they are three variants of one problem, not three problems.

**Level 5 — Independent audit.** ``expert/benchmark_review.xlsx``: every quarantined task + a random sample of
the auto-verified ones, stratified by category (size: profile ``benchmark_audit``); accept / revise / reject in
``expert/returned/``. The share of audited auto-verified tasks experts do not accept estimates the residual
error of the automatic validation (Wilson 95 % interval).

Literature equivalence: phase 11 writes NEW tasks from Pharo seeds; none is a direct adaptation of a literature
task, so no cross-language differential test applies and no equivalence to a literature task is claimed
(``differential_tests_pass`` is null with the reason).

Outputs: benchmark_verified/ (tasks.jsonl — what a solver sees; tasks_private.jsonl — with reference, tests,
gates, provenance; tasks/<task_id>/), validation_reports/<task_id>.json + validation_reports.csv (one per task),
mutants.jsonl (one per mutant), independent_tests.jsonl, quarantine.csv, gates.csv, validation_report.md,
expert/benchmark_review.xlsx, figures.
"""
from __future__ import annotations

from collections import Counter
import math

import pandas as pd

from .. import agreement as A
from .. import figures as F
from .. import oracle as O
from .. import pharo as P
from .. import taxonomy as T
from .. import workbook as W
from ..common import now_iso, read_jsonl, write_csv, write_json, write_jsonl, write_text
from ..llm import Client, ledger_summary
from .profile import latest_taxonomy
from .benchmark import tonel_like


def public_view(task: dict, rec: dict, variant: str, extra: dict) -> dict:
    """What a model under evaluation receives (no reference, no tests)."""
    c = task["class"]
    return {"task_id": f"{rec['task_id']}::{variant}", "base_task_id": rec["task_id"], "task_type": variant,
            "behavior_ids": rec["behavior_ids"], "category_id": rec["category_id"], "subcategory_id": rec["subcategory_id"],
            "statement": task["statement"], "class": c["name"], "superclass": c.get("superclass", "Object"),
            "instance_variables": c.get("instance_variables", []), "context_methods": [m["source"] for m in task["context_methods"]],
            "target_methods": [{"selector": m["selector"], "signature": m["signature"]} for m in task["target_methods"]],
            "target_pharo_version": rec["target_pharo_version"], **extra}


def prompt_text(pub: dict) -> str:
    """The solver-facing text of a task variant (identical for every evaluated model)."""
    ivars = " ".join(pub["instance_variables"])
    lines = [pub["statement"].strip(), "", f"Pharo {pub['target_pharo_version']}. Class {pub['class']} (subclass of "
             f"{pub['superclass']}" + (f", instance variables: {ivars}" if ivars else "") + ")."]
    if pub["context_methods"]:
        lines += ["", "It already has these methods:"] + [f"```smalltalk\n{m}\n```" for m in pub["context_methods"]]
    if pub["task_type"] == "code_generation":
        lines += ["", "Implement " + ("this method" if len(pub["target_methods"]) == 1 else "these methods") + ":"]
        lines += [f"```smalltalk\n{m['signature']}\n\t^ self shouldBeImplemented\n```" for m in pub["target_methods"]]
    elif pub["task_type"] == "code_completion":
        lines += ["", "Complete this method (keep the given beginning, write the whole method):",
                  f"```smalltalk\n{pub['given_prefix']}\n```"]
        lines += [f"Also implement:\n```smalltalk\n{m['signature']}\n```" for m in pub["target_methods"][1:]]
    elif pub["task_type"] == "bug_repair":
        lines += ["", "This implementation has a bug: its tests fail. Fix it (write the whole corrected method):",
                  f"```smalltalk\n{pub['buggy_source']}\n```"]
    lines += ["", "Answer with the complete method source(s) in a ```smalltalk block: the message pattern on the first line, "
              "then the body; no class name, no `>>`."]
    return "\n".join(lines)


def completion_prefix(source: str, share: float = 0.4) -> str:
    lines = source.strip().splitlines()
    body = lines[1:]
    k = max(1, math.ceil(len(body) * share)) if len(body) > 1 else 0
    return "\n".join([lines[0]] + body[:k])


def write_oracle_tests(ph, verified, oracles) -> dict:
    """{task_id: (tests, untestable, rejected)}: each oracle model, alone, writes tests from the solver's view and
    the contract."""

    def one(c):
        pub = public_view(c["task"], c, "code_generation", {})
        user = O.oracle_user(pub, c.get("contract"))
        answers, untestable = [], []
        for o in oracles:
            ans = o.ask(system=O.ORACLE_PROMPT, user=user, schema=O.ORACLE_SCHEMA, max_tokens=3500,
                        prompt_version=O.ORACLE_VERSION, phase=ph.key, item_id=f"{c['task_id']}#{o.name}")
            if ans.valid:
                answers.append((o.name, ans.data))
                untestable += [{**u, "oracle": o.name} for u in ans.data.get("untestable") or []]
            else:
                ph.log.info(f"{c['task_id']}: oracle {o.name} gave no valid answer ({'; '.join(ans.errors)[:160]})")
        tests, rejected = O.oracle_tests(answers, c.get("contract"))
        return c["task_id"], (tests, untestable, rejected)
    # tasks are independent: compute.llm_parallel at once
    return dict(ph.pmap(one, [c for c, _ in verified], "independent test suites"))


def check_task(ph, runner, c, r, tests, cap, repeat):
    """Levels 2–3 for one task, in clean images: the independent tests on the reference (pass 1), the agreeing
    ones again alone with every mutant (pass 2), the task's own tests with the reference and every mutant."""
    task = c["task"]
    cls = task["class"]["name"]
    meta = {t["selector"]: t for t in tests}
    mutants = O.mutants_of(r, task, cap)
    cands = [O.mutant_candidate(task, m) for m in mutants if m["rechecked"] and m["mutalk_status"] != "terminated"]
    status, oracle_cands = {}, {}
    if tests:
        first = runner.run(P.job(O.oracle_task(task, tests), mutation=False, repeat=repeat, lenient_tests=True),
                           ph.path("work", c["task_id"], "oracle1"))
        status = O.classify_tests(tests, first, cls)
        for attempt in range(2):     # a test that turns out flaky stops the harness: once more without it
            keep = [meta[s] for s in O.useful(status)]
            if not keep:
                break
            second = runner.run(P.job(O.oracle_task(task, keep), mutation=False, repeat=repeat, candidates=cands),
                                ph.path("work", c["task_id"], f"oracle2-{attempt}"))
            status = O.confirm_tests(status, second, repeat)
            if all(status[t["selector"]]["status"] in ("agrees", "vacuous") for t in keep):
                oracle_cands = {x["name"]: x for x in second.get("candidates") or []}
                break
    ref = {"name": "reference", "methods": [{"class": cls, "side": "instance", "source": m["source"]} for m in task["target_methods"]]}
    rr = runner.run(P.job(task, mutation=False, repeat=1, candidates=[ref] + cands), ph.path("work", c["task_id"], "variants"))
    res = {x["name"]: x for x in rr.get("candidates") or []}
    records = O.classify_mutants(mutants, res, oracle_cands, O.useful(status), meta)
    return {"status": status, "records": records, "ok_ref": P.candidate_passed(res.get("reference", {}))}


def run(ph):
    tax = latest_taxonomy(ph)
    cands = [c for c in read_jsonl(ph.upstream("benchmark", "benchmark_candidates.jsonl")) if c["validation_status"] == "executes"]
    if ph.upstream_state("benchmark").get("status") == "provisional":
        ph.mark_provisional("candidates come from a provisional phase 11")
    runner = P.Runner(ph.cfg, ph.log.info)
    ph.log.info(f"{len(cands)} candidates to verify; image {runner.info.get('system_version', '?')}, sandbox {runner.sandbox}, "
                f"{runner.workers} images in parallel")
    pcfg, bcfg = ph.cfg["pharo"], ph.cfg["benchmark"]
    repeat = int(pcfg["repeat"])

    # level 1 (+ MuTalk): the execution gates
    def verify(c):
        r = runner.run(P.job(c["task"], mutation=True, repeat=repeat), ph.path("work", c["task_id"], "full"))
        g, first = P.gates(r, ph.cfg, mutation=True)
        return c, r, g, first

    results = runner.map(verify, cands)
    verified, gate_rows = [], []
    for c, r, g, first in results:
        gate_rows.append({"task_id": c["task_id"], "behavior_id": c["behavior_ids"][0], **{k: g[k] for k, _ in P.GATES},
                          "first_failed_gate": first, "mutation_score": r.get("mutation_score"),
                          "tests": len(r.get("tests") or []), "wall_seconds": r.get("wall_seconds"),
                          "static_problems": "; ".join(P.static_problems(r.get("facts")))})
        if first is None:
            verified.append((c, r))
        ph.log.info(f"{c['task_id']}: " + ("passes the execution gates" if first is None else f"rejected at {first}")
                    + (f" (mutation score {r['mutation_score']:.2f})" if r.get("mutation_score") is not None else ""))
    gdf = pd.DataFrame(gate_rows)
    write_csv(ph.path("gates.csv"), gdf)

    # levels 2-4: independent tests, valid mutants, behaviour alignment
    oracles = [Client(s, ph.cfg) for s in bcfg["oracles"]]
    for o in oracles:
        if o.model == bcfg["author"].get("model"):
            ph.warn(f"oracle {o.name} is the task author's model: its tests are not independent")
    suites = write_oracle_tests(ph, verified, oracles)
    cap = int(bcfg["max_mutants_rechecked"])
    checks = runner.map(lambda cr: check_task(ph, runner, cr[0], cr[1], suites[cr[0]["task_id"]][0], cap, repeat), verified)

    private, public, reports, all_mutants, all_tests, quarantine = [], [], [], [], [], []
    for (c, r), chk in zip(verified, checks):
        task = c["task"]
        tests, untestable, rejected = suites[c["task_id"]]
        status, records = chk["status"], chk["records"]
        for t in tests:
            all_tests.append({"task_id": c["task_id"], **t, **status.get(t["selector"], {"status": "invalid", "detail": "not run"})})
        for x in rejected:
            all_tests.append({"task_id": c["task_id"], **x, "status": "invalid", "detail": x["reason"]})
        for m in records:
            orig = next(t["source"] for t in task["target_methods"] if t["selector"] == m["selector"])
            all_mutants.append({"task_id": c["task_id"], **m, "original_source": orig})
        counter = [{"test": s, "property": next(t["property"] for t in tests if t["selector"] == s), "detail": v["detail"]}
                   for s, v in status.items() if v["status"] == "counterexample"]
        n_ok = len(O.useful(status))
        al = O.alignment(c.get("contract"), tests, status, untestable)
        ms = O.mutation_summary(records)
        why = []
        if not chk["ok_ref"]:
            why.append("the reference does not pass through the candidate path")
        if counter:
            why.append(f"{len(counter)} independent test(s) fail on the reference")
        if n_ok < int(bcfg["min_oracle_tests"]):
            why.append(f"{n_ok} agreeing independent test(s) (< {bcfg['min_oracle_tests']})")
        if al["status"] == "no_contract":
            why.append("no behaviour contract")
        elif al["status"] == "violated" or al["coverage"] < float(bcfg["min_property_coverage"]):
            why.append(f"behaviour alignment {al['status']} (coverage {al['coverage']})")
        status_ = "auto_verified" if not why else "quarantined"
        bug = O.pick_bug(records)
        vars_ = {"code_generation": {}, "code_completion": {"given_prefix": completion_prefix(task["target_methods"][0]["source"])}}
        if bug:
            vars_["bug_repair"] = {"buggy_source": bug["source"], "mutation_operator": bug["operator"], "mutant_id": bug["mutant_id"],
                                   "failing_tests": [f["test"] for f in bug["task_tests_failing"]]}
        wanted = [v for v in c["task_types"] if v in vars_]
        rep = {"task_id": c["task_id"], "behavior_id": c["behavior_ids"][0], "category_id": c["category_id"],
               "source_type": c.get("source_type", "new_from_pharo_seed"), "source_task_id": c.get("source_task_id"),
               "literature_inspiration": c["provenance"].get("literature_example"), "contract_id": c.get("contract_id"),
               "reference_compiles": True, "reference_tests_pass": True, "fail_to_pass": True,
               "independent_tests": len(tests), "independent_tests_agreeing": n_ok,
               "independent_tests_by_status": dict(Counter(s["status"] for s in status.values())),
               "independent_tests_pass": not counter and n_ok >= int(bcfg["min_oracle_tests"]),
               "counterexamples": counter,
               "property_tests_pass": al["status"] in ("verified", "partial"),
               "behavior_alignment": al["status"], "property_coverage": al["coverage"], "properties": al["properties"],
               "differential_tests_pass": None,
               "differential_note": "not applicable: a new task from a Pharo seed has no executable original to compare with",
               "mutalk_mutation_score": r.get("mutation_score"), **{k: ms[k] for k in (
                   "valid_mutants", "killed_mutants", "mutation_score_lower", "mutation_score_upper", "bug_repair_eligible")},
               "mutants_by_class": ms["counts"], "bug_repair_mutant": bug["mutant_id"] if bug else None,
               "variants": wanted, "human_review": "pending", "status": status_, "quarantine_reasons": why,
               "image": runner.info.get("system_version"), "validated_at": now_iso()}
        reports.append(rep)
        if why:
            quarantine.append({"task_id": c["task_id"], "behavior_id": c["behavior_ids"][0], "reasons": "; ".join(why),
                               "counterexamples": " | ".join(f"{x['test']} ({x['property']}): {x['detail'][:160]}" for x in counter)})
        for v in wanted:
            pub = public_view(task, c, v, vars_[v])
            public.append(pub)
            private.append({**pub, "reference": [m["source"] for m in task["target_methods"]],
                            "task": task, "gates": {k: True for k, _ in P.GATES}, "mutation_score": r.get("mutation_score"),
                            "contract_id": c.get("contract_id"), "behavior_alignment": al["status"],
                            "provenance": c["provenance"], "taxonomy_version": c["taxonomy_version"],
                            "image": runner.info, "validation_status": status_, "verified_at": now_iso(),
                            "prompt": prompt_text(pub)})
        d = ph.path("benchmark_verified", "tasks", c["task_id"])
        write_text(d / "statement.md", f"# {c['task_id']} — {c['behavior']}\n\n{task['statement']}\n\n"
                   f"- behaviour: {c['behavior_ids'][0]} ({T.label(tax, c['category_id'])} › {T.label(tax, c['subcategory_id'])})\n"
                   f"- contract: {c.get('contract_id')} — alignment {al['status']} (coverage {al['coverage']})\n"
                   f"- variants: {', '.join(wanted)}\n- seed: {c['provenance']['seed_url']}\n"
                   f"- mutation score: MuTalk {r.get('mutation_score')}, bounds {ms['mutation_score_lower']}–{ms['mutation_score_upper']}\n"
                   f"- status: {status_}" + (f" ({'; '.join(why)})" if why else "") + "\n")
        ref, tst = tonel_like(task)
        write_text(d / "reference.st", ref)
        write_text(d / "tests.st", tst)
        write_text(d / "independent_tests.st", "\n".join(f'"{t["oracle"]} · {t["property"]} · {status.get(t["selector"], {}).get("status")}"\n'
                                                         f"{task['class']['name']}OTest >> {t['source']}\n" for t in tests))
        write_json(d / "result.json", r)

    # level 5: experts (every quarantined task + a stratified random sample of the auto-verified ones)
    auto = [x for x in reports if x["status"] == "auto_verified"]
    sample = O.audit_sample(auto, int(ph.sizes.get("benchmark_audit", len(auto))), ph.seed)
    review_ids = [x["task_id"] for x in reports if x["status"] == "quarantined"] + sample
    decisions = import_decisions(ph)
    final = {}
    for x in reports:
        ds = decisions.get(x["task_id"]) or []
        if ds:
            votes = [str(d["decision"]).lower() for d in ds]
            final[x["task_id"]] = "rejected" if "reject" in votes else "needs_revision" if "revise" in votes else "accepted"
            x["human_review"] = final[x["task_id"]]
            x["expert_review"] = ds
            x["status"] = final[x["task_id"]]
        x["audit_reason"] = ("quarantined" if x["task_id"] in review_ids and x["task_id"] not in sample
                             else "random audit sample" if x["task_id"] in sample else None)
    for p in private:
        if p["base_task_id"] in final:
            p["validation_status"] = final[p["base_task_id"]]
            p["expert_review"] = decisions[p["base_task_id"]]
    if not decisions:
        ph.mark_provisional("no expert audit of the tasks yet (expert/benchmark_review.xlsx → expert/returned/)")
    audited = [final[t] for t in sample if t in final]
    residual = None
    if audited:
        k = sum(v != "accepted" for v in audited)
        lo, hi = A.wilson(k, len(audited))
        residual = {"audited": len(audited), "not_accepted": k, "share": round(k / len(audited), 3),
                    "ci95": [round(lo, 3), round(hi, 3)]}
    write_review(ph, verified, reports, review_ids, sample)

    usable = ("accepted", "auto_verified")
    write_jsonl(ph.path("benchmark_verified", "tasks.jsonl"), [p for p, q in zip(public, private) if q["validation_status"] in usable])
    write_jsonl(ph.path("benchmark_verified", "tasks_private.jsonl"), private)
    for x in reports:
        write_json(ph.path("validation_reports", f"{x['task_id']}.json"), x)
    flat = ["task_id", "behavior_id", "source_type", "contract_id", "reference_compiles", "reference_tests_pass", "fail_to_pass",
            "independent_tests", "independent_tests_agreeing", "independent_tests_pass", "property_tests_pass",
            "behavior_alignment", "property_coverage", "differential_tests_pass", "mutalk_mutation_score", "valid_mutants",
            "killed_mutants", "mutation_score_lower", "mutation_score_upper", "bug_repair_eligible", "bug_repair_mutant",
            "human_review", "audit_reason", "status"]
    write_csv(ph.path("validation_reports.csv"), pd.DataFrame([{k: x.get(k) for k in flat} for x in reports], columns=flat))
    write_jsonl(ph.path("mutants.jsonl"), all_mutants)
    write_jsonl(ph.path("independent_tests.jsonl"), all_tests)
    write_csv(ph.path("quarantine.csv"), pd.DataFrame(quarantine, columns=["task_id", "behavior_id", "reasons", "counterexamples"]))

    n_var = pd.Series([p["task_type"] for p in private if p["validation_status"] in usable]).value_counts().to_dict() if private else {}
    st = pd.Series([x["status"] for x in reports]).value_counts().to_dict() if reports else {}
    ph.metric("candidates", len(cands))
    ph.metric("pass_execution_gates", len(verified))
    ph.metric("task_status", st)
    ph.metric("verified_tasks", sum(x["status"] in usable for x in reports))
    ph.metric("task_variants", n_var)
    ph.metric("rejected", int(gdf["first_failed_gate"].notna().sum()) if len(gdf) else 0)
    ph.metric("mean_mutation_score", round(float(gdf["mutation_score"].dropna().mean()), 3) if len(gdf) and gdf["mutation_score"].notna().any() else None)
    ph.metric("mutants", pd.Series([m["classification"] for m in all_mutants]).value_counts().to_dict() if all_mutants else {})
    ph.metric("independent_tests", pd.Series([t["status"] for t in all_tests]).value_counts().to_dict() if all_tests else {})
    ph.metric("expert_reviews", sum(len(v) for v in decisions.values()))
    ph.metric("residual_error", residual)
    ph.metric("sandbox", runner.sandbox)
    ph.metric("llm", ledger_summary(ph.cfg, ph.key))
    write_text(ph.path("validation_report.md"), report(gdf, reports, all_mutants, all_tests, runner, oracles, decisions, residual, ph.cfg))
    figures(ph, gdf, private, reports, all_mutants, tax)


def import_decisions(ph) -> dict:
    decisions = {}
    for f in W.returned_files(ph.path("expert", "returned")):
        reviewer, rows, prov = W.read_sheet(f, "Benchmark Task Validation")
        for row in rows:
            if row.get("decision"):
                decisions.setdefault(row.get("task_id"), []).append({**{k: row.get(k) for k in (
                    "clarity", "correctness", "realism", "test_quality", "decision", "comment")}, "reviewer": reviewer,
                    "file": f.name, "sha256": prov["sha256"]})
    return decisions


def write_review(ph, verified, reports, review_ids, sample):
    by_id = {c["task_id"]: (c, r) for c, r in verified}
    rep = {x["task_id"]: x for x in reports}
    rows = []
    for tid in review_ids:
        c, r = by_id[tid]
        t, x = c["task"], rep[tid]
        rows.append({"task_id": tid, "why_reviewed": "random audit sample" if tid in sample else "quarantined: " + "; ".join(x["quarantine_reasons"]),
                     "behavior": f"{c['behavior_ids'][0]} {c['behavior']}",
                     "contract": O.contract_text(c.get("contract")) or "(none)",
                     "statement": t["statement"],
                     "reference": "\n\n".join(m["source"] for m in t["context_methods"] + t["target_methods"]),
                     "tests": "\n\n".join(s["source"] for s in t["tests"]),
                     "counterexamples": "\n".join(f"{e['test']} ({e['property']}): {e['detail'][:200]}" for e in x["counterexamples"]),
                     "alignment": f"{x['behavior_alignment']} ({x['property_coverage']}): "
                                  + ", ".join(f"{k} {v}" for k, v in x["properties"].items()),
                     "seed": c["provenance"]["seed_url"], "mutation_score": r.get("mutation_score"), "variants": ", ".join(x["variants"])})
    wb = W.new_workbook("Benchmark task audit (phase 12)", [
        "Each row is a task that passed every execution gate. Rows marked 'quarantined' disagree with the independent tests "
        "(see counterexamples: is the reference wrong, or the independent test?) or miss contract properties; rows marked "
        "'random audit sample' were drawn at random among the automatically verified tasks, stratified by category: "
        "your decisions on them estimate how often the automatic validation is wrong.",
        "Rate 1 (poor) to 5 (excellent): clarity of the statement, correctness of the reference, realism (is this what Pharo "
        "developers do?), quality of the tests. decision: accept · revise (explain in comment) · reject. "
        "Save as expert/returned/<your name>.xlsx with your name in B3."],
        {"generated": now_iso(), "tasks": len(rows), "quarantined": len(rows) - len(sample), "audit_sample": len(sample)})
    five = ["1", "2", "3", "4", "5"]
    W.add_sheet(wb, "Benchmark Task Validation", rows,
                [("task_id", "task_id", 22, False), ("why_reviewed", "why_reviewed", 24, False), ("behavior", "behavior", 26, False),
                 ("contract", "contract", 40, False), ("statement", "statement", 50, False),
                 ("reference", "reference", 50, False), ("tests", "tests", 50, False), ("counterexamples", "counterexamples", 40, False),
                 ("alignment", "alignment", 24, False), ("seed", "seed", 24, False),
                 ("mutation_score", "mutation_score", 10, False), ("variants", "variants", 16, False),
                 ("clarity", "clarity", 9, True), ("correctness", "correctness", 11, True), ("realism", "realism", 9, True),
                 ("test_quality", "test_quality", 11, True), ("decision", "decision", 11, True), ("comment", "comment", 30, True)],
                choices={"clarity": five, "correctness": five, "realism": five, "test_quality": five,
                         "decision": ["accept", "revise", "reject"]}, row_height=180)
    W.save(wb, ph.path("expert", "benchmark_review.xlsx"))


def report(gdf, reports, mutants, tests, runner, oracles, decisions, residual, cfg) -> str:
    b = cfg["benchmark"]
    n = len(reports)
    cnt = lambda rows, key: pd.Series([x[key] for x in rows]).value_counts().to_dict() if rows else {}  # noqa: E731
    lines = ["# Benchmark validation", "", f"Image: {runner.info.get('system_version')} · image sha256 "
             f"{str(runner.info.get('image_sha256'))[:16]} · harness {runner.info.get('harness_sha256')} · sandbox: {runner.sandbox}",
             f"Independent oracles: {', '.join(f'{o.name} ({o.model})' for o in oracles)} · author: {b['author'].get('model')}", "",
             "## Level 1 — Executability (first failure = rejection reason)", "",
             "| gate | passed | failed | not reached |", "|---|---|---|---|"]
    for k, desc in P.GATES:
        if k in gdf:
            col = gdf[k]
            lines.append(f"| {k} — {desc} | {int((col == True).sum())} | {int((col == False).sum())} | {int(col.isna().sum())} |")  # noqa: E712
    ts = cnt(tests, "status")
    lines += ["", "## Level 2 — Independent oracle", "",
              f"{len(tests)} independent tests on {n} tasks: " + ", ".join(f"{k} {v}" for k, v in sorted(ts.items())) + ".",
              f"A task needs ≥ {b['min_oracle_tests']} agreeing tests and no counterexample: "
              f"{sum(x['independent_tests_pass'] for x in reports)} / {n} do.", "",
              "## Level 3 — Mutants (re-run in a clean image on the task's and the independent tests)", ""]
    mc = cnt(mutants, "classification")
    lines += [f"{len(mutants)} mutants: " + ", ".join(f"{k} {v}" for k, v in sorted(mc.items())) + ".",
              "Mutation score bounds per task: killed / (killed + all survivors) ≤ score ≤ killed / (killed + detectable "
              "survivors); 'possibly equivalent' survivors are never assumed equivalent in the lower bound.",
              f"Tasks with a bug-repair mutant (compiles, killed reproducibly by a failure or error, violates the contract): "
              f"{sum(1 for x in reports if x['bug_repair_mutant'])} / {n}.", "",
              "## Level 4 — Behaviour alignment (contract properties)", ""]
    lines += [", ".join(f"{k} {v}" for k, v in sorted(cnt(reports, "behavior_alignment").items())) + f" (minimum coverage {b['min_property_coverage']}).",
              "", "## Level 5 — Independent audit", ""]
    lines += [f"Quarantined tasks: {sum(1 for x in reports if x['quarantine_reasons'])} (all sent to the experts). "
              f"Expert decisions: {sum(len(v) for v in decisions.values())}."]
    lines += [f"Residual error of the automatic validation: {residual['not_accepted']} / {residual['audited']} audited auto-verified "
              f"tasks not accepted = {residual['share']} (95 % CI {residual['ci95'][0]}–{residual['ci95'][1]})." if residual
              else "Residual error: not estimated yet (no returned audit)."]
    lines += ["", "## Status", "", ", ".join(f"{k} {v}" for k, v in sorted(cnt(reports, "status").items())), "",
              "Literature equivalence is not claimed: every task is new (from a Pharo seed) and validated against its own "
              "behaviour contract; no cross-language differential test applies.", ""]
    return "\n".join(lines) + "\n"


def figures(ph, gdf, private, reports, mutants, tax):
    if len(gdf):
        passed = [int((gdf[k] == True).sum()) for k, _ in P.GATES]  # noqa: E712
        F.barh([f"{k}" for k, _ in P.GATES], passed, f"Execution gates passed ({len(gdf)} candidates)", "candidates passing",
               ph.figure("gates_funnel.png"), colors=[F.STATUS["good"]] * len(passed))
        ms = gdf["mutation_score"].dropna()
        if len(ms):
            fig, ax = F.plt.subplots(figsize=(7, 3.6))
            ax.hist(ms, bins=[i / 10 for i in range(11)], color=F.SERIES[0], rwidth=0.9)
            ax.axvline(float(ph.cfg["pharo"]["min_mutation_score"]), color=F.STATUS["critical"], lw=1.5)
            ax.text(float(ph.cfg["pharo"]["min_mutation_score"]), ax.get_ylim()[1] * 0.9, " threshold", color=F.INK_2, fontsize=8)
            ax.set_xlabel("mutation score (killed / evaluated mutants)")
            ax.set_ylabel("tasks")
            ax.set_title("Test strength: MuTalk mutation score of the tasks")
            ax.grid(axis="x", visible=False)
            F.save(fig, ph.figure("mutation_scores.png"))
    if mutants:
        vc = pd.Series([m["classification"] for m in mutants]).value_counts()
        good = {"killed": F.STATUS["good"], "survived_detectable": F.STATUS["critical"]}
        F.barh(list(vc.index), list(vc.values), f"Mutants re-run in a clean image ({len(mutants)})", "mutants",
               ph.figure("mutant_classes.png"), colors=[good.get(k, F.SERIES[0]) for k in vc.index],
               note="survived_detectable = a fault the independent tests see and the task's tests miss")
    if reports:
        st = pd.Series([x["status"] for x in reports]).value_counts()
        F.barh(list(st.index), list(st.values), f"Tasks after the five validation levels ({len(reports)})", "tasks",
               ph.figure("task_status.png"))
    usable = [p for p in private if p["validation_status"] in ("accepted", "auto_verified")]
    if usable:
        df = pd.DataFrame([{"category": T.label(tax, p["category_id"]), "variant": p["task_type"]} for p in usable])
        tab = df.groupby(["category", "variant"]).size().unstack(fill_value=0)
        F.stacked_barh(list(tab.index), {v: tab[v].tolist() for v in tab.columns}, "Verified task variants per category",
                       "tasks", ph.figure("verified_per_category.png"))
