"""Phase 11 — Benchmark validation.

**Execution** (isolated image, resource limits, no network: pbr/pharo.py). Each candidate that executed
in phase 10 is run again with the full checks, every one a fact observed in the image, in this order
(the first failure is the rejection reason): installs · hermetic · enough tests · reference passes
``pharo.repeat`` times (once in reverse order) · fail-to-pass (each target replaced by a stub fails a test)
· mutation score ≥ ``pharo.min_mutation_score`` (MuTalk).

**Variants** of every verified task (evaluation task types, §2):
* ``code_generation``: statement + class context + stubs;
* ``code_completion``: the same with the first lines of the reference body given;
* ``bug_repair``: the reference with a bug — a MuTalk mutant that the tests kill — to fix. A second image
  run checks the evaluation path: the reference, submitted as a candidate, passes; the buggy version fails.

**Experts** review clarity, correctness, realism and test quality (``expert/benchmark_review.xlsx``,
sheet Benchmark Task Validation): accept / revise / reject; completed files in ``expert/returned/``.
Without them a verified task is ``auto_verified`` (the phase is provisional).

Outputs: benchmark_verified/ (tasks.jsonl — what a solver sees; tasks_private.jsonl — with reference, tests,
gates, provenance; tasks/<task_id>/ with statement.md, reference.st, tests.st, result.json), gates.csv,
validation_report.md, expert/benchmark_review.xlsx, figures.
"""
from __future__ import annotations

import math

import pandas as pd

from .. import figures as F
from .. import pharo as P
from .. import taxonomy as T
from .. import workbook as W
from ..common import now_iso, read_jsonl, write_csv, write_json, write_jsonl, write_text
from .p08_profile import latest_taxonomy
from .p10_benchmark import tonel_like


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


def pick_mutant(result: dict, task: dict) -> dict | None:
    sels = {m["selector"] for m in task["target_methods"]}
    killed = [m for m in (result.get("mutation") or {}).get("mutants", [])
              if m.get("status") == "killed" and m.get("selector") in sels and m.get("source")]
    killed.sort(key=lambda m: (m["operator"], len(m["source"]), m["source"]))
    return killed[0] if killed else None


def run(ph):
    tax = latest_taxonomy(ph)
    cands = [c for c in read_jsonl(ph.upstream(10, "benchmark_candidates.jsonl")) if c["validation_status"] == "executes"]
    if ph.upstream_state(10).get("status") == "provisional":
        ph.mark_provisional("candidates come from a provisional phase 10")
    runner = P.Runner(ph.cfg, ph.log.info)
    ph.log.info(f"{len(cands)} candidates to verify; image {runner.info.get('system_version', '?')}, sandbox {runner.sandbox}, "
                f"{runner.workers} images in parallel")
    pcfg = ph.cfg["pharo"]

    def verify(c):
        r = runner.run(P.job(c["task"], mutation=True, repeat=int(pcfg["repeat"])), ph.path("work", c["task_id"], "full"))
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
        ph.log.info(f"{c['task_id']}: " + ("verified" if first is None else f"rejected at {first}")
                    + (f" (mutation score {r['mutation_score']:.2f})" if r.get("mutation_score") is not None else ""))
    gdf = pd.DataFrame(gate_rows)
    write_csv(ph.path("gates.csv"), gdf)

    # variants and the evaluation-path check (reference passes as a candidate, the bug fails)
    def variants(cr):
        c, r = cr
        task = c["task"]
        first_t = task["target_methods"][0]
        mut = pick_mutant(r, task)
        cand = [{"name": "reference", "methods": [{"class": task["class"]["name"], "side": "instance", "source": m["source"]}
                                                  for m in task["target_methods"]]}]
        if mut:
            cand.append({"name": "buggy", "methods": [{"class": task["class"]["name"], "side": "instance", "source": mut["source"]}]})
        rr = runner.run(P.job(task, mutation=False, repeat=1, candidates=cand), ph.path("work", c["task_id"], "variants"))
        res = {x["name"]: x for x in rr.get("candidates") or []}
        ok_ref = P.candidate_passed(res.get("reference", {}))
        bug_fails = mut is not None and "buggy" in res and not P.candidate_passed(res["buggy"])
        out = {"code_generation": {}, "code_completion": {"given_prefix": completion_prefix(first_t["source"])}}
        if bug_fails:
            out["bug_repair"] = {"buggy_source": mut["source"], "mutation_operator": mut["operator"]}
        return c, r, out, ok_ref, bug_fails

    var_results = runner.map(variants, verified)
    private, public = [], []
    for c, r, vars_, ok_ref, bug_fails in var_results:
        if not ok_ref:
            ph.warn(f"{c['task_id']}: the reference does not pass through the candidate path — task left out")
            continue
        wanted = [v for v in c["task_types"] if v in vars_]
        for v in wanted:
            pub = public_view(c["task"], c, v, vars_[v])
            public.append(pub)
            private.append({**pub, "reference": [m["source"] for m in c["task"]["target_methods"]],
                            "task": c["task"], "gates": {k: True for k, _ in P.GATES}, "mutation_score": r.get("mutation_score"),
                            "provenance": c["provenance"], "taxonomy_version": c["taxonomy_version"],
                            "image": runner.info, "validation_status": "auto_verified", "verified_at": now_iso(),
                            "prompt": prompt_text(pub)})
        d = ph.path("benchmark_verified", "tasks", c["task_id"])
        write_text(d / "statement.md", f"# {c['task_id']} — {c['behavior']}\n\n{c['task']['statement']}\n\n"
                   f"- behaviour: {c['behavior_ids'][0]} ({T.label(tax, c['category_id'])} › {T.label(tax, c['subcategory_id'])})\n"
                   f"- variants: {', '.join(wanted)}\n- seed: {c['provenance']['seed_url']}\n"
                   f"- mutation score: {r.get('mutation_score')}\n")
        ref, tests = tonel_like(c["task"])
        write_text(d / "reference.st", ref)
        write_text(d / "tests.st", tests)
        write_json(d / "result.json", r)

    # experts
    review = W.returned_files(ph.path("expert", "returned"))
    decisions = {}
    for f in review:
        reviewer, rows, prov = W.read_sheet(f, "Benchmark Task Validation")
        for row in rows:
            if row.get("decision"):
                decisions.setdefault(row.get("task_id"), []).append({**{k: row.get(k) for k in (
                    "clarity", "correctness", "realism", "test_quality", "decision", "comment")}, "reviewer": reviewer,
                    "file": f.name, "sha256": prov["sha256"]})
    for p in private:
        ds = decisions.get(p["base_task_id"]) or []
        if ds:
            votes = [str(d["decision"]).lower() for d in ds]
            p["validation_status"] = ("rejected" if "reject" in votes else "needs_revision" if "revise" in votes else "accepted")
            p["expert_review"] = ds
    if not decisions:
        ph.mark_provisional("no expert review of the tasks yet (expert/benchmark_review.xlsx → expert/returned/)")
    write_review(ph, tax, verified, gdf)
    write_jsonl(ph.path("benchmark_verified", "tasks.jsonl"), [p for p, q in zip(public, private) if q["validation_status"] != "rejected"])
    write_jsonl(ph.path("benchmark_verified", "tasks_private.jsonl"), private)

    n_var = pd.Series([p["task_type"] for p in private]).value_counts().to_dict() if private else {}
    ph.metric("candidates", len(cands))
    ph.metric("verified_tasks", len({p["base_task_id"] for p in private}))
    ph.metric("task_variants", n_var)
    ph.metric("rejected", int(gdf["first_failed_gate"].notna().sum()) if len(gdf) else 0)
    ph.metric("mean_mutation_score", round(float(gdf["mutation_score"].dropna().mean()), 3) if len(gdf) and gdf["mutation_score"].notna().any() else None)
    ph.metric("expert_reviews", sum(len(v) for v in decisions.values()))
    ph.metric("sandbox", runner.sandbox)
    write_text(ph.path("validation_report.md"), report(gdf, private, runner, decisions))
    figures(ph, gdf, private, tax)


def write_review(ph, tax, verified, gdf):
    rows = []
    for c, r in verified:
        t = c["task"]
        rows.append({"task_id": c["task_id"], "behavior": f"{c['behavior_ids'][0]} {c['behavior']}",
                     "statement": t["statement"],
                     "reference": "\n\n".join(m["source"] for m in t["context_methods"] + t["target_methods"]),
                     "tests": "\n\n".join(x["source"] for x in t["tests"]), "seed": c["provenance"]["seed_url"],
                     "mutation_score": r.get("mutation_score"), "variants": ", ".join(c["task_types"])})
    wb = W.new_workbook("Benchmark task validation (phase 11)", [
        "Each row is a task that passed every execution gate (installs, hermetic, tests pass 3× incl. reversed, "
        "fail-to-pass, mutation score). Rate 1 (poor) to 5 (excellent): clarity of the statement, correctness of the "
        "reference, realism (is this what Pharo developers do?), quality of the tests.",
        "decision: accept · revise (explain in comment) · reject. Save as expert/returned/<your name>.xlsx with your name in B3."],
        {"generated": now_iso(), "tasks": len(rows)})
    five = ["1", "2", "3", "4", "5"]
    W.add_sheet(wb, "Benchmark Task Validation", rows,
                [("task_id", "task_id", 22, False), ("behavior", "behavior", 26, False), ("statement", "statement", 50, False),
                 ("reference", "reference", 50, False), ("tests", "tests", 50, False), ("seed", "seed", 24, False),
                 ("mutation_score", "mutation_score", 10, False), ("variants", "variants", 16, False),
                 ("clarity", "clarity", 9, True), ("correctness", "correctness", 11, True), ("realism", "realism", 9, True),
                 ("test_quality", "test_quality", 11, True), ("decision", "decision", 11, True), ("comment", "comment", 30, True)],
                choices={"clarity": five, "correctness": five, "realism": five, "test_quality": five,
                         "decision": ["accept", "revise", "reject"]}, row_height=180)
    W.save(wb, ph.path("expert", "benchmark_review.xlsx"))


def report(gdf, private, runner, decisions) -> str:
    lines = ["# Benchmark validation", "", f"Image: {runner.info.get('system_version')} · image sha256 "
             f"{str(runner.info.get('image_sha256'))[:16]} · harness {runner.info.get('harness_sha256')} · sandbox: {runner.sandbox}", "",
             "## Gates (first failure = rejection reason)", "", "| gate | passed | failed | not reached |", "|---|---|---|---|"]
    for k, desc in P.GATES:
        if k in gdf:
            col = gdf[k]
            lines.append(f"| {k} — {desc} | {int((col == True).sum())} | {int((col == False).sum())} | {int(col.isna().sum())} |")  # noqa: E712
    lines += ["", f"Verified task variants: {len(private)}; expert decisions: {sum(len(v) for v in decisions.values())}", ""]
    return "\n".join(lines) + "\n"


def figures(ph, gdf, private, tax):
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
    if private:
        df = pd.DataFrame([{"category": T.label(tax, p["category_id"]), "variant": p["task_type"]} for p in private])
        tab = df.groupby(["category", "variant"]).size().unstack(fill_value=0)
        F.stacked_barh(list(tab.index), {v: tab[v].tolist() for v in tab.columns}, "Verified task variants per category",
                       "tasks", ph.figure("verified_per_category.png"))
