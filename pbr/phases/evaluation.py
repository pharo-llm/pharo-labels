"""Phase 13 — Model evaluation.

Every chat model of ``evaluation.models`` receives the same prompt for every verified task variant
(code generation, code completion, bug repair) under the same conditions (temperature 0, fixed seed,
``evaluation.max_tokens``). Each answer is parsed into method sources and run, as an untrusted candidate, in
the isolated image (one image per task, all models' candidates in it; resource limits, no network). The
``reference`` model submits the reference solutions: it must score 100% (a check of the whole path).

Recorded per (task variant, model): pass (every test passes), tests passed / total, error kind (no code,
syntax or compile error, undeclared variable, test failure, timeout), generation latency and tokens,
image wall time, model configuration (name, provider, digest, quantisation, temperature, seed).

Aggregated by model × category / subcategory / behaviour / task type, with 95% Wilson intervals.
Reproducibility: ``reproducibility.json`` (config hash, git commit, Python and package versions, image and
harness hashes, model digests, prompt version). ``release/``: the benchmark (public + private tasks),
taxonomy, coverage matrix, evaluation results and README — what is published.

Outputs: evaluation_results.csv, evaluation_summary.csv, answers.jsonl, reproducibility.json, release/, figures.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from .. import agreement as A
from .. import figures as F
from .. import pharo as P
from .. import taxonomy as T
from ..common import ROOT, now_iso, read_jsonl, stable_hash, write_csv, write_json, write_jsonl, write_text
from ..llm import Client, LLMError, ledger_summary
from .profile import base_taxonomy_path, latest_taxonomy

PROMPT_VERSION = "evaluation-v1"
SYSTEM = "You are an expert Pharo (Smalltalk) developer. Answer with the requested method source(s) only, in a ```smalltalk block."


def error_kind(c: dict | None) -> str:
    if c is None:
        return "no candidate"
    errs = " ".join(c.get("errors") or [])
    if "syntax" in errs:
        return "syntax error"
    if "undeclared" in errs:
        return "undeclared variable"
    if errs:
        return "compile error"
    outs = [t.get("outcome") for t in (c.get("tests") or {}).values()]
    if outs and all(o == "pass" for o in outs):
        return "pass"
    if "timeout" in outs:
        return "timeout"
    if "error" in outs:
        return "runtime error"
    return "test failure"


def run(ph):
    tax = latest_taxonomy(ph)
    tasks = read_jsonl(ph.upstream("verification", "benchmark_verified", "tasks_private.jsonl"))
    tasks = [t for t in tasks if t["validation_status"] in ("accepted", "auto_verified")]
    if not tasks:
        raise RuntimeError("no verified task to evaluate (phase 12)")
    if ph.upstream_state("verification").get("status") == "provisional":
        ph.mark_provisional("tasks are auto-verified by execution but not yet reviewed by experts")
    ecfg = ph.cfg["evaluation"]
    runner = P.Runner(ph.cfg, ph.log.info)
    models = []
    for spec in ecfg["models"]:
        cl = Client(spec, ph.cfg)
        try:
            info = cl.info() if spec.get("provider") != "reference" else {"name": "reference", "provider": "reference"}
            models.append((cl, info))
        except Exception as e:  # noqa: BLE001 — a missing model is reported, the others run
            ph.warn(f"model {spec['name']} not available: {e}")
    ph.log.info(f"{len(tasks)} task variants × {len(models)} models: {[m[0].name for m in models]}")

    answers = []
    for cl, info in models:
        for t in ph.progress(tasks, f"{cl.name}: answers", every_s=60):
            if cl.provider == "reference":
                srcs = t["reference"]
                usage, valid = {}, True
            else:
                try:
                    ans = cl.ask(system=SYSTEM, user=t["prompt"], schema=None, max_tokens=int(ecfg["max_tokens"]),
                                 prompt_version=PROMPT_VERSION, phase=ph.key, item_id=t["task_id"])
                    srcs, usage, valid = P.split_methods(ans.text), ans.usage, True
                except LLMError as e:      # a model that loops or crashes fails this task; the evaluation goes on
                    ph.warn(f"{cl.name} on {t['task_id']}: no answer ({str(e)[:120]}) — counted as a failure")
                    srcs, usage, valid = [], {}, False
            answers.append({"task_id": t["task_id"], "model": cl.name, "methods": srcs, "usage": usage, "valid": valid})
    write_jsonl(ph.path("answers.jsonl"), answers)

    by_base = {}
    for t in tasks:
        by_base.setdefault(t["base_task_id"], []).append(t)

    def evaluate(base_id):
        group = by_base[base_id]
        task = group[0]["task"]
        targets = {m["selector"] for m in task["target_methods"]}
        cands = []
        for t in group:
            for a in answers:
                if a["task_id"] != t["task_id"]:
                    continue
                ms = [{"class": task["class"]["name"], "side": "instance", "source": s} for s in a["methods"]
                      if P.method_selector(s) in targets or len(a["methods"]) == 1]
                cands.append({"name": f"{a['model']}||{t['task_type']}", "methods": ms})
        r = runner.run(P.job(task, mutation=False, repeat=1, candidates=cands), ph.path("work", base_id))
        return base_id, r

    results = runner.map(evaluate, list(by_base))
    rows = []
    for base_id, r in results:
        res = {c["name"]: c for c in r.get("candidates") or []}
        for t in by_base[base_id]:
            for cl, info in models:
                c = res.get(f"{cl.name}||{t['task_type']}")
                a = next((x for x in answers if x["task_id"] == t["task_id"] and x["model"] == cl.name), {})
                tests = (c or {}).get("tests") or {}
                kind = error_kind(c) if a.get("methods") else ("model error" if not a.get("valid", True) else "no code in answer")
                rows.append({"task_id": t["task_id"], "base_task_id": base_id, "task_type": t["task_type"],
                             "behavior_id": t["behavior_ids"][0], "category_id": t["category_id"],
                             "subcategory_id": t["subcategory_id"], "model": cl.name, "provider": info.get("provider"),
                             "model_id": info.get("model"), "model_digest": info.get("digest"),
                             "quantization": info.get("quantization"), "temperature": info.get("temperature"),
                             "passed": kind == "pass", "outcome": kind,
                             "tests_passed": sum(v.get("outcome") == "pass" for v in tests.values()), "tests_total": len(tests),
                             "generation_latency_s": (a.get("usage") or {}).get("latency_s"),
                             "output_tokens": (a.get("usage") or {}).get("output_tokens"),
                             "image_wall_s": r.get("wall_seconds"), "image_stage": r.get("stage")})
    df = pd.DataFrame(rows)
    df["category"] = df["category_id"].map(lambda c: T.label(tax, c))
    df["subcategory"] = df["subcategory_id"].map(lambda c: T.label(tax, c))
    df["behavior"] = df["behavior_id"].map(lambda c: T.label(tax, c))
    write_csv(ph.path("evaluation_results.csv"), df)

    summ = []
    for dim in ("all", "task_type", "category", "subcategory", "behavior"):
        keys = ["model"] + ([] if dim == "all" else [dim])
        for k, g in df.groupby(keys):
            k = k if isinstance(k, tuple) else (k,)
            n, s = len(g), int(g["passed"].sum())
            lo, hi = A.wilson(s, n)
            summ.append({"dimension": dim, "model": k[0], "value": k[1] if len(k) > 1 else "all", "tasks": n, "passed": s,
                         "pass_rate": round(s / n, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    sdf = pd.DataFrame(summ)
    write_csv(ph.path("evaluation_summary.csv"), sdf)
    ref = sdf[(sdf["dimension"] == "all") & (sdf["model"] == "reference")]
    if len(ref) and ref["pass_rate"].iloc[0] < 1:
        ph.warn(f"the reference solutions pass only {ref['pass_rate'].iloc[0]:.0%}: the evaluation path is broken for some tasks")
    for r in sdf[sdf["dimension"] == "all"].itertuples():
        ph.metric(f"pass@1_{r.model}", r.pass_rate)
    ph.metric("task_variants", len(tasks))
    ph.metric("llm", ledger_summary(ph.cfg, ph.key))

    repro = reproducibility(ph, runner, models)
    write_json(ph.path("reproducibility.json"), repro)
    release(ph, tax, df, sdf, repro)
    figures(ph, df, sdf)


def reproducibility(ph, runner, models) -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    pkgs = {}
    for p in ("pandas", "numpy", "scipy", "matplotlib", "openpyxl", "jsonschema", "pyarrow", "requests"):
        try:
            pkgs[p] = __import__(p).__version__
        except Exception:  # noqa: BLE001
            pkgs[p] = None
    return {"generated": now_iso(), "config_hash": stable_hash(ph.cfg), "profile": ph.cfg["profile_name"],
            "git_commit": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")), "python": sys.version.split()[0], "platform": platform.platform(),
            "packages": pkgs, "pharo_image": runner.info, "prompt_version": PROMPT_VERSION,
            "models": [info for _, info in models], "evaluation": ph.cfg["evaluation"], "pharo": ph.cfg["pharo"],
            "rerun": "PROFILE=" + ph.cfg["profile_name"] + " ./run.sh 12  (LLM answers are cached in results/cache/llm; "
                     "delete the cache to ask the models again)"}


def release(ph, tax, df, sdf, repro):
    rel = ph.path("release")
    if rel.exists():
        shutil.rmtree(rel)
    rel.mkdir(parents=True)
    bv = ph.upstream("verification", "benchmark_verified")
    shutil.copytree(bv, rel / "benchmark")
    t5 = sorted(ph.upstream("validation").glob("taxonomy_v*.json")) or [base_taxonomy_path(ph)]
    shutil.copy2(t5[-1], rel / t5[-1].name)
    shutil.copy2(ph.upstream("coverage", "coverage_matrix.csv"), rel / "coverage_matrix.csv")
    df.to_csv(rel / "evaluation_results.csv", index=False)
    sdf.to_csv(rel / "evaluation_summary.csv", index=False)
    write_json(rel / "reproducibility.json", repro)
    overall = sdf[sdf["dimension"] == "all"].sort_values("pass_rate", ascending=False)
    lines = [f"# PharoBench-Research release ({now_iso()[:10]})", "",
             f"Taxonomy {tax['version']} ({tax['status']}); {df['base_task_id'].nunique()} tasks, {df['task_id'].nunique()} task variants "
             f"(code generation, code completion, bug repair), target Pharo {ph.cfg['benchmark']['pharo_version']}.", "",
             "Every task traces back to a behaviour ID of the taxonomy, the coverage evidence that motivated it "
             "(coverage_matrix.csv), the real Pharo method that seeded it (repository, commit, link) and its execution "
             "gates in a pinned Pharo image (benchmark/tasks_private.jsonl, benchmark/tasks/<id>/result.json).", "",
             "| model | pass@1 | 95% CI | tasks |", "|---|---|---|---|"]
    lines += [f"| {r.model} | {r.pass_rate:.1%} | {r.ci_low:.1%}–{r.ci_high:.1%} | {r.tasks} |" for r in overall.itertuples()]
    lines += ["", "Files: `benchmark/tasks.jsonl` (what a solver sees), `benchmark/tasks_private.jsonl` (reference, tests, "
              "gates, provenance), `evaluation_*.csv`, `reproducibility.json`.", ""]
    write_text(rel / "README.md", "\n".join(lines))


def figures(ph, df, sdf):
    ov = sdf[sdf["dimension"] == "all"].sort_values("pass_rate", ascending=False)
    err = np.array([(ov["pass_rate"] - ov["ci_low"]).clip(lower=0), (ov["ci_high"] - ov["pass_rate"]).clip(lower=0)]).T * 100
    F.barh(ov["model"], ov["pass_rate"] * 100, f"pass@1 on the Pharo benchmark ({df['task_id'].nunique()} task variants)",
           "% of task variants solved", ph.figure("pass_at_1.png"), errors=err, fmt="{:.0f}%", xlim=(0, 110),
           note="95% Wilson intervals; 'reference' = the reference solutions (must be 100%)")
    for dim, title in (("task_type", "by evaluation task type"), ("category", "by behaviour category")):
        sub = sdf[(sdf["dimension"] == dim) & (sdf["model"] != "reference")]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="value", columns="model", values="pass_rate") * 100
        F.heatmap(piv.values, list(piv.index), list(piv.columns), f"pass@1 {title} (%)", ph.figure(f"pass_by_{dim}.png"),
                  fmt="{:.0f}", vmin=0, vmax=100, cbar_label="% solved")
    oc = df[df["model"] != "reference"].groupby(["model", "outcome"]).size().unstack(fill_value=0)
    order = [c for c in ["pass", "test failure", "runtime error", "timeout", "compile error", "undeclared variable",
                         "syntax error", "no code in answer", "model error", "no candidate"] if c in oc.columns]
    F.stacked_barh(list(oc.index), {c: oc[c].tolist() for c in order}, "Outcome of each answer", "task variants",
                   ph.figure("outcomes.png"), colors={"pass": F.STATUS["good"], "test failure": F.STATUS["warning"],
                                                      "runtime error": F.STATUS["serious"], "compile error": F.STATUS["critical"],
                                                      "syntax error": "#8f1d1a", "undeclared variable": "#b4443b",
                                                      "timeout": "#6250d6", "no code in answer": F.STATUS["neutral"], "model error": "#3d3c39",
                                                      "no candidate": "#cfcdc7"})
