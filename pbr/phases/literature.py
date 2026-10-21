"""Phase 2 — Literature ingestion.

Imports every task of every existing benchmark (pinned sources of ../labeling-existing-benchmarks, downloaded
once into ../data/existing-benchmarks/) and records, per task: its source (benchmark, dataset, revision,
paper), what the evaluated model receives (description), the reference answer, how it is evaluated (tests),
its programming language, and the evaluation task type of the benchmark (config ``literature.task_types``).

Outputs
  literature_tasks.parquet     one row per task; ``task_uid`` = "<benchmark>::<item id>" (stable)
  literature_benchmarks.csv    one row per benchmark: source, revision, size, task type, languages, status
  figures/tasks_per_benchmark.png, figures/task_types.png, figures/languages.png
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import re
import urllib.error

import pandas as pd

from .. import figures as F
from ..common import resolve, write_csv, write_parquet
from ..taxonomy import TASK_TYPES

EXT_LANG = {".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript", ".jsx": "JavaScript",
            ".java": "Java", ".go": "Go", ".rs": "Rust", ".rb": "Ruby", ".php": "PHP", ".c": "C", ".h": "C",
            ".cpp": "C++", ".cc": "C++", ".hpp": "C++", ".cs": "C#", ".kt": "Kotlin", ".scala": "Scala",
            ".swift": "Swift", ".sh": "Shell"}
PRO_LANG = {"js": "JavaScript", "ts": "TypeScript", "py": "Python", "go": "Go", "python": "Python", "java": "Java"}


def load_loader(ph):
    path = resolve(ph.cfg["literature"]["loader"])
    spec = importlib.util.spec_from_file_location("existing_benchmarks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def _patch_language(patch: str) -> str:
    exts = Counter()
    for f in re.findall(r"^diff --git a/(\S+)", patch or "", re.M):
        m = re.search(r"(\.[A-Za-z0-9]+)$", f)
        if m and m.group(1) in EXT_LANG and "/test" not in f.lower():
            exts[EXT_LANG[m.group(1)]] += 1
    return exts.most_common(1)[0][0] if exts else "unknown"


def _s(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def record(key: str, b: dict, r: dict, loader) -> dict:
    """One literature task: description, reference, evaluation, language, dedup key."""
    item_id, card = loader.card(key, r)
    desc = ref = evaluation = ""
    lang = "Python"
    dedup = f"{key}::{item_id}"
    if key in ("humanevalplus",):
        desc, ref, evaluation = r["prompt"], r["canonical_solution"], r.get("test", "")
    elif key == "mbppplus":
        desc, ref = r["prompt"], r["code"]
        evaluation = "\n".join(r.get("test_list") or []) + "\n" + _s(r.get("test"))[:4000]
    elif key in ("bigcodebench_complete", "bigcodebench_instruct"):
        desc = r["complete_prompt"] if key.endswith("complete") else r["instruct_prompt"]
        ref, evaluation = r["code_prompt"] + r["canonical_solution"], r.get("test", "")
        dedup = f"bigcodebench::{r['task_id']}"
    elif key == "livecodebench_codegen":
        desc = f"{r['question_title']}\n{r['question_content']}\n{r.get('starter_code') or ''}"
        evaluation = _s(r.get("public_test_cases"))
        dedup = f"livecodebench::{r['question_id']}"
    elif key == "livecodebench_testoutput":
        desc = f"{r['question_content']}\n{r.get('starter_code') or ''}\nfunction: {r['function_name']}"
        evaluation = _s(r.get("test"))
        dedup = f"livecodebench::{r['question_id']}"
    elif key == "livecodebench_execution":
        desc, ref = f"{r['code']}\nassert {r['input']} == ??", _s(r.get("output"))
        evaluation = "exact match with the value the code returns"
        dedup = f"livecodebench::{r['question_id']}"
    elif key.startswith("swebench") or key == "deepswe":
        desc = _s(r.get("problem_statement") or r.get("instruction") or r.get("prompt"))
        ref = _s(r.get("patch") or r.get("solution"))
        evaluation = (f"FAIL_TO_PASS: {_s(r.get('FAIL_TO_PASS') or r.get('fail_to_pass'))[:3000]}\n"
                      f"PASS_TO_PASS: {_s(r.get('PASS_TO_PASS') or r.get('pass_to_pass'))[:1000]}\n"
                      f"test_patch:\n{_s(r.get('test_patch'))[:4000]}")
        if key == "swebench_pro":
            lang = PRO_LANG.get(str(r.get("repo_language", "")).lower(), str(r.get("repo_language") or "unknown"))
        elif key in ("swebench_multilingual", "deepswe"):
            lang = _patch_language(ref)
    elif key == "terminalbench2":
        desc, ref, evaluation = r.get("instruction.md", ""), r.get("solution/solve.sh", ""), r.get("tests/test.sh", "")
        lang = "Shell / mixed"
    elif key == "nl2repo":
        desc, evaluation = r.get("start.md", ""), r.get("test_commands.json", "")
        ref = f"the original repository of the library '{item_id}'"
    tt = b["_task_type"]
    return {
        "task_uid": f"{key}::{item_id}", "benchmark": key, "benchmark_name": b["name"], "group": b["group"],
        "item_id": str(item_id), "source_dataset": b.get("repo", ""), "source_revision": b.get("rev", ""),
        "source_reference": b.get("ref", ""), "repo": _s(r.get("repo")) or None, "language": lang,
        "task_type": tt.get("task_type", "other"), "task_subtype": tt.get("subtype"),
        "granularity": tt.get("granularity"), "evaluation_method": tt.get("evaluation"),
        "description": desc, "reference_solution": ref, "evaluation": evaluation,
        "has_tests": bool(evaluation.strip()), "card": card, "dedup_key": dedup,
        "description_chars": len(desc), "content_sha": hashlib.sha256(desc.encode()).hexdigest()[:16],
    }


def run(ph):
    loader, loader_path = load_loader(ph)
    ph.log.info(f"benchmark loader: {loader_path} ({len(loader.BENCHMARKS)} benchmarks, data in {loader.DATA})")
    types = ph.cfg["literature"]["task_types"]
    rows, bench = [], []
    for key, b in loader.BENCHMARKS.items():
        tt = types.get(key)
        if tt is None or tt.get("task_type") not in TASK_TYPES:
            ph.warn(f"{key}: no valid task type in config literature.task_types — recorded as 'other'")
            tt = {"task_type": "other"}
        b = {**b, "_task_type": tt}
        info = {"benchmark": key, "name": b["name"], "group": b["group"], "source": b.get("repo"),
                "revision": b.get("rev"), "reference": b.get("ref"), "task_type": tt["task_type"],
                "task_subtype": tt.get("subtype"), "granularity": tt.get("granularity"),
                "evaluation_method": tt.get("evaluation"), "local_path": str(loader.DATA / key / str(b.get("rev", ""))[:12])}
        try:
            raw = loader.load_rows(key, 0, ph.seed)
        except urllib.error.HTTPError as e:
            why = f"HTTP {e.code}" + (" (gated: set HF_TOKEN and accept the terms)" if b.get("gated") else "")
            ph.warn(f"{b['name']}: not available — {why}")
            bench.append({**info, "status": f"not available: {why}", "n_tasks": 0})
            continue
        except Exception as e:  # noqa: BLE001 — one benchmark must not stop the others
            ph.warn(f"{b['name']}: could not be read — {type(e).__name__}: {e}")
            bench.append({**info, "status": f"error: {e}", "n_tasks": 0})
            continue
        recs = []
        for r in raw:
            try:
                recs.append(record(key, b, r, loader))
            except Exception as e:  # noqa: BLE001
                ph.warn(f"{key}: item skipped ({type(e).__name__}: {e})")
        ids = Counter(x["task_uid"] for x in recs)
        dup_ids = [k for k, n in ids.items() if n > 1]
        if dup_ids:
            ph.warn(f"{key}: {len(dup_ids)} duplicated item ids (kept first): {dup_ids[:3]}")
            seen = set()
            recs = [x for x in recs if not (x["task_uid"] in seen or seen.add(x["task_uid"]))]
        rows += recs
        langs = Counter(x["language"] for x in recs)
        bench.append({**info, "status": "ok", "n_tasks": len(recs),
                      "languages": "; ".join(f"{k} ({v})" for k, v in langs.most_common()),
                      "with_tests": sum(x["has_tests"] for x in recs),
                      "unique_problems": len({x["dedup_key"] for x in recs})})
        ph.log.info(f"{b['name']:<40} {len(recs):>5} tasks  {tt['task_type']:<16} {dict(langs.most_common(3))}")

    df = pd.DataFrame(rows)
    bdf = pd.DataFrame(bench)
    write_parquet(ph.path("literature_tasks.parquet"), df)
    write_csv(ph.path("literature_benchmarks.csv"), bdf)

    ph.metric("benchmarks_available", int((bdf["status"] == "ok").sum()))
    ph.metric("benchmarks_unavailable", int((bdf["status"] != "ok").sum()))
    ph.metric("tasks", len(df))
    ph.metric("unique_problems", int(df["dedup_key"].nunique()))
    ph.metric("task_types", df["task_type"].value_counts().to_dict())
    ph.metric("languages", df["language"].value_counts().head(12).to_dict())

    # ---------------------------------------------------------------- figures
    ok = bdf[bdf["status"] == "ok"].sort_values("n_tasks", ascending=False)
    tt_order = list(TASK_TYPES)
    F.barh(ok["name"], ok["n_tasks"], "Tasks per benchmark", "tasks", ph.figure("tasks_per_benchmark.png"),
           colors=[F.color(tt_order.index(t)) for t in ok["task_type"]],
           legend={t: F.color(tt_order.index(t)) for t in tt_order if t in set(ok["task_type"])})
    vc = df.drop_duplicates("task_uid")["task_type"].value_counts()
    F.barh(vc.index, vc.values, "Evaluation task types across literature tasks", "tasks", ph.figure("task_types.png"),
           colors=[F.color(tt_order.index(t)) for t in vc.index])
    lv = df["language"].value_counts().head(15)
    F.barh(lv.index, lv.values, "Programming languages of literature tasks", "tasks", ph.figure("languages.png"),
           note="SWE-bench Multilingual / DeepSWE: language of the files changed by the reference patch")
