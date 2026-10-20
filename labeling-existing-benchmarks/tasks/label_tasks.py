"""Download the code benchmarks LLMs are evaluated on and name the TASK each one asks the model to do.

The task is what the model is asked to do (for example "complete a function body from its signature
and docstring", "resolve a GitHub issue with a patch"), not the topic of the code (sorting, parsing).
It is read from the benchmark itself: for every item of every benchmark, the labelling model
sees what the evaluated model receives, the reference answer and how the answer is checked, and names
the task in an imperative phrase about inputs and outputs. The few-shot examples are tasks of none of
the benchmarks below (summarization, test generation, translation), so no answer is suggested.

A benchmark with several scenarios (BigCodeBench Complete / Instruct, LiveCodeBench code generation /
test output prediction / code execution) is several benchmarks here: each scenario is its own task.

Consolidation is done by the same model, in two steps (generate, then consolidate: Wan et al.,
TnT-LLM, KDD 2024). (1) Per scenario, it reads the names given to its items, writes the one task they
share and the items that do not fit it: agreement = share of the items that fit. (2) It groups the
scenarios whose model receives the same kind of input and must produce the same kind of output: one row
of the final list, results/tasks.csv, per group. Embeddings are not used for this: on these phrases
they rate "implement a function" and "implement a library" closer than two wordings of the same task.

Usage: python tasks/label_tasks.py [--model qwen3.8:27b] [--sample 5 (pilot)] [--only swebench_pro ...]
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))      # python -I leaves out the script's folder
from benchmarks import BENCHMARKS, RESULTS, Ollama, _draw, _sha, card, load_rows  # noqa: E402

OUT = RESULTS / "tasks"
PROMPT_VERSION = "benchmark-task-v1"

HEADER = ("Each item below comes from a benchmark that evaluates code language models. It shows what the evaluated "
          "model receives, the reference answer, and how the answer is checked.\n"
          "Name the TASK the benchmark asks the model to do: what it is given and what it must produce, as one "
          "imperative phrase of 4 to 12 words. Do not name the topic or domain of this particular item.\n"
          "Input: what the model is given, in a few words. Output: what it must produce, in a few words.\n\n")

EXAMPLES = [   # tasks of none of the benchmarks above
    ("The model receives:\npublic int max(int[] xs) {\n  int m = xs[0];\n  for (int x : xs) if (x > m) m = x;\n  return m;\n}\n"
     "Reference answer:\nReturns the largest element of a non-empty array.\nChecked by: BLEU against the reference",
     "Summarize a function in one natural-language sentence", "a function's source code", "a one-sentence description"),
    ("The model receives:\ndef slugify(title):\n    return re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')\n"
     "Write unit tests for this function.\nReference answer:\ndef test_spaces():\n    assert slugify('Hello World') == 'hello-world'\n"
     "Checked by: line coverage and mutation score of the generated tests",
     "Write unit tests for a given function", "a function's source code", "a test suite for it"),
    ("The model receives:\nTranslate this Java method to Python:\nstatic boolean isEven(int n) { return n % 2 == 0; }\n"
     "Reference answer:\ndef is_even(n):\n    return n % 2 == 0\nChecked by: the translated code passes the original tests",
     "Translate a function from one programming language to another", "a function in a source language",
     "the equivalent function in a target language"),
    ("The model receives:\na web browser open on an online shop, and this goal:\nBuy the cheapest red umbrella and ship it "
     "to the saved address.\nReference answer:\nclick(search); type('red umbrella'); click(sort by price); click(item 1); "
     "click(buy)\nChecked by: the final state of the shop's database",
     "Reach a goal by acting in a web browser", "a web page and a goal", "a sequence of browser actions"),
]




def _block(text: str, task=None, inp=None, out=None) -> str:
    s = f"### Item\n{text.strip()}\n### Task:"
    if task:
        s += f" {task}\n### Input: {inp}\n### Output: {out}\n\n"
    return s


def prompt(text: str) -> str:
    return HEADER + "".join(_block(*e) for e in EXAMPLES) + _block(text)


_FIELD = {k: re.compile(rf"^[#*\s]*{k}\**\s*:[*\s]*(.*)$", re.IGNORECASE | re.MULTILINE) for k in ("Task", "Input", "Output")}


def parse(answer: str) -> dict:
    text = (answer or "").strip()
    if not _FIELD["Task"].match(text):
        text = "Task: " + text
    out = {}
    for k, rx in _FIELD.items():
        m = rx.search(text)
        out[k.lower()] = m.group(1).strip().strip("\"'`*.").strip() if m else ""
    if out["task"]:
        out["task"] = out["task"][0].upper() + out["task"][1:]
    return out



# ------------------------------------------------------------------------------------- consolidation
SCENARIO_PROMPT = """Below are the tasks a model named for {n} items of the benchmark {name}: each distinct name once, \
with the number of items that got it.

{lines}

Write the one task this benchmark asks the evaluated model to do: an imperative phrase of 4 to 12 words about what \
the model is given and what it must produce, naming no topic of any item. Then list the line numbers whose task is a \
different one. Answer in exactly this format:
Task: ...
Input: ...
Output: ...
Different: <line numbers separated by commas, or none>"""

LIST_PROMPT = """Below are the tasks of {k} code benchmarks, one per line as Id | Task | Input | Output.

{lines}

Group the benchmarks that ask the model to do the same task: the model is given the same kind of input AND must \
produce the same kind of output; if either differs, the tasks differ (a problem statement is not the same input as \
code; one function is not the same output as a whole repository; a predicted value is not the same output as code). Every Id goes in exactly one group. Write one line per group, in exactly this format and \
nothing else:
Task: <imperative phrase of 4 to 12 words> | Input: <...> | Output: <...> | Ids: <ids separated by commas>"""


def consolidate_scenario(llm, model: str, name: str, answers: list[dict]) -> dict:
    """The one task of a benchmark from the names given to its items, and the items that do not fit it."""
    named = [a["task"] for a in answers if a["task"]]
    counts = sorted(Counter(named).items(), key=lambda kv: (-kv[1], kv[0]))
    lines = "\n".join(f"{i}. {t} ({c} items)" for i, (t, c) in enumerate(counts, 1))
    raw = llm.chat(model, SCENARIO_PROMPT.format(n=len(named), name=name, lines=lines), continue_text=False,
                   max_tokens=400, num_ctx=32768)
    m = re.search(r"Different\W*:\s*(.*)", raw, re.IGNORECASE)
    different = sorted({int(x) for x in re.findall(r"\d+", m.group(1))} & set(range(1, len(counts) + 1))) if m else []
    off = sum(counts[i - 1][1] for i in different)
    return {**parse(raw), "agreement": round(1 - off / len(named), 2), "labelled": len(answers), "distinct_names": len(counts),
            "different_tasks": "; ".join(f"{counts[i - 1][0]} ({counts[i - 1][1]})" for i in different), "answer": raw}


def consolidate_list(llm, model: str, scen: list[dict]) -> tuple[list[dict], str]:
    """One list: the benchmarks whose model is given the same kind of input and must produce the same kind of output."""
    ids = {f"B{i}": s for i, s in enumerate(scen, 1)}
    lines = "\n".join(f"{k} | {s['task']} | {s['input']} | {s['output']}" for k, s in ids.items())
    raw = llm.chat(model, LIST_PROMPT.format(k=len(scen), lines=lines), continue_text=False, max_tokens=900)
    groups, used = [], set()
    for line in raw.splitlines():
        parts = {p.split(":", 1)[0].strip(" *-").lower(): p.split(":", 1)[1].strip() for p in line.split("|") if ":" in p}
        members = [k for k in dict.fromkeys(re.findall(r"B\d+", parts.get("ids", ""))) if k in ids and k not in used]
        if parts.get("task") and members:
            used.update(members)
            groups.append({"task": parts["task"].strip("*. "), "input": parts.get("input", ""),
                           "output": parts.get("output", ""), "members": [ids[k] for k in members]})
    for k, s in ids.items():          # a benchmark the answer left out keeps its own task
        if k not in used:
            groups.append({"task": s["task"], "input": s["input"], "output": s["output"], "members": [s]})
    return groups, raw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen3.8:27b")
    ap.add_argument("--sample", type=int, default=0, help="0 = every item (default); n = a pilot of n items per benchmark")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", choices=list(BENCHMARKS))
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    llm = Ollama(os.getenv("OLLAMA_HOST", "localhost:11434"))
    answers_path = OUT / "answers.jsonl"
    done = {}
    if answers_path.exists():
        for line in answers_path.read_text(encoding="utf-8").splitlines():
            try:
                a = json.loads(line)
                done[(a["benchmark"], a["item_id"], a["prompt_sha"], a["model"])] = a
            except (ValueError, KeyError):
                pass

    keys = args.only or list(BENCHMARKS)
    per, skipped = {}, {}
    with answers_path.open("a", encoding="utf-8") as log:
        for key in keys:
            b = BENCHMARKS[key]
            try:
                rows = load_rows(key, args.sample, args.seed)
            except urllib.error.HTTPError as e:
                why = f"HTTP {e.code}" + (": gated dataset, set HF_TOKEN and accept its terms on Hugging Face" if b.get("gated") else "")
                print(f"{b['name']}: not downloaded ({why})", flush=True)
                skipped[key] = why
                continue
            items = [card(key, r) for r in rows]
            if b["kind"] in ("hf_parquet", "hf_jsonl_stream"):
                items = _draw(sorted(items), args.sample, args.seed)
            todo = sum((key, item_id, _sha(prompt(text)), args.model) not in done for item_id, text in items)
            print(f"{b['name']}: {len(rows)} items downloaded, {len(items) - todo} of {len(items)} already labelled, "
                  f"labelling {todo} …", flush=True)
            got = []
            for item_id, text in items:
                p = prompt(text)
                k = (key, item_id, _sha(p), args.model)
                if k not in done:
                    t = time.monotonic()
                    raw = llm.chat(args.model, p)
                    done[k] = {"benchmark": key, "item_id": item_id, "model": args.model, "prompt_version": PROMPT_VERSION,
                               "prompt_sha": k[2], "seconds": round(time.monotonic() - t, 2), "output": raw, **parse(raw)}
                    log.write(json.dumps(done[k], ensure_ascii=False) + "\n")
                    log.flush()
                got.append(done[k])
            print(f"{b['name']}: {len(got)} of {len(items)} labelled", flush=True)
            per[key] = {"size": len(rows), "answers": got}

    print("Consolidating: the task of each benchmark, then one list …", flush=True)
    scen = [{"key": key, **consolidate_scenario(llm, args.model, BENCHMARKS[key]["name"], d["answers"])}
            for key, d in per.items() if any(a["task"] for a in d["answers"])]
    groups, list_answer = consolidate_list(llm, args.model, scen) if scen else ([], "")
    rows = [{"task": g["task"], "input": g["input"], "output": g["output"],
             "benchmarks": "; ".join(BENCHMARKS[s["key"]]["name"] for s in g["members"]),
             "groups": " ".join(sorted({BENCHMARKS[s["key"]]["group"] for s in g["members"]})),
             "task_of_each_benchmark": "; ".join(f"{BENCHMARKS[s['key']]['name']}: {s['task']}" for s in g["members"])}
            for g in groups]
    rows.sort(key=lambda r: (-len(r["benchmarks"].split("; ")), r["task"]))
    (OUT / "consolidation.json").write_text(json.dumps({"model": args.model, "prompt_version": PROMPT_VERSION,
                                                        "list_answer": list_answer, "benchmarks": scen},
                                                       indent=2, ensure_ascii=False), encoding="utf-8")

    with (OUT / "tasks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["task"])
        w.writeheader()
        w.writerows(rows)
    with (OUT / "benchmarks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "group", "reference", "dataset", "revision", "items", "labelled", "task", "agreement",
                    "input", "output", "items_with_another_task"])
        for s in scen:
            b = BENCHMARKS[s["key"]]
            w.writerow([b["name"], b["group"], b["ref"], b["repo"], b["rev"], per[s["key"]]["size"], s["labelled"],
                        s["task"], s["agreement"], s["input"], s["output"], s["different_tasks"]])
        for key, why in skipped.items():
            b = BENCHMARKS[key]
            w.writerow([b["name"], b["group"], b["ref"], b["repo"], b["rev"], "", "", "", "", "", "", f"not downloaded: {why}"])
    md = ["# Tasks of the existing code benchmarks\n",
          f"{len(rows)} tasks from {len(scen)} benchmark scenarios, named by {args.model} from "
          f"{f'a pilot of {args.sample} items of each' if args.sample else 'every item of each'} (what the model receives, the reference answer, how it is checked). Agreement = share of the items "
          f"whose task is the benchmark's task.\n", "| Task | Input | Output | Benchmarks |", "|---|---|---|---|"]
    md += [f"| {r['task']} | {r['input']} | {r['output']} | {r['benchmarks']} |" for r in rows]
    md += ["", "## Each benchmark", "", "| Benchmark | Group | Task | Agreement |", "|---|---|---|---:|"]
    md += [f"| {BENCHMARKS[s['key']]['name']} | {BENCHMARKS[s['key']]['group']} | {s['task']} | {s['agreement']:.0%} |" for s in scen]
    md += [f"| {BENCHMARKS[k]['name']} | {BENCHMARKS[k]['group']} | not downloaded: {why} | |" for k, why in skipped.items()]
    (OUT / "tasks.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n{len(rows)} tasks: {OUT / 'tasks.csv'}  (table: {OUT / 'tasks.md'})")
    for r in rows:
        print(f"  - {r['task']}  [{r['benchmarks']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
