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

Usage: python label_benchmarks.py [--model qwen3.8:27b] [--sample 5 (pilot)] [--only swebench_pro ...]
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data"
OUT = HERE / "results"
PROMPT_VERSION = "benchmark-task-v1"
HF = "https://huggingface.co/datasets"

# scenario -> source. kind: hf_parquet | hf_jsonl_stream | hf_task_dirs | github_task_dirs
# group: "frontier" = in 2026 model cards of large AI companies; "classic" = reference benchmark of the surveys
BENCHMARKS = {
    "humanevalplus": dict(name="HumanEval+", group="classic", kind="hf_parquet", repo="evalplus/humanevalplus",
                          rev="d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
                          files=["data/test-00000-of-00001-5973903632b82d40.parquet"],
                          ref="Chen et al. 2021; Liu et al., NeurIPS 2023"),
    "mbppplus": dict(name="MBPP+", group="classic", kind="hf_parquet", repo="evalplus/mbppplus",
                     rev="b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
                     files=["data/test-00000-of-00001-d5781c9c51e02795.parquet"],
                     ref="Austin et al. 2021; Liu et al., NeurIPS 2023"),
    "bigcodebench_complete": dict(name="BigCodeBench-Complete", group="classic", kind="hf_parquet",
                                  repo="bigcode/bigcodebench", rev="b74c0d0bf70d2c0bc459be537895cca163007f1a",
                                  files=["data/v0.1.4-00000-of-00001.parquet"], ref="Zhuo et al., ICLR 2025"),
    "bigcodebench_instruct": dict(name="BigCodeBench-Instruct", group="classic", kind="hf_parquet",
                                  repo="bigcode/bigcodebench", rev="b74c0d0bf70d2c0bc459be537895cca163007f1a",
                                  files=["data/v0.1.4-00000-of-00001.parquet"], ref="Zhuo et al., ICLR 2025"),
    # code generation: the six release files (4.5 GB with their tests) are streamed; only the problem is kept
    "livecodebench_codegen": dict(name="LiveCodeBench code generation", group="frontier", kind="hf_jsonl_stream",
                                  repo="livecodebench/code_generation_lite", rev="0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
                                  files=[f"test{i}.jsonl" for i in ("", 2, 3, 4, 5, 6)], ref="Jain et al., ICLR 2025",
                                  keep=["question_id", "question_title", "question_content", "starter_code",
                                        "public_test_cases", "platform"]),
    "livecodebench_testoutput": dict(name="LiveCodeBench test output prediction", group="frontier", kind="hf_parquet",
                                     repo="livecodebench/test_generation", rev="6f3ac40bbecf81eba15899139d279b077f2816fd",
                                     files=["test.parquet"], ref="Jain et al., ICLR 2025"),
    "livecodebench_execution": dict(name="LiveCodeBench code execution", group="frontier", kind="hf_parquet",
                                    repo="livecodebench/execution-v2", rev="ff6ea0e2a638001006ddcc41259eff23a4283fb2",
                                    files=["data/test-00000-of-00001.parquet"], ref="Jain et al., ICLR 2025"),
    "swebench_verified": dict(name="SWE-bench Verified", group="frontier", kind="hf_parquet",
                              repo="princeton-nlp/SWE-bench_Verified", rev="c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
                              files=["data/test-00000-of-00001.parquet"], ref="Jimenez et al., ICLR 2024; OpenAI 2024"),
    "swebench_pro": dict(name="SWE-bench Pro (public)", group="frontier", kind="hf_parquet", repo="ScaleAI/SWE-bench_Pro",
                         rev="2d52cb3df914a3fcf80c7f66738b3a88ae37fc50", files=["data/default/test-00000-of-00001.parquet"],
                         ref="Deng et al. (Scale AI) 2025"),
    "swebench_multilingual": dict(name="SWE-bench Multilingual", group="frontier", kind="hf_parquet",
                                  repo="SWE-bench/SWE-bench_Multilingual", rev="846e647b9f33c0b51b739d005d13d85493c9af09",
                                  files=["data/test-00000-of-00001.parquet"], ref="Yang et al., SWE-smith 2025"),
    "terminalbench2": dict(name="Terminal-Bench 2.0", group="frontier", kind="hf_task_dirs",
                           repo="harborframework/terminal-bench-2.0", rev="f2e8c75e23add71613117eecc9498f53bcd7e04e",
                           files=["instruction.md", "solution/solve.sh", "tests/test.sh"],
                           ref="Merrill et al., ICLR 2026"),
    "nl2repo": dict(name="NL2Repo-Bench", group="frontier", kind="github_task_dirs",
                    repo="multimodal-art-projection/NL2RepoBench", rev="781a1da1ee41fb8edb0bed22f586d69111610edf",
                    dir="test_files", files=["start.md", "test_commands.json"], ref="arXiv 2512.12730"),
    # gated on Hugging Face: needs HF_TOKEN and the terms accepted on the dataset page; skipped otherwise
    "deepswe": dict(name="DeepSWE", group="frontier", kind="hf_parquet", repo="datacurve/deep-swe",
                    rev="6d6f134460c137e24c6bb7e1e69954116ea9dbb3", files=["data/test-00000-of-00001.parquet"],
                    ref="Datacurve 2026", gated=True),
}

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


# ------------------------------------------------------------------------------------- download
def _get(url: str, token: str | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 * 2 ** attempt)


def _cached(path: Path, fetch) -> bytes:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(fetch())
        tmp.replace(path)
    return path.read_bytes()


def load_rows(key: str, sample: int, seed: int) -> list[dict]:
    """The rows of a benchmark (task-directory benchmarks: every task directory, or a pilot)."""
    b = BENCHMARKS[key]
    base = CACHE / key / b["rev"][:12]
    token = os.getenv("HF_TOKEN") if b.get("gated") else None
    if b["kind"] in ("hf_parquet", "hf_jsonl_stream"):
        rows = []
        for f in b["files"]:
            url = f"{HF}/{b['repo']}/resolve/{b['rev']}/{f}"
            if b["kind"] == "hf_parquet":
                import pyarrow.parquet as pq
                rows += pq.read_table(io.BytesIO(_cached(base / Path(f).name, lambda: _get(url, token)))).to_pylist()
            else:                                    # stream: keep only the fields the card shows
                def stream():
                    out = []
                    with urllib.request.urlopen(url, timeout=300) as r:
                        for line in r:
                            if line.strip():
                                d = json.loads(line)
                                out.append(json.dumps({k: d.get(k) for k in b["keep"]}, ensure_ascii=False))
                    return ("\n".join(out) + "\n").encode()
                data = _cached(base / (Path(f).stem + ".fields.jsonl"), stream).decode()
                rows += [json.loads(x) for x in data.splitlines() if x.strip()]
        return rows
    if b["kind"] == "hf_task_dirs":
        listing = json.loads(_cached(base / "listing.json", lambda: _get(f"https://huggingface.co/api/datasets/{b['repo']}/tree/{b['rev']}?recursive=false")))
        dirs = sorted(e["path"] for e in listing if e["type"] == "directory" and not e["path"].startswith("."))
        url = lambda d, f: f"{HF}/{b['repo']}/resolve/{b['rev']}/{d}/{f}"   # noqa: E731
    else:
        api = f"https://api.github.com/repos/{b['repo']}/contents/{b['dir']}?ref={b['rev']}"
        listing = json.loads(_cached(base / "listing.json", lambda: _get(api)))
        dirs = sorted(e["name"] for e in listing if e["type"] == "dir")
        url = lambda d, f: f"https://raw.githubusercontent.com/{b['repo']}/{b['rev']}/{b['dir']}/{d}/{f}"   # noqa: E731
    rows = []
    for d in _draw(dirs, sample, seed):
        row = {"task_id": d}
        for f in b["files"]:
            try:
                row[f] = _cached(base / d / f, lambda: _get(url(d, f))).decode("utf-8", "replace")
            except urllib.error.HTTPError:
                row[f] = ""
        rows.append(row)
    return rows


def _draw(xs: list, sample: int, seed: int) -> list:
    """Every item (sample 0, the default), or a random pilot of ``sample`` items."""
    if sample <= 0 or sample >= len(xs):
        return list(xs)
    return random.Random(seed).sample(list(xs), sample)


# ------------------------------------------------------------------------------------- item cards
def _cut(text, n: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= n else text[:n].rstrip() + "\n[...]"


def _first(row: dict, *keys) -> str:
    return next((str(row[k]) for k in keys if row.get(k)), "")


def card(key: str, r: dict) -> tuple[str, str]:
    """(item id, card): what the evaluated model receives, the reference answer, how it is checked."""
    C = 1500
    if key == "humanevalplus":
        return r["task_id"], (f"The model receives:\n{_cut(r['prompt'], C)}\nReference answer:\n{_cut(r['canonical_solution'], 800)}\n"
                              f"Checked by: hidden unit tests run on the completed function")
    if key == "mbppplus":
        return f"Mbpp/{r['task_id']}", (f"The model receives:\n{_cut(r['prompt'], C)}\n{_cut((r.get('test_list') or [''])[0], 300)}\n"
                                        f"Reference answer:\n{_cut(r['code'], 800)}\nChecked by: hidden unit tests run on the function")
    if key == "bigcodebench_complete":
        return r["task_id"], (f"The model receives:\n{_cut(r['complete_prompt'], C)}\nReference answer:\n{_cut(r['canonical_solution'], 800)}\n"
                              f"Checked by: unit tests run on the completed function")
    if key == "bigcodebench_instruct":
        return r["task_id"], (f"The model receives:\n{_cut(r['instruct_prompt'], C)}\nReference answer:\n"
                              f"{_cut(r['code_prompt'] + r['canonical_solution'], 800)}\nChecked by: unit tests run on the function")
    if key == "livecodebench_codegen":
        tests = _cut(r.get("public_test_cases"), 300)
        return r["question_id"], (f"The model receives:\n{r['question_title']}\n{_cut(r['question_content'], C)}\n"
                                  + (f"Starter code:\n{r['starter_code']}\n" if (r.get("starter_code") or "").strip() else "")
                                  + f"Reference answer: (none published)\nChecked by: hidden input/output tests, e.g. {tests}")
    if key == "livecodebench_testoutput":
        t = json.loads(r["test"])[0] if r.get("test") else {}
        return f"{r['question_id']}/{r['test_id']}", (
            f"The model receives:\n{_cut(r['question_content'], C)}\n{r['starter_code']}\n"
            f"assert {r['function_name']}({_cut(t.get('input'), 300)}) == ??\n"
            f"Reference answer:\n{_cut(t.get('output'), 300)}\nChecked by: exact match with the expected output (no code is run)")
    if key == "livecodebench_execution":
        return r["id"], (f"The model receives:\n{_cut(r['code'], C)}\nassert {r['input']} == ??\n"
                         f"Reference answer:\n{_cut(r['output'], 300)}\nChecked by: exact match with the value the code returns")
    if key in ("swebench_verified", "swebench_pro", "swebench_multilingual"):
        return r["instance_id"], (f"The model receives:\nthe repository {r['repo']} at commit {r['base_commit'][:10]}, and this issue:\n"
                                  f"{_cut(r['problem_statement'], C)}\nReference answer:\n{_cut(r['patch'], 900)}\n"
                                  f"Checked by: tests that must now pass: {_cut(r.get('FAIL_TO_PASS') or r.get('fail_to_pass'), 200)}; "
                                  f"tests that must still pass")
    if key == "terminalbench2":
        return r["task_id"], (f"The model receives:\na terminal in a Docker container, and this instruction:\n{_cut(r['instruction.md'], C)}\n"
                              f"Reference answer:\n{_cut(r['solution/solve.sh'], 700)}\n"
                              f"Checked by: {_cut(r['tests/test.sh'], 300)}")
    if key == "nl2repo":
        return r["task_id"], (f"The model receives:\nan empty workspace and this document:\n{_cut(r['start.md'], C)}\n"
                              f"Reference answer:\nthe original repository of the library '{r['task_id']}'\n"
                              f"Checked by: running the library's own test suite: {_cut(r['test_commands.json'], 200)}")
    if key == "deepswe":
        return (_first(r, "instance_id", "task_id", "id", "name") or "?"), (
            f"The model receives:\nthe repository, and this instruction:\n{_cut(_first(r, 'instruction', 'problem_statement', 'prompt'), C)}\n"
            f"Reference answer:\n{_cut(_first(r, 'patch', 'solution', 'solution_patch'), 900)}\nChecked by: the task's tests")
    raise KeyError(key)


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


# ------------------------------------------------------------------------------------- Ollama
class Ollama:
    def __init__(self, url: str):
        self.url = url if url.startswith("http") else "http://" + url

    def _post(self, path: str, payload: dict, timeout: int = 600) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read())
            except (urllib.error.URLError, TimeoutError):
                if attempt == 3:
                    raise
                time.sleep(2 * 2 ** attempt)

    def chat(self, model: str, text: str, continue_text: bool = True, max_tokens: int = 120, num_ctx: int = 8192) -> str:
        if continue_text:
            text = ("Continue the text below: write only what comes after its last line, in exactly the format of the "
                    "examples, and nothing else.\n\n" + text)
        payload = {"model": model, "stream": False, "think": False, "messages": [{"role": "user", "content": text}],
                   "options": {"temperature": 0, "seed": 42, "num_predict": max_tokens, "num_ctx": num_ctx,
                               "stop": ["### Item"] if continue_text else []}}
        return self._post("/api/chat", payload)["message"]["content"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


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
            print(f"{b['name']}: {len(rows)} items downloaded, {len(items)} labelled", flush=True)
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
