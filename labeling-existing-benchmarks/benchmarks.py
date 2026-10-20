"""The code benchmarks LLMs are evaluated on, used by every part of this folder (tasks/).

- BENCHMARKS: each benchmark (a scenario of a benchmark counts as its own), pinned to a revision
- load_rows: downloads a benchmark once into ../data/already-benchmarks/<benchmark>/<revision>/, then reads it there
- card: one item as the labelling model sees it: what the evaluated model receives, the reference answer, how it
  is checked
- Ollama: the client of the local labelling models

Usage: python benchmarks.py [--only swebench_pro ...]   downloads every benchmark (no model needed)
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "already-benchmarks"      # shared by the projects of pharo-labels
RESULTS = HERE / "results"
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
    base = DATA / key / b["rev"][:12]
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



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download every benchmark into ../data/already-benchmarks/")
    ap.add_argument("--only", nargs="*", choices=list(BENCHMARKS))
    args = ap.parse_args(argv)
    for key in args.only or list(BENCHMARKS):
        try:
            print(f"{BENCHMARKS[key]['name']}: {len(load_rows(key, 0, 42))} items", flush=True)
        except urllib.error.HTTPError as e:
            print(f"{BENCHMARKS[key]['name']}: not downloaded (HTTP {e.code}"
                  + (": gated dataset, set HF_TOKEN and accept its terms on Hugging Face)" if BENCHMARKS[key].get("gated") else ")"),
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
