"""What the existing code benchmarks really ask, according to ONE model: every item in a category, a subcategory,
and its actual task. run.sh runs it once per model, each into results/<model>/; compare.py then compares the models.

Three levels, read from the benchmark itself (what the evaluated model receives, the reference answer, how it is
checked), so that each benchmark can be described as "SWE-bench Verified: 30% Databases > query construction >
'fix wrong ORDER BY when a query is combined with union', ...". The model does every step on its own:

1. describe: for every item, its SUBJECT (what the code is about, a noun phrase: "GUI widgets", "date parsing") and
   its actual TASK (what precisely the item asks: "fix a widget that is not redrawn after a resize").
2. categories: it groups the subjects into at most --max-categories categories, each with a definition:
   <out>/categories.csv. Kept once written: edit it (merge, rename, define) and rerun to classify again with your
   list; delete it to let the model propose a new one.
3. classify: it puts every item in one category (or Other).
4. subcategories: inside each category, it groups the actual tasks of its items into at most --max-subcategories
   subcategories: <out>/subcategories.csv (kept and editable like categories.csv).
5. classify again: it puts every item of a category in one of its subcategories.

Generate, then classify (TnT-LLM, Wan et al., KDD 2024). Every answer is saved in <out>/answers/*.jsonl and reused
by a rerun. The figures are drawn from <out>/items.csv by figures.py.

Usage: python cluster.py --model qwen3.8:27b --out results/qwen3.8_27b [--sample 20] [--workers 1] [--only ...]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import urllib.error

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))      # python -I leaves out the script's folder
from benchmarks import BENCHMARKS, Ollama, _draw, _sha, card, load_rows  # noqa: E402

PROMPT_VERSION = "benchmark-clusters-v1"

DESCRIBE_PROMPT = """Below is one item of a benchmark that evaluates code language models: what the evaluated model \
receives, the reference answer and how it is checked.

{card}

Describe this item in exactly two lines and nothing else:
Subject: <what its code is about, a noun phrase of 2 to 6 words: its application domain, or the kind of data or \
concept it works on, e.g. GUI widgets, date parsing, SQL queries, 2D grid paths, HTTP requests, array indexing>
Task: <what precisely the evaluated model must do here, 5 to 15 words; if it fixes or changes existing code, say \
what is wrong or what changes, e.g. "fix a widget that is not redrawn after a resize", "add a timeout option to the \
HTTP client", "return the longest palindromic substring of a string">"""

CATEGORY_CHUNK = """Below are the subjects of {n} items of code benchmarks (what their code is about), each distinct \
subject once with its number of items.

{lines}

Group these subjects into at most {k} categories of what code is about (for example: GUI and user interfaces, arrays \
and lists, dates and times, networking and HTTP). The categories must not overlap, and each must be broad enough to \
hold several subjects. Answer one line per category, in exactly this format and nothing else:
Category: <2 to 4 words> | Definition: <one sentence: what code of this category is about>"""

CATEGORY_MERGE = """Different parts of a set of code benchmarks gave these lists of categories of what code is about:

{lines}

Merge them into ONE list of at most {k} categories that covers every category above: join the ones that mean the \
same, keep the categories from overlapping, and keep each broad enough to hold several subjects. Answer one line per \
category, in exactly this format and nothing else:
Category: <2 to 4 words> | Definition: <one sentence: what code of this category is about>"""

SUB_CHUNK = """Below are the tasks of {n} items of code benchmarks whose code is about {category}, each distinct task \
once with its number of items.

{lines}

Group these tasks into at most {k} subcategories of {category}: what the code does, or what is wrong in it and gets \
fixed. The subcategories must not overlap, and each must be broad enough to hold several tasks. Answer one line per \
subcategory, in exactly this format and nothing else:
Category: <2 to 6 words> | Definition: <one sentence>"""

SUB_MERGE = """Different parts of the items about {category} gave these lists of subcategories:

{lines}

Merge them into ONE list of at most {k} subcategories of {category} that covers every subcategory above: join the \
ones that mean the same, keep them from overlapping, and keep each broad enough to hold several tasks. Answer one \
line per subcategory, in exactly this format and nothing else:
Category: <2 to 6 words> | Definition: <one sentence>"""

CLASSIFY = """Below is one item of a benchmark that evaluates code language models: what the evaluated model \
receives, the reference answer and how it is checked.

{card}

{question}
{options}

Answer with the number of one option only."""

_CATEGORY = re.compile(r"Category\W*:\s*(.+?)\s*\|\s*Definition\W*:\s*(.+)", re.IGNORECASE)
_FIELD = {k: re.compile(rf"^[#*\s-]*{k}\**\s*:[*\s]*(.*)$", re.IGNORECASE | re.MULTILINE) for k in ("Subject", "Task")}


# ------------------------------------------------------------------------------------- saved answers
class Answers:
    """An append-only JSONL of a model's answers, keyed by item, prompt and model: a rerun reuses them."""

    def __init__(self, path: Path):
        self.path, self.lock, self.done = path, threading.Lock(), {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    a = json.loads(line)
                    self.done[(a["benchmark"], a["item_id"], a["prompt_sha"], a["model"])] = a
                except (ValueError, KeyError):
                    pass

    def ask(self, llm, model: str, key: str, item_id: str, text: str, parse, max_tokens: int) -> dict:
        k = (key, item_id, _sha(text), model)
        if k not in self.done:
            t = time.monotonic()
            raw = llm.chat(model, text, continue_text=False, max_tokens=max_tokens)
            a = {"benchmark": key, "item_id": item_id, "model": model, "prompt_version": PROMPT_VERSION,
                 "prompt_sha": k[2], "seconds": round(time.monotonic() - t, 2), "output": raw, **parse(raw)}
            with self.lock:
                self.done[k] = a
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
        return self.done[k]


def run_all(jobs: list, workers: int, label: str) -> list:
    """Runs the (function, args) jobs, ``workers`` at a time, reporting progress; results in order."""
    out, lock, count = [None] * len(jobs), threading.Lock(), [0]

    def one(i):
        out[i] = jobs[i][0](*jobs[i][1])
        with lock:
            count[0] += 1
            if count[0] % 50 == 0 or count[0] == len(jobs):
                print(f"  {label}: {count[0]}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, range(len(jobs))))
    return out


# ------------------------------------------------------------------------------------- parsing
def parse_description(raw: str) -> dict:
    out = {}
    for k, rx in _FIELD.items():
        m = rx.search(raw or "")
        v = m.group(1).strip().strip("\"'`*.").strip() if m else ""
        out[k.lower()] = v[:1].upper() + v[1:]
    return out


def parse_categories(raw: str) -> list[tuple[str, str]]:
    seen, out = set(), []
    for m in _CATEGORY.finditer(raw or ""):
        name, definition = m.group(1).strip("*. "), m.group(2).strip("* ")
        if name and name.lower() not in seen and name.lower() != "other":
            seen.add(name.lower())
            out.append((name, definition))
    return out


def choice(options: list[tuple[str, str]]):
    """Parses an answer that is the number of one of ``options`` (Other when it is none of them)."""
    def parse(raw: str) -> dict:
        m = re.search(r"\d+", raw or "")
        n = int(m.group()) if m else 0
        return {"choice": options[n - 1][0] if 1 <= n <= len(options) else "Other"}
    return parse


# ------------------------------------------------------------------------------------- taxonomy
def propose(llm, model: str, phrases: list[str], k: int, chunk_prompt: str, merge_prompt: str, chunk: int = 300,
            **fmt) -> tuple[list, list]:
    """Categories from the distinct phrases (by chunk, then merged), and the model's raw answers."""
    counts = sorted(Counter(p for p in phrases if p).items(), key=lambda kv: (-kv[1], kv[0]))
    answers, proposed = [], []
    for i in range(0, len(counts), chunk):
        part = counts[i:i + chunk]
        raw = llm.chat(model, chunk_prompt.format(n=sum(c for _, c in part), k=k, lines="\n".join(f"- {t} ({c})" for t, c in part),
                                                  **fmt), continue_text=False, max_tokens=1500, num_ctx=16384)
        answers.append(raw)
        proposed.append(parse_categories(raw))
    if len(proposed) <= 1:
        return (proposed[0] if proposed else []), answers
    lines = "\n\n".join(f"List {i}:\n" + "\n".join(f"{n}: {d}" for n, d in cats) for i, cats in enumerate(proposed, 1))
    raw = llm.chat(model, merge_prompt.format(lines=lines, k=k, **fmt), continue_text=False, max_tokens=1500, num_ctx=16384)
    answers.append(raw)
    return parse_categories(raw), answers


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------------------------- run
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen3.8:27b")
    ap.add_argument("--out", type=Path, required=True, help="this model's results folder")
    ap.add_argument("--sample", type=int, default=0, help="0 = every item (default); n = a pilot of n items per benchmark")
    ap.add_argument("--max-categories", type=int, default=20)
    ap.add_argument("--max-subcategories", type=int, default=8)
    ap.add_argument("--min-items", type=int, default=5, help="a category with fewer items is not split")
    ap.add_argument("--workers", type=int, default=1, help="requests sent to Ollama at the same time")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", choices=list(BENCHMARKS))
    args = ap.parse_args(argv)
    out, model = args.out.resolve(), args.model
    out.mkdir(parents=True, exist_ok=True)
    llm = Ollama(os.getenv("OLLAMA_HOST", "localhost:11434"))
    answers = {name: Answers(out / "answers" / f"{name}.jsonl") for name in ("describe", "category", "subcategory")}

    items, skipped = [], {}
    for key in args.only or list(BENCHMARKS):
        try:
            rows = load_rows(key, args.sample, args.seed)
        except urllib.error.HTTPError as e:
            skipped[key] = f"not downloaded (HTTP {e.code})"
            print(f"{BENCHMARKS[key]['name']}: {skipped[key]}", flush=True)
            continue
        cards = sorted(card(key, r) for r in rows)
        if BENCHMARKS[key]["kind"] in ("hf_parquet", "hf_jsonl_stream"):
            cards = _draw(cards, args.sample, args.seed)
        items += [(key, i, text) for i, text in cards]
    print(f"{model}: {len(items)} items of {len({k for k, _, _ in items})} benchmarks -> {out}", flush=True)

    # 1. subject and actual task of every item
    print(f"1. Describing {len(items)} items (subject, actual task) …", flush=True)
    desc = run_all([(answers["describe"].ask, (llm, model, k, i, DESCRIBE_PROMPT.format(card=t), parse_description, 80))
                    for k, i, t in items], args.workers, "described")
    desc = {(a["benchmark"], a["item_id"]): a for a in desc}

    # 2. categories (kept once written)
    cat_path = out / "categories.csv"
    if not read_csv(cat_path):
        print("2. Proposing the categories …", flush=True)
        cats, raw = propose(llm, model, [d["subject"] for d in desc.values()], args.max_categories, CATEGORY_CHUNK, CATEGORY_MERGE)
        (out / "categories_answer.json").write_text(json.dumps({"model": model, "answers": raw}, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
        if not cats:
            print("The model proposed no category: see categories_answer.json", file=sys.stderr)
            return 1
        write_csv(cat_path, ["category", "definition"], cats)
    else:
        print(f"2. Categories: {cat_path} (kept; delete it for new ones)", flush=True)
    cats = [(r["category"].strip(), r.get("definition", "").strip()) for r in read_csv(cat_path)
            if r.get("category", "").strip() and r["category"].strip().lower() != "other"]
    cats.append(("Other", "Code about none of the categories above."))
    definition = dict(cats)

    # 3. every item in a category
    opts = "\n".join(f"{n}. {c}: {d}" for n, (c, d) in enumerate(cats, 1))
    q = "Categories of what code is about. Which ONE is this item's code about (its subject, not the task it asks)?"
    print(f"3. Putting {len(items)} items in {len(cats)} categories …", flush=True)
    category = {(a["benchmark"], a["item_id"]): a["choice"] for a in run_all(
        [(answers["category"].ask, (llm, model, k, i, CLASSIFY.format(card=t, question=q, options=opts), choice(cats), 16))
         for k, i, t in items], args.workers, "categorised")}

    # 4. subcategories of each category (kept once written)
    members = defaultdict(list)
    for k, i, t in items:
        members[category[k, i]].append((k, i, t))
    sub_path = out / "subcategories.csv"
    subs = defaultdict(list)
    for r in read_csv(sub_path):
        if r.get("subcategory", "").strip() and r["subcategory"].strip().lower() != "other":
            subs[r["category"].strip()].append((r["subcategory"].strip(), r.get("definition", "").strip()))
    split = [c for c, _ in cats if len(members[c]) >= args.min_items]
    for c in split:
        if c not in subs:
            print(f"4. Proposing the subcategories of {c} ({len(members[c])} items) …", flush=True)
            proposed, _ = propose(llm, model, [desc[k, i]["task"] for k, i, _ in members[c]], args.max_subcategories,
                                  SUB_CHUNK, SUB_MERGE, category=c)
            subs[c] = proposed or [(c, definition[c])]
            write_csv(sub_path, ["category", "subcategory", "definition"], [[cat, s, d] for cat, lst in subs.items() for s, d in lst])

    # 5. every item of a category in one of its subcategories
    jobs = []
    for c in split:
        options = subs[c] + [("Other", f"Code about {c} that fits none of the subcategories above.")]
        q = f"This item's code is about {c}. Subcategories of {c}: which ONE describes what this item does or fixes?"
        o = "\n".join(f"{n}. {s}: {d}" for n, (s, d) in enumerate(options, 1))
        jobs += [(answers["subcategory"].ask, (llm, model, k, i, CLASSIFY.format(card=t, question=q, options=o), choice(options), 16))
                 for k, i, t in members[c]]
    print(f"5. Putting {len(jobs)} items in subcategories …", flush=True)
    subcategory = {(a["benchmark"], a["item_id"]): a["choice"] for a in run_all(jobs, args.workers, "subcategorised")}

    # results
    rows = [[BENCHMARKS[k]["name"], k, i, category[k, i], subcategory.get((k, i), ""), desc[k, i]["task"], desc[k, i]["subject"]]
            for k, i, _ in items]
    write_csv(out / "items.csv", ["benchmark", "benchmark_key", "item_id", "category", "subcategory", "task", "subject"], rows)
    md = [f"# What the existing code benchmarks ask, according to {model}\n",
          f"{len(items)} items of {len({k for k, _, _ in items})} benchmarks"
          + (f" ({args.sample} per benchmark, pilot)" if args.sample else "")
          + f". Every step by {model}: the subject and actual task of each item, the categories and subcategories, "
          "and the classification of every item.\n"]
    by_bench = defaultdict(list)
    for r in rows:
        by_bench[r[0]].append(r)
    for b, rs in by_bench.items():
        md += [f"## {b} ({len(rs)} items)\n"]
        for c, cn in Counter(r[3] for r in rs).most_common():
            md.append(f"- **{c}** {cn / len(rs):.0%}")
            for s, sn in Counter(r[4] for r in rs if r[3] == c).most_common():
                md.append(f"  - *{s or '(not split)'}* {sn / len(rs):.0%}")
                md += [f"    - {r[5]} `{r[2]}`" for r in rs if r[3] == c and r[4] == s]
        md.append("")
    md += [f"## {BENCHMARKS[k]['name']}: {why}\n" for k, why in skipped.items()]
    (out / "clusters.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    from figures import draw
    draw(out)
    print(f"{model}: results in {out} (clusters.md, items.csv, categories.csv, subcategories.csv, figures/)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
