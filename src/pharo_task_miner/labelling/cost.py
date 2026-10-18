"""What the labelling would cost on a commercial API, computed from the real prompts.

Tokens are counted on the prompts the labeller builds (``coding.build_prompt``, same line and
character caps) with the GPT tokenizer ``o200k_base`` (tiktoken). Without tiktoken or its
vocabulary (first use downloads it), they are approximated by splitting the text into words
(camel case apart), short numbers and runs of punctuation: within 5% of ``o200k_base`` on the
corpus (measured on 20,000 methods and on the instructions), and the report says so. Other providers' tokenizers differ: ``token_factor`` per model in ``api_cost``.

Two scenarios: *this run* (every call the labelling made: labelled methods and check model) and
*the whole corpus* (every method of the population labelled once). For each model of
``api_cost.models``: the standard price, the Batch API, a reasoning model's hidden tokens (billed
as output), and the prompt cache. The cache only applies to a shared prefix of at least
``cache_min_prefix_tokens``; a shorter prefix is assumed padded to that size with more worked
examples. Prices are USD per million tokens and change: update them in config.yaml.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import coding
from .models import NUM_CTX

_PIECES = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d{1,3}|[^\w\s]+|\n\t*")


def tokenizer():
    """(count function, description) for the GPT tokenizer, or a calibrated approximation."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return (lambda s: len(enc.encode(s, disallowed_special=()))), "o200k_base (GPT tokenizer, tiktoken)"
    except Exception:  # not installed, or no network for the vocabulary on first use
        return (lambda s: len(_PIECES.findall(s))), "an approximation of the GPT tokenizer (install tiktoken for exact counts)"


def method_tokens(db, ids: list[int], s: dict, count, tok_name: str, cache: Path) -> dict:
    """Tokens of the method part of the prompt of every method in ``ids``, kept in ``cache``
    (counting the corpus takes a few seconds). Also: the extra tokens of the methods the
    character cap cuts, and how many prompts would not fit in the labeller's context without it."""
    lines, chars = int(s["max_method_lines"]), s.get("max_method_chars")
    key = hashlib.sha256(json.dumps([coding.PROMPT_VERSION, coding.prefix(), lines, chars, s.get("max_tokens"),
                                     tok_name, sorted(ids)]).encode()).hexdigest()[:16]
    if cache.exists():
        try:
            c = json.loads(cache.read_text())
            if c.get("key") == key:
                return c
        except ValueError:
            pass
    prefix = count(coding.prefix())
    tokens, tests, cut, too_long = {}, set(), {}, 0
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for r in db.query("SELECT method_id, class_name, side, source, is_test FROM methods"
                          f" WHERE method_id IN ({','.join('?' * len(chunk))})", chunk):
            m = dict(r)
            n = tokens[m["method_id"]] = count(coding.method_block(m, lines, chars))
            raw = count(coding.method_block(m, lines)) if chars else n
            if raw != n:
                cut[str(m["method_id"])] = raw - n
            too_long += prefix + raw + int(s.get("max_tokens") or 80) > NUM_CTX
            if m["is_test"]:
                tests.add(m["method_id"])
    out = {"key": key, "prefix": prefix, "tokens": {str(k): v for k, v in tokens.items()}, "tests": sorted(tests),
           "cut_by_cap": cut, "too_long_without_cap": too_long}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def scenario(name: str, calls: int, method: int, output: int, prefix: int, c: dict) -> dict:
    return {"scenario": name, "calls": calls, "prefix": prefix + int(c["chat_overhead_tokens"]),
            "method_tokens": method, "output_tokens": output}


def price(sc: dict, m: dict, c: dict) -> dict:
    """USD for one scenario on one model."""
    f, keep = float(m.get("token_factor") or 1.0), 1 - float(c["batch_discount"])
    n, fixed = sc["calls"], sc["prefix"]
    inp = (n * fixed + sc["method_tokens"]) * f / 1e6
    out = sc["output_tokens"] * f / 1e6
    reasoning = n * float(c["reasoning_tokens_per_call"]) / 1e6 * m["output"]
    standard = inp * m["input"] + out * m["output"]
    row = {"model": m["name"], "provider": m.get("provider", ""), "scenario": sc["scenario"],
           "input_mtok": round(inp, 3), "output_mtok": round(out, 3), "standard": standard,
           "batch": standard * keep, "reasoning": standard + reasoning, "reasoning_batch": (standard + reasoning) * keep,
           "cached": None, "cached_batch": None}
    if m.get("cached_input") is not None:
        shared = max(fixed, int(c["cache_min_prefix_tokens"])) * n * f / 1e6
        cached = shared * m["cached_input"] + sc["method_tokens"] * f / 1e6 * m["input"] + out * m["output"]
        row.update(cached=cached, cached_batch=cached * keep)
    lo, hi = c["budget_factor"]
    row["budget_low"], row["budget_high"] = row["batch"] * lo, row["reasoning_batch"] * hi
    return row


def estimate(db, cfg, pop: dict[int, list[int]], ldir: Path, plan: dict | None, summ: dict | None) -> dict:
    """Token counts, prices of every model in both scenarios, and the cost of each saving."""
    c, s = cfg["api_cost"], dict(cfg["labelling"])
    if plan:
        s.update({k: v for k, v in plan["settings"].items() if k in ("max_method_lines", "max_method_chars", "max_tokens")})
    count, tok_name = tokenizer()
    all_ids = sorted(m for v in pop.values() for m in v)
    t = method_tokens(db, all_ids, s, count, tok_name, cfg.data_dir / "processed" / "prompt_tokens.json")
    tokens, tests = t["tokens"], set(t["tests"])
    rows = []
    for name in ("labels.jsonl", "check.jsonl"):
        p = ldir / name
        if p.exists():
            rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    seen, calls = set(), []
    for r in rows:                                      # one call per (model, method): the latest answer
        if (r.get("model"), r["method_id"]) not in seen:
            seen.add((r.get("model"), r["method_id"]))
            calls.append(r)
    out_mean = (sum(count(r.get("output") or "") for r in calls) / len(calls)) if calls else 16.0
    include_tests = bool(plan["population"]["include_tests"]) if plan else bool(s["include_tests"])
    corpus = [m for m in all_ids if include_tests or m not in tests]
    scen = []
    if calls:
        scen.append(scenario("This run", len(calls), sum(tokens.get(str(r["method_id"]), 0) for r in calls),
                             sum(count(r.get("output") or "") for r in calls), t["prefix"], c))
    scen.append(scenario("Whole corpus", len(corpus), sum(tokens[str(m)] for m in corpus),
                         round(out_mean * len(corpus)), t["prefix"], c))
    prices = [price(sc, m, c) for sc in scen for m in c["models"]]

    # the savings, on the reference model and the whole corpus
    ref = next((m for m in c["models"] if m["name"] == c.get("reference_model")), c["models"][0])
    whole = scen[-1]
    base = price(whole, ref, c)
    no_cap = price({**whole, "method_tokens": whole["method_tokens"] + sum(t["cut_by_cap"].get(str(m), 0) for m in corpus)},
                   ref, c)
    no_tests = [m for m in corpus if m not in tests]
    without_tests = price(scenario("Without tests", len(no_tests), sum(tokens[str(m)] for m in no_tests),
                                   round(out_mean * len(no_tests)), t["prefix"], c), ref, c)
    cheapest = min((p for p in prices if p["scenario"] == "Whole corpus"), key=lambda p: p["batch"])
    levers = [("Standard API, as the prompt is", base["standard"]),
              ("Batch API (answers within 24 h)", base["batch"]),
              (f"Prompt cache: a shared prefix of ≥ {c['cache_min_prefix_tokens']} tokens (now {whole['prefix']}), "
               "with more worked examples", base["cached"]),
              ("Batch API + prompt cache", base["cached_batch"]),
              ("Batch API + prompt cache, test methods left out", without_tests["cached_batch"])]
    if calls:
        levers.append(("Label the saturation sample only, as this run did", price(scen[0], ref, c)["standard"]))
    levers += [(f"The cheapest model of the table, {cheapest['model']}", price(whole, next(
                   m for m in c["models"] if m["name"] == cheapest["model"]), c)["standard"]),
               ("Costs more: a reasoning model at low effort instead of the lowest", base["reasoning"]),
               ("Costs more: no character cap on methods", no_cap["standard"])]
    local = None
    if summ and summ.get("seconds_per_method"):
        local = {"model": plan["model"]["name"], "hours": summ["seconds_per_method"] * len(corpus) / 3600}
    return {"tokenizer": tok_name, "prefix": t["prefix"], "chat_overhead": int(c["chat_overhead_tokens"]),
            "tokens": t, "scenarios": scen, "prices": prices, "reference": ref["name"], "levers": levers,
            "output_mean": out_mean, "corpus_methods": len(corpus), "local": local,
            "method_mean_this_run": (scen[0]["method_tokens"] / scen[0]["calls"]) if calls else None,
            "method_mean_corpus": whole["method_tokens"] / max(1, whole["calls"])}
