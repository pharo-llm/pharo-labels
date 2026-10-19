"""Step 4, deductive classification: several models classify a fresh random sample with a fixed
codebook; their agreement decides whether the experts get the work.

Why two phases: open codes from different models cannot be compared fairly ("Setter" and
"Accessor" are one task named twice), and LLM coding agrees much better with humans when a
codebook is given (Shah et al. 2025, inductive vs deductive coding; TnT-LLM, KDD 2024: generate
the taxonomy, then classify with it). The inductive phase (step 3) builds the codebook; this
phase measures it.

1. ``codebook.csv``: the consolidated codebook. Step 3 writes ``codebook_draft.csv`` (the
   inductive categories, a definition written by the labelling model, examples); the researcher
   copies it to ``codebook.csv`` and consolidates it (merge, rename, define). Without
   ``codebook.csv`` the draft is used as it is, and the report says so. "Other" is always added.
2. A new random sample (stratified by repository, as in step 3), disjoint from the methods the
   codebook was built from, so the codebook is not tested on its own data.
3. Each model of ``classify_models`` (different model families: errors of models from one family
   are correlated, Kim et al., ICML 2025) picks one category per method, independently.
4. Agreement: Krippendorff's alpha (any number of annotators, missing answers), Fleiss' kappa,
   Gwet's AC1 (robust to one dominant category), pairwise Cohen's kappa. Each method gets the
   majority label and an agreement level (3/3, 2/3, none).
5. Prompt sensitivity: the first model classifies ``selfcheck_size`` methods again with the
   codebook and the examples in another order; its agreement with itself is reported.
6. Gate: alpha < ``min_alpha`` -> NOT_RELIABLE.md and no expert files. Otherwise the experts get
   a sample stratified by agreement level, more where the models disagree (Neyman allocation on
   guessed error rates), with known weights, so every expert answer counts in unbiased estimates
   (Gligoric et al., NAACL 2025, choose which items humans label; validity kept by weighting).
   Methods where all models agree are checked too: shared errors hide there.

Files in ``results/labelling/classify/``: sample.json, <model>.jsonl, selfcheck.jsonl,
labels.csv (every model's label, majority, level), agreement.json, gold_selection.csv (strata
and weights: not for the experts).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

from ..storage.provenance import write_json
from . import agreement as ag, coding, sampling
from .export import _csv
from .saturation import _append, _read_jsonl

OTHER = "Other"
OTHER_DEF = "None of the categories above fits the method."
STOP = ["\n"]
CODEBOOK_COLS = ["category", "definition", "inductive_methods", "inductive_share", "examples"]


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- codebook
DEFINE = ("Below are Pharo methods that a model labelled with the programming task category \"{name}\".\n"
          "{examples}\n\nWrite a one-sentence definition of the category \"{name}\": what kind of programming "
          "task it covers, general enough for any Pharo method. Answer with the definition only.")


def draft_codebook(summary: dict, labels: list[dict], model, out: Path, log) -> Path:
    """codebook_draft.csv from the inductive taxonomy; definitions by ``model`` (cached)."""
    cache_p = out / "definitions.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    by = defaultdict(list)
    for r in labels:
        by[r["category"]].append(r)
    rows = []
    for t in summary["taxonomy"]:
        if not t["in_taxonomy"]:
            continue
        name = t["category"]
        ex = by[name][:]
        random.Random(42).shuffle(ex)
        examples = " | ".join(f"{r['signature']}: {r['summary']}" for r in ex[:5])
        if name not in cache and model is not None:
            text = model.complete(DEFINE.format(name=name, examples="\n".join(
                f"- {r['signature']}: {r['summary']}" for r in ex[:8])), 80, ["\n\n"])
            cache[name] = " ".join(text.split()).strip("\"' ")
            write_json(cache_p, cache)
        rows.append({"category": name, "definition": cache.get(name, ""), "inductive_methods": t["methods"],
                     "inductive_share": t["share"], "examples": examples})
    p = out / "codebook_draft.csv"
    _csv(p, rows, CODEBOOK_COLS)
    log(f"Codebook draft: {len(rows)} categories -> {p} (copy it to codebook.csv to consolidate it)")
    return p


def load_codebook(out: Path) -> tuple[list[dict], str]:
    """(categories with definitions, "consolidated" | "draft"); "Other" is always the last."""
    import csv
    p, kind = out / "codebook.csv", "consolidated"
    if not p.exists():
        p, kind = out / "codebook_draft.csv", "draft"
    if not p.exists():
        raise FileNotFoundError("No codebook: run ./run.sh label first.")
    with p.open(encoding="utf-8-sig", newline="") as f:
        rows = [{k: (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]
    seen, cats = set(), []
    for r in rows:
        name = r.get("category", "")
        if name and coding.key(name) not in seen and coding.key(name) != coding.key(OTHER):
            seen.add(coding.key(name))
            cats.append({"category": name, "definition": r.get("definition", "")})
    cats.append({"category": OTHER, "definition": OTHER_DEF})
    return cats, kind


# --------------------------------------------------------------------------- prompt
def build_prompt(codebook: list[dict], method: dict, max_lines: int, max_chars: int | None, variant: int = 0) -> str:
    """Codebook, worked examples whose category is in the codebook, then the method. ``variant``
    > 0 shuffles the codebook (except Other) and the examples: the prompt-sensitivity check."""
    cats = codebook[:-1]
    names = {coding.key(c["category"]): c["category"] for c in cats}
    examples = [(cls, src, names[coding.key(t)]) for cls, src, t, _ in coding.EXAMPLES if coding.key(t) in names]
    if variant:
        rng = random.Random(variant)
        cats, examples = cats[:], examples[:]
        rng.shuffle(cats)
        rng.shuffle(examples)
    lines = ["Each Pharo method below is classified by the programming task it performs. Choose exactly one "
             "category of the codebook; answer Other when none fits.\n", "Codebook:"]
    lines += [f"- {c['category']}: {c['definition']}" for c in cats + [codebook[-1]]]
    text = "\n".join(lines) + "\n\n"
    for cls, src, cat in examples:
        text += f"### Method of {cls}\n{src.strip()}\n### Category: {cat}\n\n"
    cls = method["class_name"] + (" class" if method.get("side") == "class" else "")
    return text + f"### Method of {cls}\n{coding.method_source(method, max_lines, max_chars)}\n### Category:"


def parse(text: str, codebook: list[dict]) -> str | None:
    """The codebook category the answer names, or None."""
    t = re.sub(r"^[#*\s]*(Category\**\s*:)?[*\s]*", "", (text or "").strip(), flags=re.I)
    t = t.splitlines()[0].strip().strip("\"'`*.").strip() if t else ""
    if not t:
        return None
    k = coding.key(t)
    best, score = None, 0.0
    for c in codebook:
        ck = coding.key(c["category"])
        if k == ck:
            return c["category"]
        r = SequenceMatcher(None, k, ck).ratio()
        if r > score:
            best, score = c["category"], r
    return best if score >= 0.85 else None


# --------------------------------------------------------------------------- the phase
class Classification:
    def __init__(self, out: Path, settings: dict, pop: dict[int, list[int]], exclude: set[int], records, log):
        self.out, self.s, self.records, self.log = out / "classify", settings, records, log
        self.out.mkdir(parents=True, exist_ok=True)
        self.codebook, self.codebook_kind = load_codebook(out)
        p = self.out / "sample.json"
        n = int(settings["classify_sample_size"])
        if p.exists() and json.loads(p.read_text())["size"] == n:
            self.sample = json.loads(p.read_text())["method_ids"]
        else:
            self.sample = sampling.draw(pop, n, int(settings["random_seed"]) + 1000, exclude)
            write_json(p, {"size": n, "method_ids": self.sample, "excluded_inductive": len(exclude)})

    def prompt(self, m: dict, variant: int = 0) -> str:
        return build_prompt(self.codebook, m, self.s["max_method_lines"], self.s.get("max_method_chars"), variant)

    def _label(self, model, ids: list[int], path: Path, variant: int = 0) -> dict[int, dict]:
        rows = {r["method_id"]: r for r in _read_jsonl(path)}
        recs = self.records(ids)
        todo = [m for m in ids if m not in rows or rows[m].get("prompt_sha") != _sha(self.prompt(recs[m], variant))]
        if todo:
            self.log(f"  {model.name}: {len(todo)} methods to classify")
        for i, mid in enumerate(todo, 1):
            prompt = self.prompt(recs[mid], variant)
            t = time.monotonic()
            raw = model.complete(prompt, 24, STOP)
            rows[mid] = {"method_id": mid, "model": model.name, "category": parse(raw, self.codebook),
                         "output": raw, "prompt_sha": _sha(prompt), "seconds": round(time.monotonic() - t, 3)}
            _append(path, rows[mid])
            if i % 50 == 0:
                self.log(f"    {i}/{len(todo)}")
        return {m: rows[m] for m in ids}

    def run(self, models: list) -> dict:
        labels = {}
        for model in models:
            labels[model.name] = self._label(model, self.sample, self.out / f"{slug(model.name)}.jsonl")
        names = [m.name for m in models]
        k = int(self.s.get("selfcheck_size") or 0)
        self_ids = self.sample[:k]
        selfcheck = self._label(models[0], self_ids, self.out / "selfcheck.jsonl", variant=1) if k else {}
        return self.analyse(names, labels, selfcheck)

    # ------------------------------------------------------------------ statistics
    def analyse(self, names: list[str], labels: dict[str, dict[int, dict]], selfcheck: dict[int, dict]) -> dict:
        units = {m: [labels[n][m]["category"] for n in names] for m in self.sample}
        maj = {m: ag.majority(u) for m, u in units.items()}
        rows = [{"method_id": m, **{n: units[m][i] or "" for i, n in enumerate(names)},
                 "majority": maj[m][0] or "", "level": maj[m][1]} for m in self.sample]
        _csv(self.out / "labels.csv", rows, ["method_id", *names, "majority", "level"])
        levels = Counter(l for _, l in maj.values())
        pairs = [{"models": [a, b], "cohen_kappa": ag.cohen_kappa([u[i] for u in units.values()], [u[j] for u in units.values()]),
                  "agreement": round(sum(u[i] == u[j] and u[i] is not None for u in units.values()) / len(units), 3)}
                 for i, a in enumerate(names) for j, b in enumerate(names) if i < j]
        first = names[0]
        sc = ag.cohen_kappa([labels[first][m]["category"] for m in selfcheck], [selfcheck[m]["category"] for m in selfcheck]) \
            if selfcheck else None
        n = len(self.sample)
        dist = {}
        for c in [c["category"] for c in self.codebook]:
            dist[c] = {nm: round(sum(labels[nm][m]["category"] == c for m in self.sample) / n, 4) for nm in names}
            dist[c]["majority"] = round(sum(maj[m][0] == c for m in self.sample) / n, 4)
        alpha = ag.krippendorff_alpha(list(units.values()))
        res = {"models": names, "codebook": self.codebook, "codebook_kind": self.codebook_kind, "sample": n,
               "krippendorff_alpha": alpha, "fleiss_kappa": ag.fleiss_kappa(list(units.values())),
               "gwet_ac1": ag.gwet_ac1(list(units.values())), "pairs": pairs,
               "levels": dict(levels), "no_answer": {nm: sum(labels[nm][m]["category"] is None for m in self.sample) for nm in names},
               "selfcheck": {"model": first, "methods": len(selfcheck), "cohen_kappa": sc,
                             "agreement": round(sum(labels[first][m]["category"] == selfcheck[m]["category"] for m in selfcheck)
                                                / len(selfcheck), 3) if selfcheck else None},
               "distribution": dist, "min_alpha": float(self.s["min_alpha"]),
               "seconds_per_method": {nm: round(sum(labels[nm][m]["seconds"] for m in self.sample) / n, 3) for nm in names}}
        unanimous = all(len(set(u)) == 1 and u[0] is not None for u in units.values())   # alpha undefined: no variation
        res["reliable"] = (alpha is not None and alpha >= res["min_alpha"]) or (alpha is None and unanimous)
        write_json(self.out / "agreement.json", res)
        return res

    # ------------------------------------------------------------------ expert sample
    def allocate(self, strata: dict[str, list[int]]) -> dict[str, int]:
        """Neyman allocation: m_h proportional to N_h * sqrt(e_h (1 - e_h)), e_h a guessed error
        rate of the majority label in the stratum; at least ``gold_min_per_stratum``, at most N_h."""
        total = min(int(self.s["gold_size"]), sum(map(len, strata.values())))
        guess = {"3/3": 0.1, "2/3": 0.35, "none": 0.6, **(self.s.get("gold_error_guess") or {})}
        low = int(self.s.get("gold_min_per_stratum") or 0)
        alloc = {h: min(len(v), low) for h, v in strata.items()}
        free = {h for h in strata if len(strata[h]) > alloc[h]}
        while sum(alloc.values()) < total and free:
            left = total - sum(alloc.values())
            w = {h: len(strata[h]) * math.sqrt(guess.get(h, 0.35) * (1 - guess.get(h, 0.35))) for h in free}
            tw = sum(w.values()) or 1.0
            moved = False
            for h in sorted(free):
                add = min(len(strata[h]) - alloc[h], max(1, round(left * w[h] / tw)), total - sum(alloc.values()))
                if add > 0:
                    alloc[h] += add
                    moved = True
            free = {h for h in free if len(strata[h]) > alloc[h]}
            if not moved:
                break
        return alloc

    def gold_sample(self, res: dict) -> tuple[list[int], dict]:
        import csv
        with (self.out / "labels.csv").open(encoding="utf-8") as f:
            rows = {int(r["method_id"]): r for r in csv.DictReader(f)}
        strata = defaultdict(list)
        for m in self.sample:
            strata[rows[m]["level"]].append(m)
        alloc = self.allocate(strata)
        rng = random.Random(int(self.s["random_seed"]) + 2000)
        chosen, sel = [], []
        for h in sorted(strata):
            pick = rng.sample(sorted(strata[h]), alloc[h])
            chosen += pick
            for m in pick:
                sel.append({"method_id": m, "stratum": h, "stratum_methods": len(strata[h]), "stratum_checked": alloc[h],
                            "weight": round(len(strata[h]) / alloc[h], 4), **{n: rows[m][n] for n in res["models"]},
                            "majority": rows[m]["majority"]})
        _csv(self.out / "gold_selection.csv", sel, list(sel[0]) if sel else ["method_id"])
        rng.shuffle(chosen)
        return chosen, {h: {"N": len(strata[h]), "m": alloc[h]} for h in strata}


EXPERT_README = """# Expert validation of the Pharo task taxonomy

Language models classified methods of real Pharo projects by the programming task they perform,
with the codebook below. Your judgement is the reference the models are measured against, and it
decides the final taxonomy. Work alone: do not discuss your answers with another expert before
both are returned.

1. **method_review.csv**: {m} methods. Open the `url` (or read `source`) and write in
   `expert_category` the codebook category that fits best, or `Other` with a name in
   `other_name`. The models' answers are not shown on purpose.
2. **codebook_review.csv**: {n} categories with their definitions and examples.
   - `keep_yes_no`: is this a real, distinct kind of programming task in Pharo?
   - `rename_to`: a better name, if any.
   - `merge_into`: the category it is really the same as, if any.
   - `definition_fix`: a better definition, if needed.

`examples.md` shows every category with methods and their source code.

Please send back the two CSV files.
"""

NOT_RELIABLE = """# Codebook not reliable yet

{k} models classified {n} new methods with the codebook. Krippendorff's alpha = {alpha}, below
{t} (Krippendorff's threshold for tentative conclusions). Gwet's AC1 = {ac1}. No expert files
were written: the experts would be measuring a codebook the models cannot apply consistently.

What to change, then `./run.sh classify` again (answers are reused when the prompt is unchanged):
- `classify/labels.csv`: which categories do the models confuse? Merge them, or make their
  definitions in `codebook.csv` sharper (copy `codebook_draft.csv` first if it is not there);
- check `agreement.json`: one model far from the others (pairs) can be replaced;
- a low self-check means the answers depend on the prompt order: the definitions are too vague.

To send the expert files anyway, set `MIN_ALPHA=0` in run.sh (and say so in the paper).
"""


def expert_pack(cls: Classification, res: dict, labels_inductive: list[dict], ev: Path, log) -> dict:
    gold, strata = cls.gold_sample(res)
    recs = cls.records(gold)
    names = [c["category"] for c in cls.codebook]
    _csv(ev / "method_review.csv",
         [{"method_id": m, "repository": recs[m]["full_name"], "method": recs[m]["signature"], "url": recs[m]["url"],
           "source": coding.method_source(recs[m], 60), "expert_category": "", "other_name": "", "comments": ""}
          for m in gold], ["method_id", "repository", "method", "url", "source", "expert_category", "other_name", "comments"])
    by = defaultdict(list)
    for r in labels_inductive:
        by[r["category"]].append(r)
    cb_rows, md = [], ["# Codebook with examples\n", "Categories: " + ", ".join(names) + "\n"]
    examples = int(cls.s.get("examples_per_category") or 5)
    for c in cls.codebook:
        ex = by.get(c["category"], [])[:]
        random.Random(42).shuffle(ex)
        cb_rows.append({"category": c["category"], "definition": c["definition"],
                        "examples": " | ".join(f"{r['signature']}: {r['summary']}" for r in ex[:examples]),
                        "keep_yes_no": "", "rename_to": "", "merge_into": "", "definition_fix": "", "comments": ""})
        md.append(f"## {c['category']}\n\n{c['definition']}\n")
        for r in ex[:3]:
            md.append(f"- **{r['signature']}**: {r['summary']}")
        md.append("")
    _csv(ev / "codebook_review.csv", cb_rows, list(cb_rows[0]))
    (ev / "examples.md").write_text("\n".join(md), encoding="utf-8")
    (ev / "README.md").write_text(EXPERT_README.format(m=len(gold), n=len(names)), encoding="utf-8")
    log(f"Expert files: {len(gold)} methods (" + ", ".join(f"{h}: {s['m']} of {s['N']}" for h, s in sorted(strata.items()))
        + f") -> {ev}")
    return {"methods": len(gold), "strata": strata}
