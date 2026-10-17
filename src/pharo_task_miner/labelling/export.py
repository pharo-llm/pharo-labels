"""Final taxonomy, statistics, figures and the files Pharo experts fill in.

``results/labelling/``
  taxonomy.csv / taxonomy.md      the categories with their share of the corpus (95% Wilson
                                  intervals, finite population correction) and examples
  labels.csv                      every labelled method with its final category
  summary.json                    the numbers the report uses
``results/labelling/expert_validation/``
  README.md                       what to do, for the experts
  taxonomy_review.csv             one row per category: keep / rename / merge / split / definition
  method_review.csv               labelled methods to judge one by one (with the check model's label)
  candidate_review.csv            names proposed too rarely to become categories
  examples.md                     every category with example methods and their source code
``results/figures/``              saturation_curve.png, category_distribution.png
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random

from ..storage.provenance import write_json
from . import coding, sampling

UNCLASSIFIED = "(rare name: not a category)"
NO_ANSWER = "(no usable answer)"

# reference palette of the dataviz method (single series, light surface)
SERIES, SURFACE, INK, MUTED, GRID = "#2a78d6", "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def _csv(path: Path, rows: list[dict], cols: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def final_category(run, row: dict) -> str:
    if not row["raw_task"]:
        return NO_ANSWER
    return run.codebook.final(row["raw_task"]) or UNCLASSIFIED


def export(run, plan: dict, check_rows: list[dict] | None, out: Path, figures: Path, log) -> dict:
    labels = list(run.labels.values())
    recs = run.records([r["method_id"] for r in labels])
    N = plan["population"]["methods"]
    conf = plan["sample_size"]["confidence"]
    for r in labels:
        r["category"] = final_category(run, r)
    taxonomy = run.codebook.taxonomy()
    n_all = len(labels)
    initial = [r for r in labels if r["round"] == 0]
    by_cat = defaultdict(list)
    for r in labels:
        by_cat[r["category"]].append(r)

    # ------------------------------------------------------------------ taxonomy with statistics
    rows = []
    for name in sorted(by_cat, key=lambda c: (-len(by_cat[c]), c)):
        rs = by_cat[name]
        k0 = sum(1 for r in rs if r["round"] == 0)
        lo, hi = sampling.wilson(len(rs), n_all, conf, N)
        lo0, hi0 = sampling.wilson(k0, len(initial), conf, N)
        info = run.codebook.cats.get(name, {})
        rows.append({
            "category": name, "in_taxonomy": name in taxonomy, "seed": info.get("seed", False),
            "below_min_share": name in taxonomy and len(rs) < float(run.s.get("min_share") or 0) * n_all,
            "first_round": info.get("first_round"), "admitted_round": info.get("admitted_round"),
            "methods": len(rs), "share": round(len(rs) / n_all, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "estimated_methods_in_corpus": round(len(rs) / n_all * N),
            "methods_initial_sample": k0, "share_initial_sample": round(k0 / len(initial), 4) if initial else None,
            "ci_low_initial": round(lo0, 4), "ci_high_initial": round(hi0, 4),
            "repositories": len({r["repository"] for r in rs}),
            "test_share": round(sum(r["is_test"] for r in rs) / len(rs), 3),
            "name_variants": "; ".join(sorted(info.get("variants", []))),
            "merged_by_meaning": "; ".join(f"{k} ({v})" for k, v in sorted((info.get("merged") or {}).items())),
            "example_summaries": " | ".join(r["summary"] for r in rs[:3])})
    cols = list(rows[0]) if rows else []
    _csv(out / "taxonomy.csv", rows, cols)
    unused_seeds = [n for n, c in run.codebook.cats.items() if c["seed"] and n not in taxonomy]

    flat = [{"method_id": r["method_id"], "round": r["round"], "repository": r["repository"],
             "method": r["signature"], "is_test": r["is_test"], "url": recs[r["method_id"]]["url"],
             "summary": r["summary"], "proposed_task": r["raw_task"], "category": r["category"]} for r in labels]
    _csv(out / "labels.csv", flat, list(flat[0]) if flat else [])

    # ------------------------------------------------------------------ check model agreement
    agreement = None
    check_by = {c["method_id"]: c for c in (check_rows or [])}
    if check_rows:
        pairs = [(run.labels[m]["category"], c["category"] or NO_ANSWER) for m, c in check_by.items()]
        both = [(a, b) for a, b in pairs if a in taxonomy]
        agree = sum(a == b for a, b in both)
        po = agree / len(both) if both else 0.0
        ca, cb = Counter(a for a, _ in both), Counter(b for _, b in both)
        pe = sum(ca[k] * cb[k] for k in ca) / (len(both) ** 2) if both else 0.0
        agreement = {"model": check_rows[0]["model"], "items": len(pairs), "items_with_a_category": len(both),
                     "agreement": round(po, 3), "cohen_kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None,
                     "check_model_no_answer": sum(1 for _, b in pairs if b == NO_ANSWER)}

    # ------------------------------------------------------------------ expert validation pack
    ev = out / "expert_validation"
    k = int(run.s.get("examples_per_category") or 5)
    rng = random.Random(run.s["random_seed"])
    tax_rows = []
    for row in rows:
        if not row["in_taxonomy"]:
            continue
        ex = by_cat[row["category"]][:]
        rng.shuffle(ex)
        tax_rows.append({"category": row["category"] + (" †" if row["below_min_share"] else ""),
                         "methods_labelled": row["methods"],
                         "share_of_corpus": f"{row['share']:.1%} [{row['ci_low']:.1%}, {row['ci_high']:.1%}]",
                         "repositories": row["repositories"],
                         "names_merged_into_it": "; ".join(sorted(set(row["name_variants"].split("; ")) - {row["category"]}
                                                                  - {""})),
                         "examples": " | ".join(f"{r['signature']}: {r['summary']}" for r in ex[:k]),
                         "keep_yes_no": "", "rename_to": "", "merge_into": "", "split_into": "",
                         "definition": "", "comments": ""})
    _csv(ev / "taxonomy_review.csv", tax_rows, list(tax_rows[0]) if tax_rows else ["category"])

    review = run.review_ids(int(run.s["method_review_size"]))
    rrecs = run.records(review)
    mrows = [{"method_id": m, "repository": rrecs[m]["full_name"], "method": rrecs[m]["signature"],
              "url": rrecs[m]["url"], "source": coding.method_source(rrecs[m], 60),
              "model_summary": run.labels[m]["summary"], "model_category": run.labels[m]["category"],
              "check_model_category": (check_by.get(m) or {}).get("category") or "",
              "category_correct_yes_no": "", "correct_category": "", "comments": ""} for m in review]
    _csv(ev / "method_review.csv", mrows, list(mrows[0]) if mrows else ["method_id"])

    cand = [{"proposed_name": n, "methods": c["support"], "first_round": c["first_round"],
             "examples": " | ".join(f"{r['signature']}: {r['summary']}" for r in labels
                                    if r["raw_task"] and run.codebook.match(r["raw_task"]) == n)[:600],
             "should_be_a_category_yes_no": "", "belongs_to_category": "", "comments": ""}
            for n, c in sorted(run.codebook.cats.items(), key=lambda kv: (-kv[1]["support"], kv[0]))
            if not run.codebook.in_taxonomy(n) and c["support"] > 0]
    _csv(ev / "candidate_review.csv", cand, list(cand[0]) if cand else ["proposed_name"])

    md = ["# Task categories with examples\n",
          f"{len(taxonomy)} categories found by `{plan['model']['name']}` in {n_all} methods of the "
          f"{plan['population']['repositories']} repositories of the study corpus.\n"]
    for row in tax_rows:
        md.append(f"## {row['category']}\n")
        md.append(f"{row['methods_labelled']} labelled methods, {row['share_of_corpus']} of the corpus, "
                  f"{row['repositories']} repositories.\n")
        ex = by_cat[row["category"].removesuffix(" †")][:]
        random.Random(run.s["random_seed"]).shuffle(ex)
        for r in ex[:3]:
            rec = recs[r["method_id"]]
            md.append(f"**{r['signature']}** ([{rec['full_name']}]({rec['url']})): {r['summary']}\n")
            md.append("```smalltalk\n" + coding.method_source(rec, 25) + "\n```\n")
    (ev / "examples.md").write_text("\n".join(md), encoding="utf-8")
    (out / "taxonomy.md").write_text("\n".join(md), encoding="utf-8")
    (ev / "README.md").write_text(EXPERT_README.format(n=len(taxonomy), m=len(mrows), c=len(cand)), encoding="utf-8")

    # ------------------------------------------------------------------ figures
    figs = _figures(run.curve, rows, taxonomy, figures, plan, log)

    summary = {
        "methods_labelled": n_all, "initial_sample": len(initial), "rounds": len(run.curve) - 1,
        "categories": len(taxonomy),
        "categories_at_min_share": sum(1 for r in rows if r["in_taxonomy"] and not r["below_min_share"]),
        "seed_categories": coding.SEED_CATEGORIES, "unused_seed_categories": unused_seeds,
        "candidates": len(cand), "no_answer": len(by_cat.get(NO_ANSWER, [])),
        "rare_names": len(by_cat.get(UNCLASSIFIED, [])),
        "labelled_tests": sum(r["is_test"] for r in labels), "repositories_covered": len({r["repository"] for r in labels}),
        "seconds_per_method": round(sum(r["seconds"] for r in labels) / n_all, 3) if n_all else None,
        "margin_of_error_all_labelled": round(sampling.margin_for(n_all, N, conf), 4),
        "agreement": agreement, "taxonomy": rows, "curve": run.curve, "figures": figs,
        "files": {"taxonomy": str(out / "taxonomy.csv"), "labels": str(out / "labels.csv"),
                  "expert_validation": str(ev)}}
    write_json(out / "summary.json", summary)
    return summary


def _figures(curve: list[dict], rows: list[dict], taxonomy: list[str], figures: Path, plan: dict, log) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib is not installed: figures skipped.")
        return {}
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})

    def frame(ax):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    out = {}
    # saturation curve: cumulative categories against cumulative labelled methods
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    frame(ax)
    xs = [c["methods_total"] for c in curve]
    ys = [c["categories_total"] for c in curve]
    dense = len(xs) > 15                                   # markers only when they stay apart
    ax.plot(xs, ys, color=SERIES, linewidth=2, marker=None if dense else "o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axvline(xs[0], color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.annotate(f"initial sample (n = {xs[0]})", (xs[0], ys[0]), xytext=(8, -16), textcoords="offset points",
                color=MUTED, fontsize=9)
    for c in curve[1:]:
        if c["new_categories"]:
            ax.annotate(f"+{c['new_categories']}", (c["methods_total"], c["categories_total"]), xytext=(0, 8),
                        textcoords="offset points", ha="center", color=MUTED, fontsize=8)
    ax.annotate(f"{ys[-1]} categories", (xs[-1], ys[-1]), xytext=(6, -14), textcoords="offset points",
                color=INK, fontsize=9)
    ax.set_xlabel("Methods labelled")
    ax.set_ylabel("Task categories")
    ax.set_title("Saturation: categories found as more methods are labelled", loc="left", color=INK, fontsize=11)
    ax.set_ylim(0, max(ys) * 1.15 + 1)
    fig.tight_layout()
    p = figures / "saturation_curve.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out["saturation_curve"] = str(p)

    # share of each category (with its 95% interval), largest first
    tx = [r for r in rows if r["in_taxonomy"]][::-1]
    if tx:
        fig, ax = plt.subplots(figsize=(8, 0.32 * len(tx) + 1.2), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        y = range(len(tx))
        share = [r["share"] * 100 for r in tx]
        ax.barh(y, share, height=0.62, color=SERIES, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(share, y, xerr=[[s - r["ci_low"] * 100 for s, r in zip(share, tx)],
                                    [r["ci_high"] * 100 - s for s, r in zip(share, tx)]],
                    fmt="none", ecolor=MUTED, elinewidth=1, capsize=2)
        ax.set_yticks(list(y), [r["category"] for r in tx])
        for i, r in enumerate(tx):
            ax.text(r["ci_high"] * 100 + 0.4, i, f"{r['share']:.1%}", va="center", color=MUTED, fontsize=8)
        ax.set_xlabel("Share of the corpus methods (%), 95% interval")
        ax.set_title("Task categories in the study corpus", loc="left", color=INK, fontsize=11)
        fig.tight_layout()
        p = figures / "category_distribution.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out["category_distribution"] = str(p)
    return out


EXPERT_README = """# Expert validation of the Pharo task taxonomy

A small language model read methods of real Pharo projects and named the programming task each
one performs. Methods were drawn at random until new methods stopped producing new categories
(saturation). Your judgement decides what the final taxonomy is. Please do not look at the
other files before filling these in.

1. **taxonomy_review.csv**: {n} categories, one row each, with examples.
   - `keep_yes_no`: is this a real, distinct kind of programming task in Pharo?
   - `rename_to`: a better name, if any.
   - `merge_into`: the category it is really the same as, if any.
   - `split_into`: names of the categories it should be split into, separated by `;`.
   - `definition`: one sentence defining the category (please write one for every kept category).
2. **method_review.csv**: {m} methods drawn at random from the labelled ones.
   Open the `url`, read the `source`, then fill `category_correct_yes_no` and, when it is wrong,
   `correct_category` (an existing category, or a new name).
3. **candidate_review.csv**: {c} names the model proposed too rarely to become categories.
   `should_be_a_category_yes_no`, or `belongs_to_category` when one of the categories covers it.

`examples.md` shows every category with three methods and their source code.
"""
