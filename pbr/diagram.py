"""The pipeline figure of the README (pipeline.png): from the repositories to the two final taxonomies.

    python -m pbr diagram         (re-draw it after changing the pipeline or config.yaml)

Settings (thresholds, confidence, margin, annotators) are read from config.yaml; the corpus numbers are the ones
the pipeline measured on the corpus snapshot (repository_scores.csv / method_filters_summary.csv when present).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mp

from . import figures as F
from .agreement import cochran, saturation_patience
from .common import ROOT, load_config

PHARO, LIT, SHARED = "#2a78d6", "#eb6834", "#52514e"


LINE, HEAD, PAD, GAP = 0.0205, 0.040, 0.016, 0.040      # in axis units (the figure is drawn on a 1 x 1 axis)


def height(lines) -> float:
    return HEAD + PAD + len(lines) * LINE + PAD * 0.6


def box(ax, x, top, w, title, lines, color, num=None) -> float:
    """A titled box whose top is at ``top``; returns its bottom."""
    h = height(lines)
    y = top - h
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", facecolor="white",
                                   edgecolor=color, lw=1.8))
    ax.add_patch(mp.FancyBboxPatch((x, top - HEAD), w, HEAD, boxstyle="round,pad=0,rounding_size=0.012",
                                   facecolor=color, edgecolor=color, lw=1.8))
    ax.text(x + 0.012, top - HEAD / 2, (f"{num}  " if num else "") + title, color="white", fontsize=11, weight="bold",
            va="center")
    for i, line in enumerate(lines):
        ax.text(x + 0.014, top - HEAD - PAD - (i + 0.5) * LINE, line, fontsize=9, color=F.INK, va="center")
    return y


def arrow(ax, x, y0, y1, color=SHARED):
    ax.add_patch(mp.FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=16, color=color, lw=1.8))


def draw(path: Path | None = None) -> Path:
    cfg = load_config()
    r, d, a = cfg["repositories"], cfg["discovery"], [x["model"] for x in cfg["llm"]["annotators"]]
    n_pharo = cochran(690_057, float(d["confidence"]), float(d["margin"]), float(d["proportion"]))
    n_lit = cochran(4_372, float(d["confidence"]), float(d["margin"]), float(d["proportion"]))
    pat = saturation_patience(float(d["min_share"]), int(d["round_size"]), float(d["confidence"]))
    conf = f"{float(d['confidence']):.0%} confidence, ±{float(d['margin']):.0%}"
    W, L, Rx = 0.455, 0.025, 0.52
    c1 = ["1,585 Pharo repositories of the corpus snapshot  → repos.txt",
          "hard filters: GitHub forks, mirrors (≥ 50% copied code), no methods",
          "weighted quality score of 11 criteria (reaper, Munaiah et al. 2017):",
          "   community · history · tests · maintenance · documentation · size ·",
          "   originality · licence · issues · loadability · popularity (low weight)",
          f"score ≥ {r['threshold']}  → selected_repos.txt (≈ 1,087 repositories)"]
    c2 = ["12 code benchmarks LLMs are evaluated on (pinned revisions):",
          "HumanEval+, MBPP+, BigCodeBench (2), LiveCodeBench (3),",
          "SWE-bench Verified / Pro / Multilingual, Terminal-Bench 2.0, NL2Repo",
          "6,433 tasks = 4,372 distinct problems (a problem counted once)",
          "per task: what the model receives, reference, tests, language,",
          "and the evaluation task type"]
    c3 = ["≈ 1.84 M methods of the selected repositories, with provenance",
          "(repository, commit, file, line), class context and syntax tree",
          "decided on the AST + the class definition:",
          "   exclude: duplicates, mirrors, unparsable, getters, setters,",
          "   constants, empty / placeholders, generated, trivial initialize",
          "   review (kept, marked): forwarding, initialisation, long, dependencies",
          "   keep separately: test methods",
          "→ production methods + test methods (counts: method_filters_summary.csv)"]
    c4p = ["strata = repository domains; ≤ 4 methods per repository;",
           "a random order = the pool of the saturation loop",
           f"start: {n_pharo} methods ({conf}; Cochran + finite population)",
           "+ targeted methods (rare areas) + test methods (separate)",
           "representativeness check: AST features, size, domains —",
           "   population vs design-weighted sample (95% intervals)"]
    c4l = ["proportional to each benchmark's number of problems;",
           "a random order = the pool of the saturation loop",
           f"start: {n_lit} problems ({conf}; Cochran + finite population)",
           "representativeness check: share of every benchmark and",
           "   task type — population vs sample (95% intervals)"]
    c5 = lambda n: [", ".join(a),                                                  # noqa: E731
                    "each, alone: category › subcategory › behaviour per item",
                    f"start sample ({n}), then rounds of {d['round_size']} new items until",
                    f"{pat} rounds in a row bring no new behaviour (saturation)",
                    "then each consolidates its own taxonomy",
                    "→ 3 taxonomies: taxonomy.png · .json · .csv · items.csv"]
    c6 = ["the 3 annotators' taxonomies merged: similarity =",
          "0.6 × text embeddings + 0.4 × overlap of the items they hold;",
          "an editor keeps only same-meaning merges, weighted by items;",
          "hierarchy by items; harmonisation (≤ 16 categories);",
          "stable IDs (C001 / S001 / B001); expert review workbook",
          "→ taxonomy_v1.json / taxonomy_v1.png (●●○ + items per concept)"]
    heights = [max(height(c1), height(c2)), height(c3), max(height(c4p), height(c4l)), height(c5(n_pharo)), height(c6)]
    total = 0.11 + sum(heights) + GAP * 4 + 0.04
    fig = F.plt.figure(figsize=(15, 15 * total * 0.92))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total)
    ax.axis("off")
    top = total - 0.025
    ax.text(0.025, top, "Pharo benchmark research — stage 1: from the repositories to the two final taxonomies",
            fontsize=16, weight="bold", va="top")
    ax.text(0.025, top - 0.038, "./run.sh: setup first (models, Pharo image, data), then phases 1–6; every phase resumes "
            "and writes results/NN_<phase>/", fontsize=10.5, color=F.INK_2, va="top")
    top -= 0.085
    b1 = box(ax, L, top, W, "Repository selection", c1, PHARO, "1")
    b2 = box(ax, Rx, top, W, "Literature ingestion", c2, LIT, "2")
    top = min(b1, b2) - GAP
    arrow(ax, L + W / 2, b1, top)
    b3 = box(ax, L, top, W, "Corpus and method filters (on the AST)", c3, PHARO, "3")
    top3 = top
    top = b3 - GAP
    arrow(ax, L + W / 2, b3, top)
    arrow(ax, Rx + W / 2, b2, top, LIT)
    ax.text(Rx + W / 2 + 0.01, (b2 + top) / 2, "the tasks", fontsize=9, color=F.INK_2, va="center")
    b4p = box(ax, L, top, W, "Sampling, representative of Pharo", c4p, PHARO, "4")
    b4l = box(ax, Rx, top, W, "Sampling, representative of the benchmarks", c4l, LIT, "4")
    top = min(b4p, b4l) - GAP
    arrow(ax, L + W / 2, b4p, top)
    arrow(ax, Rx + W / 2, b4l, top, LIT)
    b5p = box(ax, L, top, W, "Discovery on the Pharo methods", c5(n_pharo), PHARO, "5")
    b5l = box(ax, Rx, top, W, "Discovery on the benchmark tasks", c5(n_lit), LIT, "5")
    top = min(b5p, b5l) - GAP
    arrow(ax, L + W / 2, b5p, top)
    arrow(ax, Rx + W / 2, b5l, top, LIT)
    box(ax, L, top, W, "FINAL TAXONOMY of the Pharo methods", c6, PHARO, "6")
    box(ax, Rx, top, W, "FINAL TAXONOMY of the benchmark tasks", c6, LIT, "6")
    ax.text(0.5, 0.012, "Blue: Pharo methods · orange: existing benchmarks. ≈: measured on the corpus snapshot; "
            "sample sizes and thresholds come from config.yaml (python -m pbr diagram redraws this figure).",
            ha="center", fontsize=9, color=F.INK_2)
    path = Path(path or ROOT / "pipeline.png")
    return F.save(fig, path)


if __name__ == "__main__":
    print(draw())
