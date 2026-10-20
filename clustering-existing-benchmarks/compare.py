"""How much the models agree: compares the clusterings of results/<model>/items.csv, written by cluster.py.

Each model proposed its own categories, so their names cannot be matched one to one. What can be compared is the
grouping itself: do two models put the same items together? For every pair of models, on the items both labelled:

- ARI, adjusted Rand index (Hubert & Arabie 1985): 1 = the same grouping, 0 = no more alike than chance;
  >= 0.90 excellent, >= 0.80 good, >= 0.65 moderate, below poor (Steinley, Psychological Methods 2004)
- NMI, normalized mutual information (Strehl & Ghosh, JMLR 2002): 1 = one grouping tells the other exactly, 0 = nothing
- mapping: for each category of one model, the category of the other where most of its items went (and how many)

at two levels: the category, and the subcategory (category > subcategory), overall and per benchmark; plus each
benchmark's main categories according to each model, side by side.

Usage: python compare.py --models qwen3.8:27b gemma4:31b muse-glimmer:30b [--out results]
       (reads <out>/<model>/items.csv, writes <out>/comparison/; no model is called)
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from itertools import combinations
from math import comb, log
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))      # python -I leaves out the script's folder
from figures import GRID, INK, INK_2, SEQUENTIAL, SURFACE, slug  # noqa: E402,F401  (also sets the plot style)

PAIR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]     # first three categorical slots (validated all-pairs)
TOP = 3                                            # main categories per benchmark in the profiles


def folder(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


# ------------------------------------------------------------------------------------- measures
def ari(a: list, b: list) -> float:
    """Adjusted Rand index of two labelings of the same items."""
    n = len(a)
    pairs = sum(comb(c, 2) for c in Counter(zip(a, b)).values())
    sa, sb = sum(comb(c, 2) for c in Counter(a).values()), sum(comb(c, 2) for c in Counter(b).values())
    expected = sa * sb / comb(n, 2) if n > 1 else 0
    top = (sa + sb) / 2 - expected
    if top == 0:                       # one group in both (or a single item): identical groupings agree fully
        return 1.0 if len(set(a)) == len(set(b)) else 0.0
    return (pairs - expected) / top


def nmi(a: list, b: list) -> float:
    """Normalized mutual information (arithmetic mean of the two entropies)."""
    n = len(a)
    ca, cb, cab = Counter(a), Counter(b), Counter(zip(a, b))
    h = lambda c: -sum(v / n * log(v / n) for v in c.values())      # noqa: E731
    ha, hb = h(ca), h(cb)
    if ha + hb == 0:
        return 1.0
    mi = sum(v / n * log(v * n / (ca[x] * cb[y])) for (x, y), v in cab.items())
    return 2 * mi / (ha + hb)


# ------------------------------------------------------------------------------------- data
def load(out: Path, models: list[str]) -> dict[str, dict]:
    """model -> {(benchmark_key, item_id): row}"""
    data = {}
    for m in models:
        path = out / folder(m) / "items.csv"
        if not path.exists():
            print(f"{m}: no {path}, left out", file=sys.stderr)
            continue
        with path.open(encoding="utf-8") as f:
            data[m] = {(r["benchmark_key"], r["item_id"]): r for r in csv.DictReader(f)}
    return data


def label(row: dict, level: str) -> str:
    return row["category"] if level == "category" else f"{row['category']} > {row['subcategory'] or '(not split)'}"


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------------------------- figures
def short(m: str) -> str:
    return m.split(":")[0]


def matrix_figure(models: list[str], scores: dict):
    """Pairwise ARI and NMI between the models, at both levels (1 on the diagonal)."""
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.2))
    for row, level in enumerate(("category", "subcategory")):
        for col, measure in enumerate(("ARI", "NMI")):
            ax = axes[row][col]
            m = [[1.0 if a == b else scores[level, tuple(sorted((a, b)))][measure] for b in models] for a in models]
            ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
            for i in range(len(models)):
                for j in range(len(models)):
                    ax.text(j, i, f"{m[i][j]:.2f}", ha="center", va="center", fontsize=10,
                            color=SURFACE if m[i][j] > 0.55 else INK)
            ax.set_xticks(range(len(models)), [short(x) for x in models], rotation=20, ha="right")
            ax.set_yticks(range(len(models)), [short(x) for x in models])
            ax.set_xticks([x - 0.5 for x in range(1, len(models))], minor=True)
            ax.set_yticks([x - 0.5 for x in range(1, len(models))], minor=True)
            ax.grid(which="minor", color=SURFACE, linewidth=2)
            ax.tick_params(which="both", length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{measure}, {level}", loc="left", fontsize=10, color=INK)
    fig.suptitle("Do the models group the same items together?  (1 = same grouping, 0 = chance)",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    return fig


def per_benchmark_figure(models: list[str], per_bench: dict, benches: list[str], level: str):
    """ARI of each pair of models inside each benchmark: one dot per pair, so where they disagree shows."""
    pairs = list(combinations(models, 2))
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(benches) + 1.8))
    for k, (pair, color) in enumerate(zip(pairs, PAIR_COLORS * 3)):
        xs = [per_bench[level, b, pair] for b in benches]
        ax.scatter(xs, [i + (k - 1) * 0.18 for i in range(len(benches))], s=40, color=color, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=f"{short(pair[0])} ~ {short(pair[1])}")
    ax.set_yticks(range(len(benches)), benches)
    ax.invert_yaxis()
    ax.set_xlim(min(-0.1, min(v for v in per_bench.values()) - 0.05), 1.02)
    ax.axvline(0, color=GRID, linewidth=1)
    for x, name in ((0.65, "moderate"), (0.8, "good"), (0.9, "excellent")):
        ax.axvline(x, color=GRID, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(x, -0.9, name, fontsize=7, color=INK_2, ha="center")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_xlabel(f"ARI inside the benchmark, {level} level (1 = the two models group its items the same way)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, frameon=False, fontsize=8)
    ax.set_title(f"Agreement per benchmark ({level})", loc="left", fontsize=12, color=INK, pad=30)
    fig.tight_layout()
    return fig


def mapping_figure(a: str, b: str, common: list, data: dict):
    """Where the items of each category of model a went in model b (share of the row)."""
    rows, cols = Counter(data[a][u]["category"] for u in common), Counter(data[b][u]["category"] for u in common)
    ra, cb = [c for c, _ in rows.most_common()], [c for c, _ in cols.most_common()]
    cell = Counter((data[a][u]["category"], data[b][u]["category"]) for u in common)
    m = [[cell[x, y] / rows[x] for y in cb] for x in ra]
    fig, ax = plt.subplots(figsize=(0.42 * len(cb) + 4.5, 0.36 * len(ra) + 2.8))
    ax.imshow(m, cmap=LinearSegmentedColormap.from_list("seq", SEQUENTIAL), vmin=0, vmax=1, aspect="auto")
    for i, r in enumerate(m):
        for j, v in enumerate(r):
            if v >= 0.1:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color=SURFACE if v > 0.55 else INK)
    ax.set_xticks(range(len(cb)), cb, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ra)), [f"{c} ({rows[c]})" for c in ra], fontsize=8)
    ax.set_xticks([x - 0.5 for x in range(1, len(cb))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(ra))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel(f"category by {a}")
    ax.set_xlabel(f"category by {b}")
    ax.set_title(f"Where the items of each {short(a)} category went in {short(b)} (share of the row)",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def profiles_figure(models: list[str], profiles: dict, benches: list[str]):
    """Each benchmark's main categories according to each model, side by side (text table)."""
    fig, ax = plt.subplots(figsize=(4 + 4.2 * len(models), 0.62 * len(benches) + 1.2))
    ax.axis("off")
    ax.set_xlim(0, 1 + len(models))
    ax.set_ylim(len(benches) + 0.3, -0.9)
    for j, m in enumerate(models):
        ax.text(1.02 + j, -0.45, m, fontsize=10, fontweight="bold", color=INK, va="center")
    for i, b in enumerate(benches):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f1f0ec", zorder=0)
        ax.text(0.98, i, b, ha="right", va="center", fontsize=9, color=INK)
        for j, m in enumerate(models):
            ax.text(1.02 + j, i, "\n".join(f"{c} {v:.0%}" for c, v in profiles[b].get(m, [])), va="center",
                    fontsize=7.2, color=INK_2, linespacing=1.25)
    ax.set_title(f"Main categories of each benchmark, by model (top {TOP})", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------- run
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args(argv)
    data = load(args.out, args.models)
    models = list(data)
    if len(models) < 2:
        print("At least two models with an items.csv are needed to compare.", file=sys.stderr)
        return 1
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    name_of = {u: data[models[0]][u]["benchmark"] for u in common}
    benches = list(dict.fromkeys(name_of[u] for u in common))
    comp = args.out / "comparison"
    (comp / "figures").mkdir(parents=True, exist_ok=True)

    scores, per_bench, rows, bench_rows = {}, {}, [], []
    for level in ("category", "subcategory"):
        for a, b in combinations(models, 2):
            la, lb = [label(data[a][u], level) for u in common], [label(data[b][u], level) for u in common]
            scores[level, tuple(sorted((a, b)))] = {"ARI": ari(la, lb), "NMI": nmi(la, lb)}
            rows.append([level, a, b, len(common), round(ari(la, lb), 3), round(nmi(la, lb), 3),
                         len(set(la)), len(set(lb))])
            for bench in benches:
                us = [u for u in common if name_of[u] == bench]
                xa, xb = [label(data[a][u], level) for u in us], [label(data[b][u], level) for u in us]
                per_bench[level, bench, (a, b)] = ari(xa, xb)
                bench_rows.append([level, bench, a, b, len(us), round(ari(xa, xb), 3), round(nmi(xa, xb), 3)])
    write_csv(comp / "agreement.csv", ["level", "model_a", "model_b", "items", "ARI", "NMI", "groups_a", "groups_b"], rows)
    write_csv(comp / "agreement_per_benchmark.csv", ["level", "benchmark", "model_a", "model_b", "items", "ARI", "NMI"], bench_rows)

    mapping_rows = []
    for a, b in ((x, y) for x in models for y in models if x != y):
        cell = Counter((data[a][u]["category"], data[b][u]["category"]) for u in common)
        sizes = Counter(data[a][u]["category"] for u in common)
        for ca, n in sizes.most_common():
            cb, k = max(((cb, k) for (x, cb), k in cell.items() if x == ca), key=lambda t: t[1])
            mapping_rows.append([a, ca, n, b, cb, k, round(k / n, 3)])
    write_csv(comp / "category_mapping.csv", ["model_a", "category_a", "items", "model_b", "most_items_went_to", "items_there",
                                              "share"], mapping_rows)

    profiles = defaultdict(dict)
    for m in models:
        for bench in benches:
            cnt = Counter(data[m][u]["category"] for u in common if name_of[u] == bench)
            total = sum(cnt.values())
            profiles[bench][m] = [(c, n / total) for c, n in cnt.most_common(TOP)]
    write_csv(comp / "benchmark_profiles.csv", ["benchmark", "model", "rank", "category", "share"],
              [[b, m, r, c, round(v, 3)] for b in benches for m in models for r, (c, v) in enumerate(profiles[b][m], 1)])

    md = ["# How much the models agree\n",
          f"{len(common)} items labelled by {', '.join(models)}, each with its own categories. ARI: 1 = the same grouping, "
          "0 = chance (≥ 0.90 excellent, ≥ 0.80 good, ≥ 0.65 moderate: Steinley 2004). NMI: 1 = one grouping tells the "
          "other exactly.\n", "| Level | Models | ARI | NMI | groups |", "|---|---|---:|---:|---|"]
    md += [f"| {r[0]} | {r[1]} ~ {r[2]} | {r[4]:.2f} | {r[5]:.2f} | {r[6]} vs {r[7]} |" for r in rows]
    md += ["", "## Per benchmark (ARI, category level)\n", "| Benchmark | " + " | ".join(f"{short(a)} ~ {short(b)}" for a, b in combinations(models, 2)) + " |",
           "|---|" + "---:|" * len(list(combinations(models, 2)))]
    md += [f"| {bench} | " + " | ".join(f"{per_bench['category', bench, p]:.2f}" for p in combinations(models, 2)) + " |" for bench in benches]
    md += ["", "## Main categories of each benchmark, by model\n", "| Benchmark | " + " | ".join(models) + " |",
           "|---|" + "---|" * len(models)]
    md += [f"| {b} | " + " | ".join("<br>".join(f"{c} {v:.0%}" for c, v in profiles[b][m]) for m in models) + " |" for b in benches]
    (comp / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    figs = [("agreement", matrix_figure(models, scores)),
            ("agreement_per_benchmark_category", per_benchmark_figure(models, per_bench, benches, "category")),
            ("agreement_per_benchmark_subcategory", per_benchmark_figure(models, per_bench, benches, "subcategory")),
            ("profiles", profiles_figure(models, profiles, benches))]
    figs += [(f"mapping_{slug(short(a))}_to_{slug(short(b))}", mapping_figure(a, b, common, data))
             for a, b in combinations(models, 2)]
    with PdfPages(comp / "figures" / "comparison.pdf") as pdf:
        for name, fig in figs:
            fig.savefig(comp / "figures" / f"{name}.png", dpi=200)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Comparison of {', '.join(models)} on {len(common)} items: {comp} (comparison.md, agreement.csv, "
          "agreement_per_benchmark.csv, category_mapping.csv, benchmark_profiles.csv, figures/)")
    for r in rows:
        print(f"  {r[0]:<11} {short(r[1])} ~ {short(r[2])}: ARI {r[4]:.2f}, NMI {r[5]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
