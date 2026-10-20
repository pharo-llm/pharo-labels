"""Figures of one model's clustering, drawn from <out>/items.csv only: change this file, then redraw them without the
models (./run.sh figures, or python figures.py --out results/<model>).

- figures/overview.png: benchmark x category heatmap, the share of each benchmark's items in each category
- figures/<benchmark>.png: what one benchmark asks: its main categories (dark bars), inside each its main
  subcategories (light bars), and under each subcategory examples of the actual tasks of its items
- figures/all_benchmarks.pdf: all of them, one per page
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import re
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"

# ------------------------------------------------------------------------------------- what to show (edit freely)
TOP_CATEGORIES = 6          # per benchmark figure
TOP_SUBCATEGORIES = 3       # per category
EXAMPLE_TASKS = 2           # actual tasks shown under each subcategory
MIN_CELL_LABEL = 0.05       # overview: shares below this are not written in their cell

# reference palette (one hue for magnitude; text in ink, never in the bar colour)
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#7a7974", "#e4e3df"
CATEGORY_BAR, SUBCATEGORY_BAR = "#2a78d6", "#9ec5f4"
SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                     "xtick.color": INK_2, "ytick.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower().replace("+", "plus")).strip("_")


def read_items(path: Path) -> dict[str, list[dict]]:
    by_benchmark = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_benchmark[r["benchmark"]].append(r)
    return by_benchmark


def _clean(ax, keep=("bottom",)):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    ax.tick_params(length=0)


def overview(by_benchmark: dict[str, list[dict]]):
    """One row per benchmark, one column per category (most frequent first), cells = share of the row."""
    total = Counter(r["category"] for rows in by_benchmark.values() for r in rows)
    cats = [c for c, _ in total.most_common()]
    names = list(by_benchmark)
    share = [[sum(r["category"] == c for r in by_benchmark[b]) / len(by_benchmark[b]) for c in cats] for b in names]
    top = max(map(max, share))
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(cats) + 3.5), 0.42 * len(names) + 2.8))
    im = ax.imshow(share, cmap=LinearSegmentedColormap.from_list("seq", SEQUENTIAL), vmin=0, vmax=top, aspect="auto")
    for i, row in enumerate(share):
        for j, v in enumerate(row):
            if v >= MIN_CELL_LABEL:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5, color=SURFACE if v > 0.45 * top else INK)
    ax.set_xticks(range(len(cats)), cats, rotation=40, ha="right")
    ax.set_yticks(range(len(names)), [f"{b} ({len(by_benchmark[b])})" for b in names])
    ax.set_xticks([x - 0.5 for x in range(1, len(cats))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(names))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)      # 2px gap between cells
    ax.tick_params(which="both", length=0)
    _clean(ax, keep=())
    bar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, format=lambda v, _: f"{v:.0%}")
    bar.outline.set_visible(False)
    bar.set_label("share of the benchmark's items", color=INK_2)
    ax.set_title("What the code of each benchmark is about", loc="left", fontsize=12, color=INK, pad=12)
    fig.tight_layout()
    return fig


def benchmark_figure(name: str, rows: list[dict]):
    """Category (dark bar) > subcategory (light bar) > examples of actual tasks (text), as shares of the items."""
    n = len(rows)
    lines = []      # (kind, label, share)
    cats = Counter(r["category"] for r in rows)
    for c, cn in cats.most_common(TOP_CATEGORIES):
        lines.append(("category", c, cn / n))
        subs = Counter(r["subcategory"] or "(not split)" for r in rows if r["category"] == c)
        for s, sn in subs.most_common(TOP_SUBCATEGORIES):
            lines.append(("sub", s, sn / n))
            tasks = sorted({r["task"] for r in rows if r["category"] == c and (r["subcategory"] or "(not split)") == s and r["task"]},
                           key=len)
            lines += [("task", t, None) for t in tasks[:EXAMPLE_TASKS]]
        rest = len(subs) - TOP_SUBCATEGORIES
        if rest > 0:
            lines.append(("rest", f"{rest} other subcategories", sum(sn for _, sn in subs.most_common()[TOP_SUBCATEGORIES:]) / n))
    left = n - sum(cn for _, cn in cats.most_common(TOP_CATEGORIES))
    if left:
        lines.append(("category-rest", f"{len(cats) - TOP_CATEGORIES} other categories", left / n))

    height = [0.62 if k == "category" else 0.5 if k in ("sub", "rest", "category-rest") else 0.36 for k, _, _ in lines]
    fig, ax = plt.subplots(figsize=(11, sum(height) * 0.42 + 1.7))
    y, ys = 0.0, []
    for h in height:
        ys.append(-(y + h / 2))
        y += h
    vmax = max(v for k, _, v in lines if v is not None)
    for (kind, label, v), yi, h in zip(lines, ys, height):
        if kind == "task":
            ax.text(-0.01 * vmax, yi, "“" + textwrap.shorten(label, 95, placeholder="…") + "”", ha="right", va="center",
                    fontsize=7.2, style="italic", color=INK_3, transform=ax.transData)
            continue
        dark = kind in ("category", "category-rest")
        ax.barh(yi, v, height=h * 0.8, color=CATEGORY_BAR if dark else SUBCATEGORY_BAR)
        ax.text(v + 0.008 * vmax, yi, f"{v:.0%}", va="center", fontsize=8 if dark else 7.5,
                color=INK if dark else INK_2, fontweight="bold" if kind == "category" else "normal")
        ax.text(-0.01 * vmax, yi, label, ha="right", va="center", fontsize=9 if dark else 8,
                color=INK if dark else INK_2, fontweight="bold" if kind == "category" else "normal")
    ax.set_ylim(-y, 0)
    ax.set_xlim(0, vmax * 1.12)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel(f"share of the {n} items")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _clean(ax)
    ax.set_title(f"{name}: what it asks", loc="left", fontsize=12, color=INK, pad=24)
    ax.text(0, 1.015, "dark bar: category · light bar: subcategory · in quotes: actual tasks of its items",
            transform=ax.transAxes, fontsize=8, color=INK_2)
    fig.subplots_adjust(left=0.5, right=0.97, top=1 - 0.75 / fig.get_figheight(), bottom=0.55 / fig.get_figheight())
    return fig


def draw(out: Path = OUT) -> list[Path]:
    by_benchmark = read_items(out / "items.csv")
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written = []
    with PdfPages(figdir / "all_benchmarks.pdf") as pdf:
        figures = [("overview", overview(by_benchmark))]
        figures += [(slug(b), benchmark_figure(b, rows)) for b, rows in by_benchmark.items()]
        for name, fig in figures:
            fig.savefig(figdir / f"{name}.png", dpi=200)
            pdf.savefig(fig)
            plt.close(fig)
            written.append(figdir / f"{name}.png")
    return written + [figdir / "all_benchmarks.pdf"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT, help="one model's results folder, e.g. results/qwen3.8_27b")
    args = ap.parse_args(argv)
    if not (args.out / "items.csv").exists():
        print(f"No {args.out / 'items.csv'}: run ./run.sh first", file=sys.stderr)
        return 1
    for p in draw(args.out):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
