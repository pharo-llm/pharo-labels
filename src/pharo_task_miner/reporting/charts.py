"""Module 13 — figures. Every figure is drawn from the exported analysis data
(``statistics.json`` / table rows), never from ad-hoc numbers.

Style: one hue (blue) for single-series magnitude, a single-hue sequential
ramp for heatmaps, thin bars, recessive grid, text in ink colours, direct value
labels only where they aid reading, and no dual axes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
import textwrap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DIV = LinearSegmentedColormap.from_list("div_red_blue", ["#e34948", "#f2a3a2", "#f0efec", "#86b6ef", "#256abf"])
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
CONCEPT_COLORS = {"general": SERIES[0], "pharo_specific": SERIES[1], "framework_specific": SERIES[2]}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "font.size": 9,
    })


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _note(ax, text, width=110):
    ax.figure.text(0.01, 0.005, textwrap.fill(text, width), fontsize=7.5, color=INK_2, va="bottom", ha="left")
    ax.figure.subplots_adjust(bottom=0.12)


def _title(ax, text):
    ax.figure.suptitle(text, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)


def hbar(path: Path, labels, values, title, xlabel, note=None, colors=None, fmt="{:.0%}", xlim=None, ci=None):
    _style()
    n = len(labels)
    fig, ax = plt.subplots(figsize=(8.5, max(2.5, 0.28 * n + 1.2)))
    y = range(n)
    ax.barh(list(y), values, height=0.62, color=colors or SERIES[0], xerr=ci, error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 0})
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    _title(ax, title)
    if xlim:
        ax.set_xlim(*xlim)
    for i, v in enumerate(values):
        x = v + (ci[1][i] if ci else 0)  # label sits past the whisker
        ax.text(x, i, "  " + fmt.format(v), va="center", fontsize=7.5, color=INK_2)
    if note:
        _note(ax, note)
    _save(fig, path)


def domain_distribution(path: Path, rows: list[dict], n_repos: int):
    rows = [r for r in rows if r["repositories"] > 0]
    hbar(path, [r["domain"] for r in rows], [r["repositories"] for r in rows],
         "Figure 1 — Application domains of analysed repositories", "Repositories (multi-label)",
         note=f"n = {n_repos} repositories; labels are automatic (pending review) and a repository may carry several.",
         fmt="{:.0f}")


def repository_characteristics(path: Path, profiles: list[dict]):
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        ("Source files per repository (log10)", [p["n_source_files"] for p in profiles if p.get("n_source_files")], True),
        ("Parsed methods per repository (log10)", [p["n_methods"] for p in profiles if p.get("n_methods")], True),
        ("GitHub stars (log10, popularity only)", [p["stars"] + 1 for p in profiles if p.get("stars") is not None], True),
        ("Days since last push", [p["last_push_days"] for p in profiles if p.get("last_push_days") is not None], False),
    ]
    import math
    for ax, (title, vals, log) in zip(axes.flat, panels):
        if vals:
            data = [math.log10(v) for v in vals if v > 0] if log else vals
            ax.hist(data, bins=20, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("Repositories")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Figure 2 — Repository characteristics", x=0.01, ha="left", fontweight="bold", color=INK)
    _save(fig, path)


def taxonomy_tree(path: Path, categories: list[dict], prevalence: dict[str, float]):
    """Indented hierarchy with level-3 repository prevalence bars."""
    _style()
    by_parent = {}
    for c in categories:
        by_parent.setdefault(c.get("parent"), []).append(c)
    lines = []

    def walk(pid, depth):
        for c in sorted(by_parent.get(pid, []), key=lambda c: c["id"]):
            lines.append((depth, c))
            walk(c["id"], depth + 1)
    walk(None, 0)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.19 * len(lines) + 1)))
    ax.axis("off")
    for i, (depth, c) in enumerate(lines):
        y = len(lines) - i
        weight = "bold" if depth == 0 else "normal"
        ax.text(0.01 + depth * 0.03, y, c.get("name") or c["id"], fontsize=8 if depth else 9, fontweight=weight,
                color=INK if depth < 2 else INK_2, va="center")
        p = prevalence.get(c["id"])
        if p is not None:
            ax.barh(y, p * 0.35, left=0.6, height=0.55, color=CONCEPT_COLORS.get(c.get("concept_type"), SERIES[0]))
            ax.text(0.6 + p * 0.35 + 0.005, y, f"{p:.0%}", fontsize=7, color=INK_2, va="center")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, len(lines) + 1)
    _title(ax, "Figure 3 — Programming-operation taxonomy (bars: repository prevalence)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in CONCEPT_COLORS.values()]
    ax.legend(handles, [k.replace("_", " ") for k in CONCEPT_COLORS], loc="upper right", frameon=False, fontsize=8)
    _save(fig, path)


def operation_prevalence(path: Path, rows: list[dict], top: int = 30):
    rows = [r for r in rows if r.get("level") == 3][:top]
    labels = [r["operation"] for r in rows]
    vals = [r["repo_prevalence"] for r in rows]
    ci = [[v - (r["repo_prevalence_ci_low"] or v) for v, r in zip(vals, rows)],
          [(r["repo_prevalence_ci_high"] or v) - v for v, r in zip(vals, rows)]]
    colors = [CONCEPT_COLORS.get(r.get("concept_type"), SERIES[0]) for r in rows]
    hbar(path, labels, vals, "Figure 4 — Most widespread operations (share of repositories)", "Repository-level prevalence",
         note="Automatic rule-based labels (unvalidated unless stated). Whiskers: 95% Wilson interval (sampling variability only). "
              "Colour: blue general, orange Pharo-specific, aqua framework-specific.", colors=colors, xlim=(0, 1.15), ci=ci)


def heatmap(path: Path, cells: list[dict], operations: list[str], domains: list[str], title: str,
            value_key: str = "p_operation_given_domain", diverging: bool = False, label: str = "P(operation | domain)"):
    _style()
    grid = [[None for _ in domains] for _ in operations]
    idx_o = {o: i for i, o in enumerate(operations)}
    idx_d = {d: i for i, d in enumerate(domains)}
    for c in cells:
        if c["operation"] in idx_o and c["domain"] in idx_d:
            grid[idx_o[c["operation"]]][idx_d[c["domain"]]] = c.get(value_key)
    import numpy as np
    arr = np.array([[np.nan if v is None else v for v in row] for row in grid], dtype=float)
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(domains) + 3, 0.26 * len(operations) + 2.5))
    if diverging:
        arr = np.log2(np.clip(arr, 1 / 8, 8))
        im = ax.imshow(arr, cmap=DIV, norm=TwoSlopeNorm(0, -3, 3), aspect="auto")
    else:
        im = ax.imshow(arr, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=40, ha="right")
    ax.set_yticks(range(len(operations)))
    ax.set_yticklabels(operations)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(label)
    if diverging:
        cb.set_ticks([-3, -2, -1, 0, 1, 2, 3])
        cb.set_ticklabels(["≤⅛×", "¼×", "½×", "1× (typical)", "2×", "4×", "≥8×"])
    cb.outline.set_visible(False)
    _title(ax, title)
    _save(fig, path)


def pharo_capabilities(path: Path, rows: list[dict]):
    rows = sorted(rows, key=lambda r: -r["repo_balanced_frequency"])
    labels = [r["feature"] for r in rows]
    vals = [r["repo_balanced_frequency"] for r in rows]
    colors = [CONCEPT_COLORS.get(r.get("concept_type"), SERIES[0]) for r in rows]
    hbar(path, labels, vals, "Figure 6 — Language features (repository-balanced share of methods)",
         "Mean share of a repository's methods using the feature",
         note="Each repository weighs equally. Orange = Pharo-specific capability, blue = general concept.",
         colors=colors, fmt="{:.1%}")


def cross_domain(path: Path, rows: list[dict], top: int = 30):
    rows = [r for r in rows if r.get("level") == 3][:top]
    if not rows:
        return
    considered = max(r["domains_considered"] for r in rows) or 1
    hbar(path, [r["operation"] for r in rows], [r["domains_with_operation"] for r in rows],
         "Figure 7 — Operations present across application domains", "Number of domains with the operation",
         note=f"Domains with at least the minimum number of analysed repositories: {considered}.", fmt="{:.0f}",
         xlim=(0, considered + 1))


def benchmark_coverage(path: Path, rows: list[dict]):
    _style()
    rows = [r for r in rows if r.get("level") == 3]
    order = {"none": 0, "partial": 1, "full": 2}
    rows.sort(key=lambda r: (order[r["existing_benchmark_coverage"]], -(r.get("repo_prevalence") or 0)))
    rows = rows[:40]
    fig, ax = plt.subplots(figsize=(8.5, max(3, 0.26 * len(rows) + 1.5)))
    colors = {"none": "#e34948", "partial": "#eda100", "full": "#1baf7a"}
    vals = [r["covering_tasks"] for r in rows]
    ax.barh(range(len(rows)), [max(v, 0.05) for v in vals], height=0.6,
            color=[colors[r["existing_benchmark_coverage"]] for r in rows])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['operation']}  ({(r.get('repo_prevalence') or 0):.0%} repos)" for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for i, r in enumerate(rows):
        ax.text(max(vals[i], 0.05), i, f"  {r['existing_benchmark_coverage']}: {vals[i]} task(s)", va="center", fontsize=7.5, color=INK_2)
    ax.set_xlabel("Covering benchmark tasks (explicitly mapped)")
    _title(ax, "Figure 8 — Benchmark coverage gaps")
    _note(ax, "Coverage counts only tasks whose operations were mapped explicitly; no keyword matching.")
    _save(fig, path)
