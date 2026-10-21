"""One figure style for every phase (static PNG): fixed categorical order, single-hue sequential ramp,
two-pole diverging ramp with a grey midpoint, recessive grid and axes, text in ink colours (never the
series colour), one y-axis per chart. Every figure is saved as results/NN_phase/figures/<name>.png."""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#a8a7a2"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#c2491b", "#f2b393", "#ecebe7", "#86b6ef", "#1c5cab"])
STATUS = {"good": "#1f8f4e", "warning": "#d99a00", "serious": "#e26a1b", "critical": "#c9302c", "neutral": "#8a8984"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "axes.labelsize": 10, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "font.family": "DejaVu Sans", "legend.frameon": False, "legend.fontsize": 9, "text.color": INK,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
})


def color(i: int) -> str:
    """Categorical slot i (fixed order); past the eighth, the 'Other' grey."""
    return SERIES[i] if i < len(SERIES) else OTHER


def wrap(s, n: int = 34) -> str:
    return "\n".join(textwrap.wrap(str(s), n)) or str(s)


def save(fig, path: Path, note: str | None = None) -> Path:
    """Save a figure (with an optional source/caveat line at the bottom) and close it."""
    if note:
        fig.text(0.01, -0.02, note, fontsize=7.5, color=INK_2, ha="left", va="top")   # below the axes; bbox="tight" keeps it
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def barh(labels, values, title: str, xlabel: str, path: Path, *, colors=None, errors=None, note=None,
         fmt="{:.0f}", max_items: int = 40, label_width: int = 42, xlim=None, legend: dict | None = None):
    """Horizontal bars, largest first at the top, value printed at the end of each bar."""
    labels, values = list(labels)[:max_items], list(values)[:max_items]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(9, max(2.2, 0.32 * n + 1.2)))
    y = np.arange(n)[::-1]
    cols = colors if colors is not None else [SERIES[0]] * n
    xerr = None
    if errors is not None:
        e = np.array(list(errors)[:max_items], dtype=float)
        xerr = e.T if e.ndim == 2 else e
    ax.barh(y, values, color=cols, height=0.68, xerr=xerr, error_kw={"ecolor": INK_2, "lw": 0.8, "capsize": 2})
    ax.set_yticks(y, [wrap(lab, label_width) for lab in labels])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    vmax = max([v for v in values if v is not None] + [1e-9])
    ends = list(values)
    if xerr is not None:      # the label goes after the upper whisker, never on it
        up = xerr[1] if np.ndim(xerr) == 2 else xerr
        ends = [v + u for v, u in zip(values, up)]
        vmax = max(ends + [vmax])
    for yi, v, e in zip(y, values, ends):
        ax.text(e + vmax * 0.01, yi, fmt.format(v), va="center", fontsize=8, color=INK_2)
    if xlim:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(0, vmax * 1.15)
    if legend:
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c, label=k) for k, c in legend.items()], loc="lower right")
    return save(fig, path, note)


def heatmap(matrix, rows, cols, title: str, path: Path, *, cmap=SEQ, fmt="{:.0f}", vmin=None, vmax=None,
            note=None, cbar_label="", row_width=40, col_width=18, annotate=True, center=None):
    m = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(cols) + 3.5), max(2.5, 0.36 * len(rows) + 1.6)))
    if center is not None:
        span = np.nanmax(np.abs(m - center)) or 1
        vmin, vmax = center - span, center + span
    im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)), [wrap(c, col_width) for c in cols], rotation=40, ha="right")
    ax.set_yticks(range(len(rows)), [wrap(r, row_width) for r in rows])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if annotate and m.size <= 900:
        lo, hi = np.nanmin(m) if vmin is None else vmin, np.nanmax(m) if vmax is None else vmax
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isnan(m[i, j]):
                    continue
                frac = (m[i, j] - lo) / ((hi - lo) or 1)
                dark = frac > 0.6 if center is None else abs(frac - 0.5) > 0.32
                ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if dark else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_2)
    if cbar_label:
        cb.set_label(cbar_label, color=INK_2, fontsize=9)
    ax.set_title(title)
    return save(fig, path, note)


def stacked_barh(labels, parts: dict, title: str, xlabel: str, path: Path, *, note=None, colors=None,
                 normalize=False, label_width=36):
    """parts: {segment name: [value per label]}; a 2px surface gap separates segments."""
    names = list(parts)
    data = np.array([parts[k] for k in names], dtype=float)
    if normalize:
        tot = data.sum(axis=0)
        tot[tot == 0] = 1
        data = data / tot
    n = len(labels)
    fig, ax = plt.subplots(figsize=(9, max(2.2, 0.36 * n + 1.4)))
    y = np.arange(n)[::-1]
    left = np.zeros(n)
    for i, k in enumerate(names):
        ax.barh(y, data[i], left=left, color=(colors or {}).get(k, color(i)), height=0.68, label=k,
                edgecolor=SURFACE, linewidth=1.5)
        left += data[i]
    ax.set_yticks(y, [wrap(lab, label_width) for lab in labels])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(ncol=min(len(names), 5), loc="upper left", bbox_to_anchor=(0, -0.08 - 0.6 / max(n, 3)))
    return save(fig, path, note)


def scatter_labeled(x, y, labels, title, xlabel, ylabel, path, *, groups=None, note=None, log=False,
                    label_top: int = 12):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    x, y = np.asarray(x, float), np.asarray(y, float)
    g = list(groups) if groups is not None else ["all"] * len(x)
    for i, name in enumerate(dict.fromkeys(g)):
        idx = [k for k, gg in enumerate(g) if gg == name]
        ax.scatter(x[idx], y[idx], s=46, color=color(i), edgecolor=SURFACE, linewidth=1.2, label=name, zorder=3)
    if log:
        ax.set_xscale("symlog", linthresh=0.005)
        ax.set_yscale("symlog", linthresh=0.005)
    lim = max(np.nanmax(x) if len(x) else 1, np.nanmax(y) if len(y) else 1) * 1.08
    ax.plot([0, lim], [0, lim], color=GRID, lw=1, zorder=1)
    order = np.argsort(-(x + y))[:label_top]
    for k in order:
        ax.annotate(wrap(labels[k], 28), (x[k], y[k]), xytext=(4, 3), textcoords="offset points", fontsize=7,
                    color=INK_2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if groups is not None:
        ax.legend(loc="upper left")
    return save(fig, path, note)


def taxonomy_tree(blocks: list, title: str, path: Path, ncol: int = 3, note: str | None = None,
                  last_apart: bool = False) -> Path:
    """A taxonomy as boxes: one box per category, its subcategories in bold, their behaviours below.

    blocks: [(category label, [(subcategory label, [behaviour, ...]), ...]), ...]. A behaviour is a string, or a
    dict {"label", "count", "marker"}: ``marker`` (e.g. ●●○, which models proposed it) is drawn first, then a small
    bar proportional to ``count`` (the items classified under the behaviour; scaled to the largest count of the
    figure) and the count itself, then the label. Boxes are packed into ``ncol`` columns."""
    cat_w, sub_w, beh_w = 48, 56, 66
    counts = [b["count"] for _, subs in blocks for _, bs in subs for b in bs if isinstance(b, dict) and b.get("count") is not None]
    cmax = max(counts + [1])
    measured = []
    for cat, subs in blocks:
        rows = [("cat", wrap(cat, cat_w), None)]
        for sub, behs in subs:
            rows.append(("sub", wrap(sub, sub_w), None))
            for b in behs:
                if isinstance(b, dict):
                    rows.append(("beh", wrap(b["label"], beh_w - (10 if b.get("marker") else 4)), b))
                else:
                    rows.append(("beh", wrap(b, beh_w), None))
        h = 0.6 + sum({"cat": 1.2, "sub": 0.95, "beh": 0.82}[k] * (t.count("\n") + 1) for k, t, _ in rows) + 0.3
        measured.append((rows, h))
    cols, colh = [[] for _ in range(ncol)], [0.0] * ncol
    n_main = len(blocks) - 1 if last_apart else len(blocks)      # last_apart: the last block is drawn after the others
    order = sorted(range(n_main), key=lambda i: -measured[i][1]) + list(range(n_main, len(blocks)))
    for i in order:
        k = int(np.argmin(colh))
        cols[k].append(i)
        colh[k] += measured[i][1] + 0.6
    H = max(colh) or 1
    fig = plt.figure(figsize=(17, max(5, H * 0.2)))
    ax = fig.add_axes([0, 0, 1, 0.97])
    ax.set_xlim(0, ncol)
    ax.set_ylim(H, 0)
    ax.axis("off")
    bar_w = 0.075                                   # width of the largest count's bar (in column units)
    for k, col in enumerate(cols):
        y = 0.3
        for i in col:
            rows, h = measured[i]
            ax.add_patch(plt.Rectangle((k + 0.02, y), 0.96, h, facecolor="none", edgecolor=color(i % 8), lw=1.6))
            yy = y + 0.5
            for kind, text, item in rows:
                n = text.count("\n") + 1
                step = {"cat": 1.2, "sub": 0.95, "beh": 0.82}[kind]
                x = {"cat": 0.04, "sub": 0.06, "beh": 0.09}[kind]
                style = {"cat": dict(fontsize=9.5, weight="bold"), "sub": dict(fontsize=8, weight="bold", color=INK_2),
                         "beh": dict(fontsize=7)}[kind]
                ymid = yy + step * (n - 1) / 2
                if item is not None:
                    xx = k + x
                    yline = yy                     # the indicator sits on the first line of the label
                    if item.get("marker"):
                        ax.text(xx, yline, item["marker"], va="center", fontsize=7, color=INK)
                        xx += 0.055
                    c = item.get("count")
                    if c is not None:
                        w = bar_w * c / cmax
                        ax.add_patch(plt.Rectangle((xx, yline - 0.22), max(w, 0.004), 0.44, facecolor=SERIES[0],
                                                   edgecolor="none"))
                        ax.add_patch(plt.Rectangle((xx, yline - 0.22), bar_w, 0.44, facecolor="none", edgecolor=GRID, lw=0.6))
                        ax.text(xx + bar_w + 0.006, yline, str(c), va="center", fontsize=6.5, color=INK_2)
                        xx += bar_w + 0.04
                    ax.text(xx, ymid, text, va="center", linespacing=1.15, **style)
                else:
                    ax.text(k + x, ymid, text, va="center", linespacing=1.15, **style)
                yy += step * n
            y += h + 0.6
    fig.suptitle(title, fontsize=12, x=0.01, ha="left", weight="bold")
    return save(fig, path, note)
