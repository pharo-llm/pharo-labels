"""Phase 1 — Repository selection: which Pharo projects the study is built on.

Input: the Pharo repositories of the corpus snapshot (Hugging Face ``pharo-llm/pharo-labels``, downloaded by setup):
the 1,585 repositories that passed mining filter 1 of the earlier labelling study (real Pharo code: Tonel/FileTree
sources, not coursework, some history). They are listed in ``repos.txt``.

Each repository gets a **quality score** in [0, 1]: the weighted sum of criterion scores, each in [0, 1], after
the *reaper* framework for identifying engineered software projects (Munaiah, Kroh, Cabrey & Nagappan,
"Curating GitHub for engineered software projects", EMSE 2017), which scores repositories on dimensions with
weights and keeps those above a threshold. Criteria (weights and saturation points in ``config.yaml``):

| criterion | measure | reason |
|---|---|---|
| community | contributors | most GitHub repositories are personal (Kalliamvakou et al., MSR 2014); reaper "community" |
| history | commits | most repositories have very few commits (Kalliamvakou et al. 2014); reaper "history" |
| unit_testing | share of test methods | reaper "unit testing" (test-to-code ratio) |
| maintenance | development span × recency of the last push | most repositories are inactive (Kalliamvakou et al. 2014) |
| documentation | share of classes with a class comment | reaper "documentation" (comment ratio) |
| size | non-test methods | reaper "size"; toy projects exhibit few behaviours |
| originality | 1 − share of non-trivial methods copied from an older repository | code duplication on GitHub (Lopes et al., DéjàVu, OOPSLA 2017) |
| license | a license is declared | reaper "license"; redistribution of the derived benchmark |
| issues | open issues (issue tracker in use) | reaper "issues" |
| loadability | a Metacello ``BaselineOf`` class (Pharo-specific) | the project can be loaded, hence tested and executed |
| popularity | stars + forks | weak weight on purpose: stars measure popularity, not quality (Borges & Valente, JSS 2018) |

Each raw value becomes a score with a saturating transform (``log``: diminishing returns up to a cap; ``ratio``:
a share up to a cap), so no single huge repository dominates and a criterion cannot exceed its weight.
**Hard filters** come first and no score compensates them: GitHub forks, mirrors (most non-trivial methods copied
from an older repository), repositories without parsed methods.

The score is used ONLY to select the repositories studied: it is never used as a measure of importance or as a
sampling weight (brief §6: popularity is not importance).

Outputs
  repos.txt                  every repository of the snapshot (one per line)
  repository_scores.csv      raw measures, criterion scores, weighted score, hard filters, decision and reason
  selected_repos.txt         the repositories kept (input of phase 3)
  scoring_scheme.md          the criteria, weights, transforms and references actually used
  repo_metrics.parquet       per-repository measures from the parsed files (cached: recomputed only if a file changes)
  figures/: score_distribution.png, threshold_sensitivity.png, criteria_kept_vs_excluded.png,
            score_contributions.png, popularity_vs_score.png, decisions.png
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import math

import numpy as np
import pandas as pd

from .. import figures as F
from ..common import data_dir, write_csv, write_parquet, write_text
from .corpus import find_sources, load_metadata


# ---------------------------------------------------------------------------------------------- measures
def scan_repository(path) -> dict:
    """Measures of one repository from its parsed file: methods, tests, classes and their comments, a Metacello
    baseline, and the hashes of its non-trivial methods (for the copy detection across repositories)."""
    n = n_test = n_fail = 0
    classes, commented, baseline = 0, 0, False
    hashes, commit = [], None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("record") == "snapshot":
                commit = r.get("commit_sha")
                for c in r.get("classes") or []:
                    classes += 1
                    commented += bool((c.get("comment") or "").strip())
                    if (c.get("superclass") == "BaselineOf") or str(c.get("name", "")).startswith("BaselineOf"):
                        baseline = True
            elif r.get("record") == "method":
                n += 1
                n_test += bool(r.get("is_test"))
                n_fail += r.get("parse_status") != "ok"
                feats = r.get("features") or {}
                if int(feats.get("n_statements") or 0) > 1 and r.get("source_hash"):   # trivial methods are idioms
                    hashes.append(r["source_hash"])
    return {"methods": n, "test_methods": n_test, "parse_failures": n_fail, "classes": classes,
            "commented_classes": commented, "has_baseline": baseline, "commit_sha": commit,
            "hashes": hashes, "file_size": path.stat().st_size}


def measures(ph, parsed, names: list[str]) -> pd.DataFrame:
    """Per-repository measures, cached in repo_metrics.parquet (a repository is rescanned only if its file changed)."""
    cache_path = ph.path("repo_metrics.parquet")
    cache = pd.read_parquet(cache_path).set_index("repository") if cache_path.exists() else pd.DataFrame()
    rows = []
    todo = []
    for name in names:
        f = parsed / (name.replace("/", "__", 1) + ".jsonl.gz")
        if not f.exists():
            rows.append({"repository": name, "methods": 0, "missing_file": True})
            continue
        if name in cache.index and int(cache.loc[name, "file_size"]) == f.stat().st_size:
            rows.append({"repository": name, **cache.loc[name].to_dict()})
        else:
            todo.append((name, f))
    if todo:
        ph.log.info(f"scanning {len(todo)} parsed repositories ({len(rows)} measured before, from the cache)")
    for name, f in ph.progress(todo, "repositories scanned", every_s=30):
        m = scan_repository(f)
        rows.append({"repository": name, **m, "missing_file": False})
    df = pd.DataFrame(rows)
    df["hashes"] = df["hashes"].map(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
    write_parquet(cache_path, df)
    return df


def originality(df: pd.DataFrame, created: dict) -> pd.Series:
    """Share of a repository's non-trivial methods NOT already present in an older repository (oldest = original)."""
    order = sorted(df["repository"], key=lambda r: (created.get(r, "9999"), r))
    seen, out = set(), {}
    hashes = dict(zip(df["repository"], df["hashes"]))
    for r in order:
        hs = hashes.get(r) or []
        if not hs:
            out[r] = 1.0
            continue
        copied = sum(h in seen for h in hs)
        out[r] = 1 - copied / len(hs)
        seen.update(hs)
    return df["repository"].map(out)


# ---------------------------------------------------------------------------------------------- scores
def transform(x: float, kind: str, cap: float) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    x = max(0.0, float(x))
    if kind == "log":
        return min(1.0, math.log1p(x) / math.log1p(cap))
    return min(1.0, x / cap)


def maintenance(created, pushed, ref: datetime, cfg: dict) -> float:
    """Development span (created → last push, full at ``maintenance_full_span_years``) × recency of the last push
    (full within ``recency_full_years`` of the snapshot, zero beyond ``recency_zero_years``)."""
    try:
        c = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        p = datetime.fromisoformat(str(pushed).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    span = min(1.0, max(0.0, (p - c).days / 365.25 / float(cfg["maintenance_full_span_years"])))
    age = (ref - p).days / 365.25
    lo, hi = float(cfg["recency_full_years"]), float(cfg["recency_zero_years"])
    recency = 1.0 if age <= lo else max(0.0, 1 - (age - lo) / (hi - lo))
    return span * recency


def run(ph):
    rc = ph.cfg["repositories"]
    parsed, snap = find_sources(ph)
    meta, selected, domains = load_metadata(ph, snap)
    if not selected:
        selected = {f.name[:-len(".jsonl.gz")].replace("__", "/", 1) for f in parsed.glob("*.jsonl.gz")}
        ph.warn("no filter-1 selection in the snapshot: every parsed repository is listed")
    names = sorted(selected)
    write_text(ph.path("repos.txt"), "\n".join(names) + "\n")
    ph.log.info(f"repos.txt: {len(names)} repositories of the snapshot ({ph.cfg['corpus']['snapshot_revision']})")

    df = measures(ph, parsed, names)
    g = lambda r, *ks, default=None: next((meta.get(r, {}).get(k) for k in ks if meta.get(r, {}).get(k) not in (None, "")), default)  # noqa: E731
    sel = lambda r, k: (meta.get(r, {}).get("selection") or {}).get(k)                                                                   # noqa: E731
    df["contributors"] = df["repository"].map(lambda r: sel(r, "contributors"))
    df["commits"] = df["repository"].map(lambda r: sel(r, "commit_count"))
    df["stars"] = df["repository"].map(lambda r: g(r, "stargazers_count", default=0))
    df["forks"] = df["repository"].map(lambda r: g(r, "forks_count", default=0))
    df["open_issues"] = df["repository"].map(lambda r: g(r, "open_issues_count", default=0))
    df["created_at"] = df["repository"].map(lambda r: g(r, "created_at"))
    df["pushed_at"] = df["repository"].map(lambda r: g(r, "pushed_at"))
    df["is_fork"] = df["repository"].map(lambda r: bool(g(r, "fork", default=False)))
    df["license"] = df["repository"].map(lambda r: (g(r, "license") or {}).get("spdx_id") if isinstance(g(r, "license"), dict) else g(r, "license"))
    df["domain"] = df["repository"].map(lambda r: domains.get(r, "unknown"))
    df["has_license"] = df["license"].map(lambda x: 1.0 if x and str(x).upper() not in ("NOASSERTION", "NONE") else 0.0)
    df["test_method_share"] = (df["test_methods"] / df["methods"].clip(lower=1)).fillna(0)
    df["non_test_methods"] = (df["methods"] - df["test_methods"]).clip(lower=0)
    df["class_comment_share"] = (df["commented_classes"] / df["classes"].clip(lower=1)).fillna(0)
    df["stars_and_forks"] = df["stars"].fillna(0) + df["forks"].fillna(0)
    df["has_baseline"] = df["has_baseline"].fillna(False).astype(float)
    created = dict(zip(df["repository"], df["created_at"].astype(str)))
    df["originality"] = originality(df, created)
    df["duplicated_share"] = 1 - df["originality"]
    # the snapshot's date is the reference for recency (not today: results must not drift with the calendar)
    ref = pd.to_datetime(df["pushed_at"], utc=True, errors="coerce").max()
    ref = ref.to_pydatetime() if pd.notna(ref) else datetime.now(timezone.utc)
    df["maintenance"] = [maintenance(c, p, ref, rc) for c, p in zip(df["created_at"], df["pushed_at"])]

    # criterion scores and the weighted score
    crit = rc["criteria"]
    wsum = sum(float(c["weight"]) for c in crit.values())
    if abs(wsum - 1) > 1e-6:
        ph.warn(f"criterion weights sum to {wsum:.3f}, not 1: they are normalised")
    for name, c in crit.items():
        df[f"s_{name}"] = [transform(x, c["transform"], float(c["cap"])) for x in df[c["metric"]]]
    df["score"] = sum(df[f"s_{n}"] * float(c["weight"]) / wsum for n, c in crit.items()).round(4)

    # hard filters, then the threshold
    hf = rc["hard_filters"]
    reasons = []
    for r in df.itertuples():
        why = []
        if r.missing_file or r.methods < int(hf.get("min_methods", 1)):
            why.append("no parsed Pharo methods")
        if hf.get("exclude_forks", True) and r.is_fork:
            why.append("GitHub fork")
        if r.duplicated_share >= float(hf.get("max_duplicated_share", 0.5)):
            why.append(f"mirror: {r.duplicated_share:.0%} of its non-trivial methods are in an older repository")
        reasons.append(why)
    df["hard_filter"] = ["; ".join(w) for w in reasons]
    thr = float(rc["threshold"])
    df["decision"] = np.where(df["hard_filter"] != "", "excluded (hard filter)",
                              np.where(df["score"] >= thr, "selected", "excluded (score below threshold)"))
    df["reason"] = np.where(df["hard_filter"] != "", df["hard_filter"],
                            np.where(df["score"] >= thr, f"score ≥ {thr}", "score " + df["score"].astype(str) + f" < {thr}"))
    out_cols = (["repository", "decision", "reason", "score"] + [f"s_{n}" for n in crit] +
                ["contributors", "commits", "test_method_share", "maintenance", "class_comment_share", "non_test_methods",
                 "duplicated_share", "license", "open_issues", "has_baseline", "stars", "forks", "methods", "test_methods",
                 "classes", "created_at", "pushed_at", "is_fork", "domain", "commit_sha"])
    df = df.sort_values(["decision", "score"], ascending=[False, False])
    write_csv(ph.path("repository_scores.csv"), df[out_cols])
    kept = df[df["decision"] == "selected"]
    write_text(ph.path("selected_repos.txt"), "\n".join(sorted(kept["repository"])) + "\n")
    write_text(ph.path("scoring_scheme.md"), scheme_md(rc, wsum, thr, df))

    ph.metric("repositories", len(df))
    ph.metric("selected", len(kept))
    ph.metric("excluded_hard_filter", int((df["decision"] == "excluded (hard filter)").sum()))
    ph.metric("excluded_score", int((df["decision"] == "excluded (score below threshold)").sum()))
    ph.metric("threshold", thr)
    ph.metric("selected_methods", int(kept["methods"].sum()))
    ph.metric("median_score_selected", round(float(kept["score"].median()), 3) if len(kept) else None)
    ph.log.info(f"selected {len(kept)} of {len(df)} repositories ({int(kept['methods'].sum()):,} methods); "
                f"hard filters exclude {int((df['decision'] == 'excluded (hard filter)').sum())}, the score "
                f"{int((df['decision'] == 'excluded (score below threshold)').sum())}")
    figures(ph, df, crit, thr)


def scheme_md(rc, wsum, thr, df) -> str:
    lines = ["# Repository scoring scheme", "",
             "Weighted sum of criterion scores in [0, 1] (reaper framework: Munaiah et al., EMSE 2017).", "",
             "| criterion | measure | weight | transform | cap | median score (selected) |", "|---|---|---|---|---|---|"]
    kept = df[df["decision"] == "selected"]
    for n, c in rc["criteria"].items():
        med = kept[f"s_{n}"].median() if len(kept) else float("nan")
        lines.append(f"| {n} | {c['metric']} | {float(c['weight']) / wsum:.2f} | {c['transform']} | {c['cap']} | {med:.2f} |")
    hf = rc["hard_filters"]
    lines += ["", f"Hard filters: forks excluded = {hf.get('exclude_forks')}, mirrors = duplicated share ≥ "
              f"{hf.get('max_duplicated_share')}, at least {hf.get('min_methods')} parsed method(s).", "",
              f"Threshold: score ≥ {thr}. The score selects repositories only; it is never an importance or sampling weight.",
              "", "References: Kalliamvakou et al., The promises and perils of mining GitHub, MSR 2014; Munaiah et al., "
              "Curating GitHub for engineered software projects, EMSE 2017; Lopes et al., DéjàVu: a map of code duplicates "
              "on GitHub, OOPSLA 2017; Borges & Valente, What's in a GitHub star?, JSS 2018.", ""]
    return "\n".join(lines)


def figures(ph, df, crit, thr):
    eligible = df[df["hard_filter"] == ""]
    fig, ax = F.plt.subplots(figsize=(8.5, 4))
    ax.hist(eligible["score"], bins=np.linspace(0, 1, 41), color=F.SERIES[0], rwidth=0.9)
    ax.axvline(thr, color=F.STATUS["critical"], lw=1.5)
    ax.text(thr, ax.get_ylim()[1] * 0.92, f"  threshold {thr}", color=F.INK_2, fontsize=8)
    ax.set_xlabel("weighted quality score")
    ax.set_ylabel("repositories")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Repository quality scores ({len(eligible)} repositories after the hard filters)")
    F.save(fig, ph.figure("score_distribution.png"))

    ts = np.round(np.arange(0.1, 0.81, 0.05), 2)
    kept_n = [int((eligible["score"] >= t).sum()) for t in ts]
    kept_m = [int(eligible.loc[eligible["score"] >= t, "methods"].sum()) for t in ts]
    fig, axes = F.plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, vals, lab in ((axes[0], kept_n, "repositories kept"), (axes[1], [m / 1e3 for m in kept_m], "methods kept (thousands)")):
        ax.plot(ts, vals, color=F.SERIES[0], lw=2, marker="o", ms=4)
        ax.axvline(thr, color=F.STATUS["critical"], lw=1.2)
        ax.set_xlabel("threshold")
        ax.set_ylabel(lab)
    fig.suptitle("What other thresholds would keep", x=0.01, ha="left", fontsize=12, weight="bold")
    F.save(fig, ph.figure("threshold_sensitivity.png"))

    names = list(crit)
    sel = eligible[eligible["score"] >= thr]
    exc = eligible[eligible["score"] < thr]
    fig, ax = F.plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(names))
    for i, (lab, d) in enumerate((("selected", sel), ("excluded by the score", exc))):
        ax.bar(x + (i - 0.5) * 0.38, [d[f"s_{n}"].mean() if len(d) else 0 for n in names], width=0.36, color=F.color(i), label=lab)
    ax.set_xticks(x, [n.replace("_", " ") for n in names], rotation=30, ha="right")
    ax.set_ylabel("mean criterion score")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend()
    ax.set_title("Criterion scores of selected vs excluded repositories")
    F.save(fig, ph.figure("criteria_kept_vs_excluded.png"))

    wsum = sum(float(c["weight"]) for c in crit.values())
    contrib = {n: (sel[f"s_{n}"] * float(c["weight"]) / wsum).mean() if len(sel) else 0 for n, c in crit.items()}
    order = sorted(contrib, key=lambda n: -contrib[n])
    F.barh([n.replace("_", " ") for n in order], [contrib[n] for n in order],
           "What makes the score of selected repositories (mean weighted contribution)", "points of score",
           ph.figure("score_contributions.png"), fmt="{:.3f}",
           note="popularity has a deliberately low weight: stars measure popularity, not quality (Borges & Valente 2018)")

    fig, ax = F.plt.subplots(figsize=(8, 5.5))
    for i, (lab, d) in enumerate((("selected", sel), ("excluded by the score", exc))):
        ax.scatter(d["stars"].fillna(0) + 1, d["score"], s=14, color=F.color(i), alpha=0.7, label=lab, edgecolor="none")
    ax.set_xscale("log")
    ax.axhline(thr, color=F.STATUS["critical"], lw=1)
    ax.set_xlabel("stars + 1 (log scale)")
    ax.set_ylabel("weighted quality score")
    ax.set_title("Popularity does not decide the selection")
    ax.legend()
    F.save(fig, ph.figure("popularity_vs_score.png"))

    dec = df["decision"].value_counts()
    hard = Counter(x.split(":")[0] for r in df["hard_filter"] if r for x in r.split("; "))
    labels = ["selected", "excluded by the score"] + [f"excluded: {k}" for k in hard]
    vals = [int(dec.get("selected", 0)), int(dec.get("excluded (score below threshold)", 0))] + list(hard.values())
    F.barh(labels, vals, f"Repository selection ({len(df)} repositories of the snapshot)", "repositories",
           ph.figure("decisions.png"), colors=[F.STATUS["good"], F.STATUS["warning"]] + [F.STATUS["critical"]] * len(hard),
           note="a repository can fail several hard filters (each is counted)")
