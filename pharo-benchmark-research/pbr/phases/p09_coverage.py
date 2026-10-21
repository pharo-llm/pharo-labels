"""Phase 9 — Literature versus Pharo comparison.

Both classified datasets are joined on the shared behaviour IDs.

* Literature side: the annotation split of phase 6 (literature tasks, one per problem), labelled by the
  AI majority (behaviours given by at least two of the three annotators), replaced by the human reference
  where phase 7 has one. Share = tasks with the behaviour / literature tasks; also in how many benchmarks.
* Pharo side: the design-weighted share of phase 8 (with its interval) and the Pharo items of phase 6
  (random design, AI majority) as a second, independent sample.

Status of every behaviour:
* ``both``: observed on both sides with at least ``coverage.min_support`` items each;
* ``pharo_only``: observed in Pharo, **not identified** in the selected literature sample (with the
  upper bound of its literature share that the sample still allows, rule of three);
* ``literature_only``: observed in the literature, **not observed** in the sampled Pharo corpus (upper bound);
* ``investigate``: too little support on a side, or the annotators do not agree on it (α < 0.667),
  or observed nowhere.
Non-observation is never read as absence: every gap carries its upper bound and its sample size.
Every row keeps example items (IDs and links) from each side as evidence.

Outputs: coverage_matrix.csv, behavior_gaps.csv, coverage_by_benchmark.csv, figures.
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from .. import agreement as A
from .. import figures as F
from .. import taxonomy as T
from ..common import read_json, write_csv
from .p08_profile import latest_taxonomy


def literature_labels(ph, tax) -> pd.DataFrame:
    items = pd.read_csv(ph.upstream(7, "item_agreement.csv"), converters={"majority": ast.literal_eval})
    lit = items[items["source"] == "literature"][["item_id", "group", "majority", "level"]].rename(columns={"group": "benchmark"})
    lit["labels"] = lit["majority"].map(list)
    lit["label_source"] = "AI majority (≥2 of 3)"
    ref_p = ph.upstream(7, "human_reference_labels.csv")
    if ref_p.exists():
        ref = pd.read_csv(ref_p)
        ref = ref[ref["item_id"].str.startswith("L:") & ref["reference"].notna()]
        m = dict(zip(ref["item_id"], ref["reference"].str.split(";")))
        hit = lit["item_id"].isin(m)
        lit.loc[hit, "labels"] = lit.loc[hit, "item_id"].map(m)
        lit.loc[hit, "label_source"] = "human reference"
    # one problem counted once (phase 3 already drew one task per problem)
    return lit


def pharo_annotation_labels(ph) -> pd.DataFrame:
    items = pd.read_csv(ph.upstream(7, "item_agreement.csv"), converters={"majority": ast.literal_eval})
    p = items[(items["source"] == "pharo") & (items["purpose"] == "random")]
    return p.assign(labels=p["majority"].map(list))


def run(ph):
    tax = latest_taxonomy(ph)
    v6 = ph.cfg["taxonomy"].get("version", "v1")
    if tax["version"] != v6:
        ph.warn(f"phase 8 uses taxonomy {tax['version']} while phases 6–7 used {v6}: literature labels are translated "
                f"explicitly with the version map (taxonomy.translate), never rewritten")
    tax6 = T.load(ph.upstream(5, f"taxonomy_{v6}.json"))
    for u in (6, 7, 8):
        if ph.upstream_state(u).get("status") == "provisional":
            ph.mark_provisional(f"phase {u} is provisional (expert input pending)")
            break
    min_sup = int(ph.cfg["coverage"]["min_support"])
    conf = float(ph.cfg["coverage"]["confidence"])
    lit = literature_labels(ph, tax6)
    if tax["version"] != v6:
        lit["labels"] = lit["labels"].map(lambda ids: T.translate(ids, tax6, tax))
    p6 = pharo_annotation_labels(ph)
    if tax["version"] != v6:
        p6["labels"] = p6["labels"].map(lambda ids: T.translate(ids, tax6, tax))
    fq = pd.read_csv(ph.upstream(8, "weighted_frequency.csv"))
    fq = fq[fq["level"] == "behavior"].set_index("id")
    prof = pd.read_parquet(ph.upstream(8, "pharo_behavior_profiles.parquet"))
    ag = read_json(ph.upstream(7, "agreement.json"), {})
    per_alpha = {k: v.get("alpha") for k, v in (ag.get("per_behavior") or {}).items()}
    n_lit, n_p6, n_prof = len(lit), len(p6), len(prof)
    benchmarks = sorted(lit["benchmark"].unique())

    rows, bench_rows = [], []
    for b in T.active(tax, "behavior"):
        bid = b["id"]
        lit_has = lit[lit["labels"].map(lambda ids: bid in ids)]
        p6_has = p6[p6["labels"].map(lambda ids: bid in ids)]
        pr_has = prof[prof["behavior_ids"].map(lambda ids: bid in ids)]
        f = fq.loc[bid] if bid in fq.index else None
        lit_n, p_n = len(lit_has), len(pr_has) + len(p6_has)
        alpha = per_alpha.get(bid)
        if lit_n >= min_sup and p_n >= min_sup:
            status = "both"
        elif lit_n == 0 and p_n >= min_sup:
            status = "pharo_only"
        elif p_n == 0 and lit_n >= min_sup:
            status = "literature_only"
        else:
            status = "investigate"
        reasons = []
        if 0 < lit_n < min_sup:
            reasons.append(f"only {lit_n} literature item(s)")
        if 0 < p_n < min_sup:
            reasons.append(f"only {p_n} Pharo item(s)")
        if lit_n == 0 and p_n == 0:
            reasons.append("observed in neither sample")
        support_status = status                   # from the counts alone, before the agreement check
        if alpha is not None and alpha < 0.667:
            reasons.append(f"annotators disagree on it (α={alpha})")
            if status != "investigate":
                status = "investigate"
        rows.append({
            "behavior_id": bid, "behavior": b["name"], "subcategory_id": b["subcategory_id"], "category_id": b["category_id"],
            "category": T.label(tax, b["category_id"]), "status": status, "support_status": support_status,
            "reasons": "; ".join(reasons),
            "literature_tasks": lit_n, "literature_share": round(lit_n / max(n_lit, 1), 4),
            "literature_share_upper_bound": round(A.rule_of_three(n_lit, conf), 4) if lit_n == 0 else None,
            "literature_benchmarks": int(lit_has["benchmark"].nunique()),
            "literature_benchmark_names": "; ".join(sorted(lit_has["benchmark"].unique())),
            "pharo_profile_methods": len(pr_has), "pharo_weighted_share": None if f is None else f["weighted_share"],
            "pharo_ci_low": None if f is None else f["ci_low"], "pharo_ci_high": None if f is None else f["ci_high"],
            "pharo_share_upper_bound": round(A.rule_of_three(n_prof + n_p6, conf), 4) if p_n == 0 else None,
            "pharo_annotation_methods": len(p6_has), "pharo_repositories": int(pd.concat([pr_has["repository"]]).nunique()),
            "agreement_alpha": alpha,
            "literature_examples": "; ".join(lit_has["item_id"].head(3)),
            "pharo_examples": "; ".join(pr_has["provenance"].head(3)) if "provenance" in pr_has else "",
        })
        for bm in benchmarks:
            bench_rows.append({"behavior_id": bid, "benchmark": bm,
                               "tasks": int((lit_has["benchmark"] == bm).sum()),
                               "sampled": int((lit["benchmark"] == bm).sum())})
    cov = pd.DataFrame(rows)
    write_csv(ph.path("coverage_matrix.csv"), cov)
    gaps = cov[cov["status"] != "both"].copy()
    gaps["interpretation"] = gaps.apply(lambda r: (
        f"found in Pharo ({r.pharo_profile_methods + r.pharo_annotation_methods} methods), not identified in the {n_lit} sampled "
        f"literature tasks (literature share ≤ {r.literature_share_upper_bound:.1%} at {conf:.0%})" if r.status == "pharo_only" else
        f"found in {r.literature_tasks} literature tasks, not observed in the {n_prof + n_p6} sampled Pharo methods "
        f"(Pharo share ≤ {r.pharo_share_upper_bound:.1%} at {conf:.0%})" if r.status == "literature_only" else
        f"needs investigation: {r.reasons}"), axis=1)
    write_csv(ph.path("behavior_gaps.csv"), gaps)
    bdf = pd.DataFrame(bench_rows)
    write_csv(ph.path("coverage_by_benchmark.csv"), bdf)

    st = cov["status"].value_counts()
    for s in ("both", "pharo_only", "literature_only", "investigate"):
        ph.metric(s, int(st.get(s, 0)))
    ph.metric("literature_tasks", n_lit)
    ph.metric("pharo_methods", n_prof + n_p6)
    ph.metric("literature_labels_from_humans", int((lit["label_source"] == "human reference").sum()))
    figures(ph, cov, bdf, tax, n_lit, n_prof)


def figures(ph, cov, bdf, tax, n_lit, n_prof):
    status_col = {"both": F.SERIES[0], "pharo_only": F.SERIES[1], "literature_only": F.SERIES[2], "investigate": F.OTHER}
    obs = cov[(cov["literature_tasks"] > 0) | (cov["pharo_profile_methods"] > 0) | (cov["pharo_annotation_methods"] > 0)]
    if len(obs):
        fig, ax = F.plt.subplots(figsize=(9, 7))
        x = obs["literature_share"] * 100
        y = obs["pharo_weighted_share"].fillna(0) * 100
        for s, c in status_col.items():
            sub = obs["status"] == s
            ax.scatter(x[sub], y[sub], s=48, color=c, edgecolor=F.SURFACE, lw=1.2, label=s, zorder=3)
        lim = max(x.max(), y.max(), 1) * 1.1
        ax.plot([0, lim], [0, lim], color=F.GRID, lw=1, zorder=1)
        top = (x + y).sort_values(ascending=False).index[:14]
        for i in top:
            ax.annotate(F.wrap(f"{obs.loc[i, 'behavior_id']} {obs.loc[i, 'behavior']}", 26), (x[i], y[i]), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color=F.INK_2)
        ax.set_xlabel(f"% of literature tasks (n={n_lit})")
        ax.set_ylabel(f"% of Pharo methods, design-weighted (n={n_prof})")
        ax.set_title("Behaviour coverage: literature vs Pharo (diagonal = same share)")
        ax.legend(loc="upper right")
        F.save(fig, ph.figure("coverage_scatter.png"), note="a point on an axis = not observed on the other side in this sample, not proven absent")
    cats = cov.groupby(["category", "status"]).size().unstack(fill_value=0)
    cats = cats.loc[cats.sum(axis=1).sort_values(ascending=False).index]
    F.stacked_barh(list(cats.index), {s: cats[s].tolist() for s in status_col if s in cats},
                   "Coverage status of behaviours, by category", "behaviours", ph.figure("coverage_by_category.png"),
                   colors=status_col)
    if len(bdf):
        b2c = bdf.merge(cov[["behavior_id", "category"]], on="behavior_id")
        piv = b2c.groupby(["category", "benchmark"])["tasks"].sum().unstack(fill_value=0)
        sampled = bdf.drop_duplicates("benchmark").set_index("benchmark")["sampled"]
        piv = (piv / sampled.reindex(piv.columns).clip(lower=1).values * 100)
        piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
        F.heatmap(piv.values, list(piv.index), list(piv.columns), "What each benchmark exercises (% of its sampled tasks)",
                  ph.figure("benchmark_by_category.png"), fmt="{:.0f}", cbar_label="% of tasks")
    g = cov[cov["status"].isin(["pharo_only"])].sort_values("pharo_weighted_share", ascending=False).head(20)
    if len(g):
        F.barh([f"{r.behavior_id} {r.behavior}" for r in g.itertuples()], g["pharo_weighted_share"].fillna(0) * 100,
               "Pharo behaviours not identified in the sampled literature", "% of Pharo methods (weighted)",
               ph.figure("gaps_pharo_only.png"), fmt="{:.1f}%", colors=[F.SERIES[1]] * len(g),
               note="candidates for the Pharo benchmark; 'not identified' ≠ absent from all literature")
