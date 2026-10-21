"""Phase 9 — Large-scale Pharo behaviour analysis.

The ``profile`` split (random design of phase 4, disjoint from the taxonomy and annotation data) is
classified with the latest taxonomy (v2 when the experts corrected v1, else v1) by the profile model(s):
``profile.models: auto`` picks the annotator closest to the experts (micro F1, phase 8), or — without
expert labels — the one agreeing most with the other two (mean pairwise κ); a list of names uses those
models and their majority.

Estimates (sampling design considered, phase 4):
* per behaviour / subcategory / category: methods in the sample, unweighted share, **design-weighted
  share** with a 95% interval (ratio estimator; linearisation variance, repositories = PSUs within groups);
* **repository prevalence**: share of sampled repositories in which the behaviour was observed (a lower
  bound: only a few methods per repository are sampled);
* **differences across repository groups** (weighted shares per group; χ² test of independence on counts,
  an approximation that ignores the clustering);
* **rare** (weighted share < ``profile.rare_share``) and **important** (expert relevance ≥ 4, phase 8) behaviours;
* **unknown and uncertain** classifications (UNKNOWN, INSUFFICIENT, high uncertainty, low confidence) and
  the missing behaviours the models described;
* **PPI correction** when expert labels exist (phase 8): category shares corrected for the model's errors
  (Angelopoulos et al. 2023; the expert items are weighted by their stratum weight).

Outputs: pharo_behavior_profiles.parquet, weighted_frequency.csv, repository_prevalence.csv,
group_differences.csv, unknowns.csv, figures.
"""
from __future__ import annotations

from collections import Counter
import re

import numpy as np
import pandas as pd

from .. import agreement as A
from .. import figures as F
from .. import taxonomy as T
from ..common import read_json, read_jsonl, write_csv, write_parquet
from ..items import load_items
from ..llm import Client, ledger_summary
from .annotation import classify, flatten, latest_raw


def latest_taxonomy(ph) -> dict:
    d7 = ph.upstream("validation")
    versions = sorted(d7.glob("taxonomy_v*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if versions:
        return T.load(versions[-1])
    return T.load(ph.upstream("taxonomy", f"taxonomy_{ph.cfg['taxonomy'].get('version', 'v1')}.json"))


def choose_models(ph) -> tuple[list[str], str]:
    want = ph.cfg["profile"].get("models", "auto")
    names = [a["name"] for a in ph.cfg["llm"]["annotators"]]
    if isinstance(want, list):
        return want, "configured"
    hv = read_json(ph.upstream("validation", "human_vs_ai.json"), {}) or {}
    scores = {m: (hv.get(m) or {}).get("micro", {}).get("f1") for m in names}
    if any(v is not None for v in scores.values()):
        best = max(names, key=lambda m: scores[m] or -1)
        return [best], f"highest micro F1 against the expert reference ({scores})"
    ag = read_json(ph.upstream("validation", "agreement.json"), {})
    pw = ag.get("overall", {}).get("pairwise_cohen_primary_behavior", {})
    mean = {m: float(np.mean([v for k, v in pw.items() if m in k.split("~") and v is not None] or [0])) for m in names}
    best = max(names, key=lambda m: mean[m])
    return [best], f"no expert labels yet: highest mean pairwise κ with the other annotators ({ {k: round(v, 3) for k, v in mean.items()} })"


def run(ph):
    tax = latest_taxonomy(ph)
    models, why = choose_models(ph)
    ph.log.info(f"taxonomy {tax['version']} ({tax['status']}); profile model(s) {models}: {why}")
    for u in ("taxonomy", "annotation", "validation"):
        if ph.upstream_state(u).get("status") == "provisional":
            ph.mark_provisional(f"phase {u} is provisional (it waits for expert input)")
            break
    items = load_items(ph, "profile", include_targeted=False)
    items = items[(items["source"] == "pharo") & (items["subset"] == "production")]   # frequencies: production code
    raw_path = ph.path("annotations_raw.jsonl")
    specs = {a["name"]: a for a in ph.cfg["llm"]["annotators"]}
    for m in models:
        classify(ph, Client(specs[m], ph.cfg), tax, items, raw_path, ph.key)
    raw = [r for r in latest_raw(read_jsonl(raw_path), tax) if r["model"] in models and r["item_id"] in set(items["item_id"])]
    ann = flatten(raw)
    if len(models) > 1:      # majority of the profile models
        rows = []
        for item_id, g in ann.groupby("item_id"):
            maj = sorted(A.majority_set([set(x) for x in g["behavior_ids"]]))
            rows.append({"item_id": item_id, "behavior_ids": maj or [T.UNKNOWN], "model": "+".join(models)})
        lab = pd.DataFrame(rows)
    else:
        lab = ann[["item_id", "behavior_ids", "model", "uncertainty", "mean_confidence", "missing_behavior"]]
    prof = items[["item_id", "repository", "group", "weight", "title", "provenance"]].merge(lab, on="item_id", how="inner")
    missing = len(items) - len(prof)
    if missing:
        ph.warn(f"{missing} profile methods have no valid label (left out of the estimates)")
    prof["category_ids"] = prof["behavior_ids"].map(lambda ids: list(dict.fromkeys(T.behavior_path(tax, b)[0] for b in ids)))
    prof["subcategory_ids"] = prof["behavior_ids"].map(lambda ids: list(dict.fromkeys(T.behavior_path(tax, b)[1] for b in ids)))
    prof["taxonomy_version"] = tax["version"]
    prof["method_id"] = prof["item_id"].str[2:]
    write_parquet(ph.path("pharo_behavior_profiles.parquet"), prof)

    rel = {}
    rp = ph.upstream("validation", "behavior_relevance.csv")
    if rp.exists():
        r = pd.read_csv(rp)
        rel = dict(zip(r["behavior_id"], r["relevance_pharo"]))
    conf = float(ph.cfg["coverage"].get("confidence", 0.95))
    rare = float(ph.cfg["profile"]["rare_share"])
    freq = []
    levels = [("category", "category_ids", T.active(tax, "category")),
              ("subcategory", "subcategory_ids", T.active(tax, "subcategory")),
              ("behavior", "behavior_ids", T.active(tax, "behavior"))]
    for lv, col, concepts in levels:
        for c in concepts + [{"id": T.UNKNOWN, "name": "Unknown"}, {"id": T.INSUFFICIENT, "name": "Insufficient context"}]:
            y = prof[col].map(lambda ids, cid=c["id"]: 1.0 if cid in ids else 0.0)
            est = A.weighted_share(y, prof["weight"], prof["repository"], prof["group"], conf)
            repos_with = prof.loc[y > 0, "repository"].nunique()
            freq.append({"level": lv, "id": c["id"], "name": c["name"], "parent": c.get("category_id") or "",
                         "methods": int(y.sum()), "unweighted_share": round(float(y.mean()), 5) if len(y) else None,
                         "weighted_share": est["share"], "ci_low": est["ci_low"], "ci_high": est["ci_high"],
                         "repositories_with": int(repos_with), "relevance_pharo": rel.get(c["id"]),
                         "upper_bound_if_zero": round(A.rule_of_three(len(y), conf), 4) if y.sum() == 0 else None})
    fq = pd.DataFrame(freq)
    fq["rare"] = (fq["weighted_share"].fillna(0) < rare) & (fq["methods"] > 0)
    fq["rare_but_important"] = fq["rare"] & (fq["relevance_pharo"].fillna(0) >= 4)
    fq["not_observed"] = fq["methods"] == 0

    # PPI: category shares corrected with the expert reference (phase 8), Pharo items only
    ref_p = ph.upstream("validation", "human_reference_labels.csv")
    if ref_p.exists() and len(models) == 1:
        ref = pd.read_csv(ref_p)
        ref = ref[ref["item_id"].str.startswith("P:") & ref["reference"].notna()]
        a6 = pd.read_parquet(ph.upstream("annotation", "annotations_validated.parquet"))
        a6 = a6[(a6["model"] == models[0]) & (a6["taxonomy_version"] == tax["version"])]
        if len(ref) and len(a6):
            j = ref.merge(a6[["item_id", "category_ids"]], on="item_id")
            j["ref_cats"] = j["reference"].map(lambda s: {T.behavior_path(tax, b)[0] for b in str(s).split(";")})
            for i, row in fq[fq["level"] == "category"].iterrows():
                cid = row["id"]
                fu = prof["category_ids"].map(lambda ids: 1.0 if cid in ids else 0.0)
                fl = j["category_ids"].map(lambda ids: 1.0 if cid in ids else 0.0)
                yl = j["ref_cats"].map(lambda s: 1.0 if cid in s else 0.0)
                p = A.ppi_mean(fu, fl, yl, conf)
                fq.loc[i, "ppi_share"], fq.loc[i, "ppi_ci_low"], fq.loc[i, "ppi_ci_high"] = p.get("share"), p.get("ci_low"), p.get("ci_high")
            ph.log.info(f"PPI correction with {len(j)} expert-labelled Pharo items (unweighted rectifier; approximate "
                        "because expert items were stratified by AI agreement)")
    write_csv(ph.path("weighted_frequency.csv"), fq)

    repos = prof.groupby("repository").agg(group=("group", "first"), methods_sampled=("item_id", "size"))
    rp_rows = []
    for lv, col, concepts in levels:
        for c in concepts:
            has = prof[prof[col].map(lambda ids, cid=c["id"]: cid in ids)].groupby("repository").size()
            n = int((has > 0).sum())
            lo, hi = A.wilson(n, len(repos), conf)
            rp_rows.append({"level": lv, "id": c["id"], "name": c["name"], "repositories_sampled": int(len(repos)),
                            "repositories_with_behavior": n, "share_observed": round(n / max(len(repos), 1), 4),
                            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                            "note": f"lower bound: ≤ {int(repos['methods_sampled'].max())} methods sampled per repository"})
    write_csv(ph.path("repository_prevalence.csv"), pd.DataFrame(rp_rows))

    # groups
    from scipy.stats import chi2_contingency
    grows = []
    cats = T.active(tax, "category")
    for c in cats:
        y = prof["category_ids"].map(lambda ids, cid=c["id"]: cid in ids)
        tab = pd.crosstab(prof["group"], y)
        p = None
        if tab.shape[1] == 2 and tab.shape[0] > 1 and (tab.values.sum(axis=0) > 0).all():
            try:
                p = float(chi2_contingency(tab.values)[1])
            except ValueError:
                p = None
        for g, sub in prof.groupby("group"):
            yy = sub["category_ids"].map(lambda ids, cid=c["id"]: 1.0 if cid in ids else 0.0)
            w = sub["weight"]
            grows.append({"category_id": c["id"], "category": c["name"], "group": g, "methods": int(len(sub)),
                          "with_category": int(yy.sum()), "weighted_share": round(float((yy * w).sum() / w.sum()), 4) if w.sum() else None,
                          "chi2_p_value_category_vs_group": round(p, 4) if p is not None else None})
    gdf = pd.DataFrame(grows)
    write_csv(ph.path("group_differences.csv"), gdf)

    unk = ann[(ann["is_unknown"]) | (ann["is_insufficient"]) | (ann["uncertainty"] == "high") | (ann["mean_confidence"] < 0.5)] \
        if len(ann) else ann
    write_csv(ph.path("unknowns.csv"), unk.merge(items[["item_id", "title", "provenance"]], on="item_id", how="left") if len(unk) else unk)

    beh = fq[fq["level"] == "behavior"]
    ph.metric("methods_profiled", int(len(prof)))
    ph.metric("repositories", int(prof["repository"].nunique()))
    ph.metric("profile_models", "+".join(models))
    ph.metric("taxonomy_version", tax["version"])
    ph.metric("behaviors_observed", int((beh["methods"] > 0).sum()))
    ph.metric("behaviors_rare", int(beh["rare"].sum()))
    ph.metric("unknown_weighted_share", fq.loc[(fq["level"] == "behavior") & (fq["id"] == T.UNKNOWN), "weighted_share"].iloc[0])
    ph.metric("uncertain_items", int(len(unk)))
    ph.metric("llm", ledger_summary(ph.cfg, ph.key))
    figures(ph, fq, gdf, prof, tax)


def figures(ph, fq, gdf, prof, tax):
    n = len(prof)
    note = f"{n} methods, {prof['repository'].nunique()} repositories; design-weighted, 95% CI (repositories as clusters)"
    cat = fq[(fq["level"] == "category") & (fq["methods"] > 0)].sort_values("weighted_share", ascending=False)
    if len(cat):
        err = np.array([(cat["weighted_share"] - cat["ci_low"]).clip(lower=0), (cat["ci_high"] - cat["weighted_share"]).clip(lower=0)]).T * 100
        F.barh([f"{r.id} {r.name}" for r in cat.itertuples()], cat["weighted_share"] * 100,
               "Pharo methods per behaviour category (corpus-wide estimate)", "% of methods", ph.figure("weighted_frequency_categories.png"),
               errors=err, fmt="{:.1f}%", note=note)
    beh = fq[(fq["level"] == "behavior") & (fq["methods"] > 0)].sort_values("weighted_share", ascending=False).head(30)
    if len(beh):
        err = np.array([(beh["weighted_share"] - beh["ci_low"]).clip(lower=0), (beh["ci_high"] - beh["weighted_share"]).clip(lower=0)]).T * 100
        F.barh([f"{r.id} {r.name}" for r in beh.itertuples()], beh["weighted_share"] * 100,
               "Most frequent behaviours in Pharo code (top 30)", "% of methods", ph.figure("weighted_frequency_behaviors.png"),
               errors=err, fmt="{:.1f}%", note=note,
               colors=[F.STATUS["warning"] if r else F.SERIES[0] for r in beh["rare"]])
    if len(gdf):
        piv = gdf.pivot_table(index="category", columns="group", values="weighted_share").fillna(0) * 100
        piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
        F.heatmap(piv.values, list(piv.index), list(piv.columns), "Behaviour categories across repository groups (% of methods)",
                  ph.figure("groups_by_category.png"), fmt="{:.0f}", cbar_label="% of methods (weighted)",
                  note="small groups have few sampled methods: read with the χ² p-values of group_differences.csv")
    uu = fq[(fq["level"] == "behavior") & fq["id"].isin(T.RESERVED)]
    unc = (prof.get("uncertainty") == "high").mean() if "uncertainty" in prof else 0
    lowc = (prof.get("mean_confidence", pd.Series(dtype=float)) < 0.5).mean() if "mean_confidence" in prof else 0
    vals = {"UNKNOWN (no behaviour fits)": float(uu.loc[uu["id"] == T.UNKNOWN, "weighted_share"].fillna(0).sum()) * 100,
            "INSUFFICIENT context": float(uu.loc[uu["id"] == T.INSUFFICIENT, "weighted_share"].fillna(0).sum()) * 100,
            "high uncertainty": float(unc or 0) * 100, "confidence < 0.5": float(lowc or 0) * 100}
    F.barh(list(vals), list(vals.values()), "Unknown and uncertain classifications", "% of profiled methods",
           ph.figure("unknown_uncertain.png"), fmt="{:.1f}%", colors=[F.STATUS["neutral"]] * 4)
