"""Phase 4 — Representative sampling.

Two kinds of samples, never mixed in an estimate:

* **random** (probability) samples, for estimating how frequent a behaviour is. Stratified two-stage
  cluster design: the strata are repository groups (application domain, small domains pooled); stage 1
  draws repositories at random inside each group, stage 2 draws at most ``methods_per_repo`` methods at
  random inside each drawn repository. A large repository therefore cannot dominate the sample, and every
  method carries its inclusion probability and design weight
  ``w = (N_h / n_h) × (M_i / m_i)`` (N_h eligible repositories of group h, n_h drawn; M_i eligible methods
  of repository i, m_i drawn), with which phase 9 estimates corpus-wide shares (Cochran 1977, ch. 10–11).
* **targeted** samples, for discovering rare behaviours: methods that send selectors or reference globals
  of rare areas (``sampling.targeted_signals``). They have no weight and are excluded from frequencies.

Three disjoint splits are drawn with the same design: ``discovery`` (phase 5: build the taxonomy),
``annotation`` (phase 7–8: measure agreement on data the taxonomy was not built from), ``profile`` (phase 9).
Literature tasks get two disjoint splits too, stratified by benchmark, one problem counted once.

Outputs
  sampling_design.json   frame, eligibility rules and counts, groups, allocation, formulas, seeds
  pilot_sample.csv       every sampled Pharo method: split, purpose, group, probabilities, weight, provenance
  literature_sample.csv  sampled literature tasks per split
  figures/: groups_corpus_vs_sample.png, methods_per_sampled_repository.png, targeted_signals.png
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
import pandas as pd

from .. import figures as F
from ..common import write_csv, write_json

SPLITS = [("discovery", "pharo_discovery", "pharo_discovery_targeted"),
          ("annotation", "pharo_annotation", "pharo_annotation_targeted"),
          ("profile", "pharo_profile", None)]
COLS = ["method_id", "repository", "class_name", "side", "selector", "is_test", "parse_status", "over_long",
        "duplicate_of", "is_trivial", "sent_selectors", "referenced_globals", "n_statements", "url"]


def frame(ph, subset: str = "production") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Eligible methods and repositories of one subset — ``production`` (decision keep, + review when
    method_filters.include_review) or ``test`` (decision keep_test: kept separately) — after the method filters of
    phase 3, with the count of every decision."""
    s = ph.cfg["sampling"]
    repos = pd.read_parquet(ph.upstream("corpus", "repositories.parquet"))
    m = pd.read_parquet(ph.upstream("corpus", "methods.parquet"), columns=COLS)
    dec = pd.read_parquet(ph.upstream("corpus", "method_filters.parquet"), columns=["method_id", "decision", "reasons"])
    m = m.merge(dec, on="method_id", how="left")
    rules = {"methods_in_corpus": int(len(m))}
    rules.update({f"decision_{k}": int(v) for k, v in m["decision"].value_counts().items()})
    if subset == "test":
        wanted = {"keep_test"}
    else:
        wanted = {"keep", "review"} if ph.cfg["method_filters"].get("include_review", True) else {"keep"}
    m = m[m["decision"].isin(wanted)]
    rules["eligible_methods"] = int(len(m))
    repos = repos[repos["status"] == "included"].copy()
    # groups (strata)
    by = s.get("group_by", "domain")
    if by == "none":
        repos["group"] = "all"
    elif by == "size":
        repos["group"] = repos["size_class"]
    else:
        repos["group"] = repos["domain"].fillna("unknown")
        counts = repos["group"].value_counts()
        small = counts[counts < int(s.get("min_group_repos", 8))].index
        # small domains are pooled with the repositories of unknown domain (a pooled group of 2-3 repositories
        # would itself be too small to sample)
        repos.loc[repos["group"].isin(small) | (repos["group"] == "unknown"), "group"] = "other or unknown domain"
    eligible_per_repo = m.groupby("repository").size()
    repos["eligible_methods"] = repos["repository"].map(eligible_per_repo).fillna(0).astype(int)
    repos = repos[repos["eligible_methods"] > 0]
    m = m[m["repository"].isin(set(repos["repository"]))].merge(repos[["repository", "group"]], on="repository")
    rules["eligible_repositories"] = int(len(repos))
    return m, repos, rules


def allocate(total: int, groups: pd.DataFrame, how: str) -> dict:
    """Methods per group: equal, or proportional to the group's eligible methods (largest remainder)."""
    if total <= 0 or groups.empty:
        return {g: 0 for g in groups.index}
    if how == "proportional":
        raw = groups["methods"] / groups["methods"].sum() * total
    else:
        raw = pd.Series(total / len(groups), index=groups.index)
    base = np.floor(raw).astype(int)
    rest = total - base.sum()
    for g in (raw - base).sort_values(ascending=False).index[:rest]:
        base[g] += 1
    return base.to_dict()


def draw_random(ph, m: pd.DataFrame, repos: pd.DataFrame, total: int, used: set, rng, split: str) -> list[dict]:
    s = ph.cfg["sampling"]
    k = int(s.get("methods_per_repo", 3))
    groups = repos.groupby("group").agg(repos=("repository", "size"), methods=("eligible_methods", "sum"))
    alloc = allocate(total, groups, s.get("allocation", "equal"))
    avail = m[~m["method_id"].isin(used)]
    by_repo = {r: g for r, g in avail.groupby("repository")}
    out = []
    for g, target in alloc.items():
        if target <= 0:
            continue
        g_repos = sorted(repos.loc[repos["group"] == g, "repository"])
        N_h = len(g_repos)
        # at least 2 repositories per group whenever possible: the variance of a stratum needs 2 PSUs
        n_h = min(N_h, max(2, math.ceil(target / k)))
        chosen = list(rng.choice(g_repos, size=n_h, replace=False)) if n_h else []
        # spread the target over the chosen repositories (at most k each); repositories with fewer
        # available methods give what they have
        per = {r: 0 for r in chosen}
        left = target
        while left > 0:
            progressed = False
            for r in chosen:
                if left and per[r] < k and per[r] < len(by_repo.get(r, [])):
                    per[r] += 1
                    left -= 1
                    progressed = True
            if not progressed:
                break
        for r in chosen:
            pool = by_repo.get(r)
            if pool is None or not per[r]:
                continue
            M_i = int(repos.loc[repos["repository"] == r, "eligible_methods"].iloc[0])
            idx = rng.choice(len(pool), size=per[r], replace=False)
            for row in pool.iloc[sorted(idx)].itertuples():
                p1, p2 = n_h / N_h, per[r] / M_i
                out.append({"method_id": row.method_id, "repository": r, "group": g, "split": split, "purpose": "random",
                            "signal": None, "stage1_prob": round(p1, 6), "stage2_prob": round(p2, 6),
                            "inclusion_prob": p1 * p2, "weight": round(1 / (p1 * p2), 4),
                            "class_name": row.class_name, "side": row.side, "selector": row.selector,
                            "is_test": row.is_test, "url": row.url})
        if left > 0:
            ph.warn(f"{split}: group {g!r} has too few methods for its allocation ({left} short)")
    return out


def signal_mask(m: pd.DataFrame, spec: dict) -> pd.Series:
    sels, globs = set(spec.get("selectors") or []), set(spec.get("globals") or [])
    mask = pd.Series(False, index=m.index)
    if sels:
        mask |= m["sent_selectors"].map(lambda xs: bool(sels.intersection(xs)))
    if globs:
        mask |= m["referenced_globals"].map(lambda xs: bool(globs.intersection(xs)))
    return mask


def draw_targeted(ph, m: pd.DataFrame, total: int, used: set, rng, split: str, masks: dict) -> list[dict]:
    if total <= 0:
        return []
    signals = [k for k, v in masks.items() if v.any()]
    per = max(1, math.ceil(total / max(len(signals), 1)))
    out = []
    for name in signals:
        pool = m[masks[name] & ~m["method_id"].isin(used)]
        # at most one method per repository and signal, so a repository full of one idiom cannot fill the signal
        pool = pool.sample(frac=1, random_state=int(rng.integers(1 << 31))).drop_duplicates("repository")
        if pool.empty:
            continue
        take = pool.sample(min(per, len(pool)), random_state=int(rng.integers(1 << 31)))
        for row in take.itertuples():
            out.append({"method_id": row.method_id, "repository": row.repository, "group": row.group, "split": split,
                        "purpose": "targeted", "signal": name, "stage1_prob": None, "stage2_prob": None,
                        "inclusion_prob": None, "weight": None, "class_name": row.class_name, "side": row.side,
                        "selector": row.selector, "is_test": row.is_test, "url": row.url})
            used.add(row.method_id)
        if len(out) >= total:
            break
    return out[:total]


def literature(ph, rng) -> tuple[pd.DataFrame, dict]:
    lit = pd.read_parquet(ph.upstream("literature", "literature_tasks.parquet"), columns=["task_uid", "benchmark", "dedup_key", "task_type"])
    used_keys, rows, plan = set(), [], {}
    for split, key in (("discovery", "literature_discovery_per_benchmark"), ("annotation", "literature_annotation_per_benchmark")):
        k = int(ph.sizes[key])
        plan[split] = {}
        for b, g in sorted(lit.groupby("benchmark"), key=lambda x: x[0]):
            g = g[~g["dedup_key"].isin(used_keys)].drop_duplicates("dedup_key")
            take = g.sample(min(k, len(g)), random_state=int(rng.integers(1 << 31))) if len(g) else g
            used_keys |= set(take["dedup_key"])
            plan[split][b] = {"population": int(lit[lit["benchmark"] == b]["dedup_key"].nunique()), "sampled": int(len(take))}
            rows += [{"task_uid": t.task_uid, "benchmark": b, "task_type": t.task_type, "split": split,
                      "weight": round(plan[split][b]["population"] / max(len(take), 1), 3)} for t in take.itertuples()]
    return pd.DataFrame(rows), plan


def run(ph):
    rng = np.random.default_rng(ph.seed)
    m, repos, rules = frame(ph, "production")
    mt, repos_t, rules_t = frame(ph, "test")
    ph.log.info("frame (production methods): " + ", ".join(f"{k}={v:,}" for k, v in rules.items()))
    ph.log.info(f"test methods, kept separately: {rules_t['eligible_methods']:,} in {rules_t['eligible_repositories']:,} repositories")
    groups = repos.groupby("group").agg(repositories=("repository", "size"), methods=("eligible_methods", "sum"))
    ph.log.info(f"{len(groups)} repository groups: " + ", ".join(f"{g} ({r.repositories})" for g, r in groups.iterrows()))
    masks = {name: signal_mask(m, spec) for name, spec in (ph.cfg["sampling"].get("targeted_signals") or {}).items()}
    signal_counts = {k: int(v.sum()) for k, v in masks.items()}

    used, rows = set(), []
    for split, rkey, tkey in SPLITS:
        r = draw_random(ph, m, repos, int(ph.sizes[rkey]), used, rng, split)
        used |= {x["method_id"] for x in r}
        t = draw_targeted(ph, m, int(ph.sizes[tkey]) if tkey else 0, used, rng, split, masks)
        used |= {x["method_id"] for x in t}
        tt = draw_random(ph, mt, repos_t, int(ph.sizes.get(f"pharo_{split}_tests", 0)), used, rng, split)
        used |= {x["method_id"] for x in tt}
        for x in r + t:
            x["subset"] = "production"
        for x in tt:
            x["subset"] = "test"
        rows += r + t + tt
        ph.log.info(f"split {split:<10}: {len(r)} random methods from {len({x['repository'] for x in r})} repositories, "
                    f"{len(t)} targeted, {len(tt)} test methods (kept separately)")
    sample = pd.DataFrame(rows)
    why = dict(zip(pd.concat([m["method_id"], mt["method_id"]]), pd.concat([m["decision"], mt["decision"]])))
    rsn = dict(zip(pd.concat([m["method_id"], mt["method_id"]]), pd.concat([m["reasons"], mt["reasons"]])))
    sample["filter_decision"] = sample["method_id"].map(why)
    sample["filter_reasons"] = sample["method_id"].map(rsn)
    assert sample["method_id"].is_unique, "splits must be disjoint"
    write_csv(ph.path("pilot_sample.csv"), sample)

    lit, lit_plan = literature(ph, rng)
    write_csv(ph.path("literature_sample.csv"), lit)
    ph.log.info(f"literature: {int((lit['split'] == 'discovery').sum())} discovery tasks, "
                f"{int((lit['split'] == 'annotation').sum())} annotation tasks, {lit['benchmark'].nunique()} benchmarks")

    design = {
        "description": "stratified two-stage cluster sampling: strata = repository groups, PSU = repository, SSU = method",
        "weight_formula": "w = (N_h / n_h) * (M_i / m_i)",
        "estimator": "ratio estimator sum(w*y)/sum(w); variance by linearisation with repositories as PSUs within strata",
        "seed": ph.seed, "profile": ph.cfg["profile_name"], "settings": ph.cfg["sampling"],
        "frame_rules": rules, "test_frame": rules_t,
        "subsets": "production = method filter decision keep (+ review); test = keep_test, sampled separately with the same design",
        "groups": {g: {"eligible_repositories": int(r.repositories), "eligible_methods": int(r.methods)} for g, r in groups.iterrows()},
        "splits": {s: {"random": int(((sample["split"] == s) & (sample["purpose"] == "random")).sum()),
                       "targeted": int(((sample["split"] == s) & (sample["purpose"] == "targeted")).sum()),
                       "repositories": int(sample.loc[sample["split"] == s, "repository"].nunique())}
                   for s, *_ in SPLITS},
        "targeted_signals": {"eligible_methods_per_signal": signal_counts,
                             "note": "targeted methods have no design weight and are never used for frequencies"},
        "literature": {"plan": lit_plan, "note": "one task per problem (dedup_key); weight = problems of the benchmark / sampled"},
    }
    write_json(ph.path("sampling_design.json"), design)
    ph.metric("eligible_methods", rules["eligible_methods"])
    ph.metric("eligible_repositories", rules["eligible_repositories"])
    ph.metric("groups", int(len(groups)))
    for s, *_ in SPLITS:
        ph.metric(f"{s}_methods", int((sample["split"] == s).sum()))
    ph.metric("literature_discovery", int((lit["split"] == "discovery").sum()))
    ph.metric("literature_annotation", int((lit["split"] == "annotation").sum()))

    # ---------------------------------------------------------------- figures
    rnd = sample[(sample["purpose"] == "random") & (sample["subset"] == "production")]
    corpus_share = groups["methods"] / groups["methods"].sum()
    unweighted = rnd.groupby("group").size() / max(len(rnd), 1)
    weighted = rnd.groupby("group")["weight"].sum() / max(rnd["weight"].sum(), 1)
    order = corpus_share.sort_values(ascending=False).index
    fig, ax = F.plt.subplots(figsize=(9, max(3, 0.42 * len(order) + 1.5)))
    y = np.arange(len(order))[::-1]
    hgt = 0.26
    for i, (lab, ser) in enumerate([("corpus (eligible methods)", corpus_share), ("sample, unweighted", unweighted),
                                    ("sample, design-weighted", weighted)]):
        ax.barh(y + (1 - i) * hgt, [ser.get(g, 0) * 100 for g in order], height=hgt - 0.02, color=F.color(i), label=lab)
    ax.set_yticks(y, [F.wrap(g, 34) for g in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("% of methods")
    ax.set_title("Repository groups: corpus vs sample\n(unweighted over-represents small groups; design weights are unbiased, noisy at small n)")
    ax.legend(loc="lower right")
    F.save(fig, ph.figure("groups_corpus_vs_sample.png"),
           note=f"random samples of all splits ({len(rnd)} methods); allocation: {ph.cfg['sampling'].get('allocation')}")

    per_repo = rnd.groupby(["split", "repository"]).size()
    fig, ax = F.plt.subplots(figsize=(7, 3.6))
    vc = per_repo.value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values, color=F.SERIES[0], width=0.6)
    ax.set_xlabel("methods drawn from one repository in one split")
    ax.set_ylabel("repositories")
    ax.set_title(f"No repository dominates: at most {ph.cfg['sampling'].get('methods_per_repo')} methods per repository")
    ax.grid(axis="x", visible=False)
    F.save(fig, ph.figure("methods_per_sampled_repository.png"))

    sc = pd.Series(signal_counts).sort_values(ascending=False)
    F.barh(sc.index, sc.values / max(rules["eligible_methods"], 1) * 100, "Targeted signals: how rare they are in the corpus",
           "% of eligible methods", ph.figure("targeted_signals.png"), fmt="{:.2f}%",
           note="a random sample of n methods sees a signal of share p with probability 1-(1-p)^n: rare areas need targeting")
