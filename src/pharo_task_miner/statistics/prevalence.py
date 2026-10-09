"""Module 12 — repository-level, method-level and repository-balanced prevalence.

Variants (sensitivity analysis):
* ``all``          every parsed method of the latest snapshot of each analysed repository
* ``deduplicated`` excludes methods whose source is duplicated from a canonical repository
* ``non_test``     excludes test classes/packages

Labels come from ``label_source`` (``rule_based`` by default). Level-3 labels
are rolled up to their level-2 and level-1 ancestors (a method counts once per
category). Special statuses are reported separately.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .uncertainty import wilson
from ..analysis.structural_features import LANGUAGE_FEATURES
from ..storage.database import Database, loads


class Corpus:
    def __init__(self, db: Database, variant: str = "all", label_source: str = "rule_based", repo_ids: set[int] | None = None):
        self.variant = variant
        latest = {r[0] for r in db.query("SELECT MAX(snapshot_id) FROM repository_snapshots GROUP BY repo_id")}
        sql = "SELECT method_id, repo_id, snapshot_id, is_test, duplicate_of_method_id, features_json FROM methods WHERE parse_status='ok'"
        self.method_repo: dict[int, int] = {}
        self.features: dict[int, dict] = {}
        for r in db.query(sql):
            if r["snapshot_id"] not in latest:
                continue
            if repo_ids is not None and r["repo_id"] not in repo_ids:
                continue
            if variant == "deduplicated" and r["duplicate_of_method_id"] is not None:
                continue
            if variant == "non_test" and r["is_test"]:
                continue
            self.method_repo[r["method_id"]] = r["repo_id"]
            self.features[r["method_id"]] = loads(r["features_json"], {}) or {}
        self.repos = sorted(set(self.method_repo.values()))
        self.methods_per_repo = defaultdict(int)
        for rid in self.method_repo.values():
            self.methods_per_repo[rid] += 1
        parent = {r["category_id"]: r["parent_id"] for r in db.query("SELECT category_id, parent_id FROM taxonomy_categories")}
        self.parent = parent
        self.labels: dict[int, set[str]] = defaultdict(set)
        for r in db.query("SELECT method_id, operation_id FROM method_labels WHERE label_source=?", (label_source,)):
            if r["method_id"] in self.method_repo:
                o = r["operation_id"]
                while o:
                    self.labels[r["method_id"]].add(o)
                    o = parent.get(o)
        self.status: dict[int, str] = {}
        for r in db.query("SELECT method_id, status FROM method_status WHERE label_source=?", (label_source,)):
            if r["method_id"] in self.method_repo:
                self.status[r["method_id"]] = r["status"]


def _stats_for(occ: dict[int, set[int]], corpus: Corpus, cfg_stats: dict, key: str) -> list[dict]:
    """occ: item -> set(method ids)."""
    level = float(cfg_stats.get("confidence_level", 0.95))
    iters = int(cfg_stats.get("bootstrap_iterations", 1000))
    seed = int(cfg_stats.get("random_seed", 42))
    n_repos, n_methods = len(corpus.repos), len(corpus.method_repo)
    # one shared set of bootstrap resamples (repositories drawn with replacement)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n_repos, size=(iters, n_repos)) if n_repos > 1 else None
    sizes = np.array([corpus.methods_per_repo[r] for r in corpus.repos], dtype=float)
    repo_pos = {r: i for i, r in enumerate(corpus.repos)}
    alpha = (1 - level) / 2
    out = []
    for item, mids in occ.items():
        per_repo = defaultdict(int)
        for m in mids:
            per_repo[corpus.method_repo[m]] += 1
        k_repo = len(per_repo)
        lo, hi = wilson(k_repo, n_repos, level)
        mlo, mhi = wilson(len(mids), n_methods, level)
        counts = np.zeros(n_repos)
        for r, c in per_repo.items():
            counts[repo_pos[r]] = c
        shares = counts / np.where(sizes > 0, sizes, 1)
        bal = float(shares.mean()) if n_repos else None
        if boot_idx is not None:
            means = shares[boot_idx].mean(axis=1)
            blo, bhi = float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))
        else:
            blo = bhi = bal
        top = max(per_repo.values()) if per_repo else 0
        out.append({
            key: item, "repositories": k_repo, "repositories_total": n_repos,
            "repo_prevalence": round(k_repo / n_repos, 4) if n_repos else None,
            "repo_prevalence_ci_low": None if lo is None else round(lo, 4), "repo_prevalence_ci_high": None if hi is None else round(hi, 4),
            "methods": len(mids), "methods_total": n_methods,
            "method_prevalence": round(len(mids) / n_methods, 5) if n_methods else None,
            "method_prevalence_ci_low": None if mlo is None else round(mlo, 5), "method_prevalence_ci_high": None if mhi is None else round(mhi, 5),
            "repo_balanced_frequency": None if bal is None else round(bal, 5),
            "repo_balanced_ci_low": None if blo is None else round(blo, 5), "repo_balanced_ci_high": None if bhi is None else round(bhi, 5),
            "dominant_repository_share": round(top / len(mids), 4) if mids else None,
            "_per_repo": dict(per_repo),
        })
    out.sort(key=lambda r: (-(r["repositories"]), -(r["methods"])))
    return out


def operation_prevalence(corpus: Corpus, cfg_stats: dict) -> list[dict]:
    occ = defaultdict(set)
    for m, ops in corpus.labels.items():
        for o in ops:
            occ[o].add(m)
    return _stats_for(occ, corpus, cfg_stats, "operation")


def status_prevalence(corpus: Corpus, cfg_stats: dict) -> list[dict]:
    occ = defaultdict(set)
    for m in corpus.method_repo:
        occ[corpus.status.get(m, "not_classified")].add(m)
    return _stats_for(occ, corpus, cfg_stats, "status")


def language_feature_prevalence(corpus: Corpus, cfg_stats: dict) -> list[dict]:
    occ = defaultdict(set)
    concept = {name: ct for name, _p, ct in LANGUAGE_FEATURES}
    for m, f in corpus.features.items():
        if not f:
            continue
        for name, pred, _ct in LANGUAGE_FEATURES:
            try:
                if pred(f):
                    occ[name].add(m)
            except KeyError:
                pass
    rows = _stats_for(occ, corpus, cfg_stats, "feature")
    for r in rows:
        r["concept_type"] = concept.get(r["feature"])
    return rows


def selector_prevalence(db: Database, corpus: Corpus, top: int = 500) -> list[dict]:
    per = defaultdict(lambda: defaultdict(int))
    for r in db.query("SELECT method_id, selector FROM message_sends"):
        rid = corpus.method_repo.get(r["method_id"])
        if rid is not None:
            per[r["selector"]][rid] += 1
    n = len(corpus.repos) or 1
    rows = []
    for sel, d in per.items():
        total = sum(d.values())
        rows.append({"selector": sel, "repositories": len(d), "repo_prevalence": round(len(d) / n, 4), "occurrences": total,
                     "dominant_repository_share": round(max(d.values()) / total, 4)})
    rows.sort(key=lambda r: (-r["repositories"], -r["occurrences"]))
    return rows[:top]


def api_prevalence(db: Database, corpus: Corpus, top: int = 300) -> tuple[list[dict], list[dict]]:
    per_name = defaultdict(set)
    per_fw = defaultdict(set)
    fw_methods = defaultdict(int)
    name_fw = {}
    for r in db.query("SELECT method_id, name, framework, defined_in_repository FROM api_references"):
        rid = corpus.method_repo.get(r["method_id"])
        if rid is None:
            continue
        if not r["defined_in_repository"]:
            per_name[r["name"]].add(rid)
            name_fw[r["name"]] = r["framework"]
        if r["framework"] and r["framework"] != "(repository)":
            per_fw[r["framework"]].add(rid)
            fw_methods[r["framework"]] += 1
    n = len(corpus.repos) or 1
    names = sorted(({"name": k, "framework": name_fw.get(k), "repositories": len(v), "repo_prevalence": round(len(v) / n, 4)}
                    for k, v in per_name.items()), key=lambda r: -r["repositories"])[:top]
    fws = sorted(({"framework": k, "repositories": len(v), "repo_prevalence": round(len(v) / n, 4),
                   "referencing_methods": fw_methods[k]} for k, v in per_fw.items()), key=lambda r: -r["repositories"])
    return names, fws


def selector_cooccurrence(db: Database, corpus: Corpus, min_repos: int = 3, top: int = 300) -> list[dict]:
    """Common combinations of selectors within methods, counted by repositories."""
    by_method = defaultdict(set)
    for r in db.query("SELECT method_id, selector FROM message_sends"):
        if r["method_id"] in corpus.method_repo:
            by_method[r["method_id"]].add(r["selector"])
    sel_repos = defaultdict(set)
    for m, sels in by_method.items():
        for s in sels:
            sel_repos[s].add(corpus.method_repo[m])
    frequent = {s for s, rs in sel_repos.items() if len(rs) >= min_repos}
    from ..analysis.method_descriptions import GENERIC
    frequent -= GENERIC
    pair_repos = defaultdict(set)
    pair_methods = defaultdict(int)
    for m, sels in by_method.items():
        s = sorted(sels & frequent)
        if len(s) > 40:
            continue
        rid = corpus.method_repo[m]
        for i, a in enumerate(s):
            for b in s[i + 1:]:
                pair_repos[(a, b)].add(rid)
                pair_methods[(a, b)] += 1
    n_m = len(by_method) or 1
    single_methods = defaultdict(int)
    for sels in by_method.values():
        for s in sels & frequent:
            single_methods[s] += 1
    rows = []
    for (a, b), rs in pair_repos.items():
        if len(rs) < min_repos:
            continue
        lift = (pair_methods[(a, b)] / n_m) / ((single_methods[a] / n_m) * (single_methods[b] / n_m))
        rows.append({"selector_a": a, "selector_b": b, "repositories": len(rs), "methods": pair_methods[(a, b)],
                     "lift": round(lift, 2)})
    rows.sort(key=lambda r: (-r["repositories"], -r["lift"]))
    return rows[:top]
