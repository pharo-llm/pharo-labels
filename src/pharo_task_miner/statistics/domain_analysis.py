"""RQ5 — association between application domains and operations.

Besides presence-based P(o | d), the *balanced share lift* compares the mean
share of a repository's methods carrying o within domain d to the same mean
over all repositories (1 = typical, 2 = twice as common). Presence saturates
for large repositories; the share lift remains informative.

P(o | d) = repositories in domain d containing o / analysed repositories in d.
Lift = P(o | d) / P(o). Cross-domain coverage = number of domains (with at
least ``min_domain_repos`` analysed repositories) in which o occurs.
Repositories are multi-label; a repository counts in every domain it carries.
"""
from __future__ import annotations

from collections import defaultdict

from .prevalence import Corpus
from ..storage.database import Database


def repo_domains(db: Database, method: str = "deterministic_rules") -> dict[int, list[str]]:
    out = defaultdict(list)
    for r in db.query("SELECT repo_id, label FROM domain_labels WHERE method=?", (method,)):
        out[r["repo_id"]].append(r["label"])
    return out


def domain_operation_matrix(db: Database, corpus: Corpus, min_domain_repos: int = 3) -> tuple[list[dict], list[dict]]:
    domains = repo_domains(db)
    ops_by_repo = defaultdict(set)
    counts = defaultdict(lambda: defaultdict(int))  # op -> repo -> methods
    for m, ops in corpus.labels.items():
        rid = corpus.method_repo[m]
        ops_by_repo[rid] |= ops
        for o in ops:
            counts[o][rid] += 1

    def share(o, rid):
        n = corpus.methods_per_repo.get(rid, 0)
        return counts[o].get(rid, 0) / n if n else 0.0

    repos_in_domain = defaultdict(set)
    for rid in corpus.repos:
        for d in domains.get(rid, ["unknown"]):
            repos_in_domain[d].add(rid)
    all_ops = sorted({o for s in ops_by_repo.values() for o in s})
    n = len(corpus.repos) or 1
    overall = {o: sum(1 for r in corpus.repos if o in ops_by_repo[r]) / n for o in all_ops}
    overall_share = {o: sum(share(o, r) for r in corpus.repos) / n for o in all_ops}
    cells, cross = [], []
    for o in all_ops:
        present_in = []
        for d, rs in sorted(repos_in_domain.items()):
            k = sum(1 for r in rs if o in ops_by_repo[r])
            p = k / len(rs) if rs else None
            bshare = sum(share(o, r) for r in rs) / len(rs) if rs else None
            cells.append({"domain": d, "operation": o, "domain_repositories": len(rs), "repositories_with_operation": k,
                          "p_operation_given_domain": None if p is None else round(p, 4),
                          "lift": round(p / overall[o], 3) if p is not None and overall[o] else None,
                          "balanced_share_in_domain": None if bshare is None else round(bshare, 5),
                          "balanced_share_lift": round(bshare / overall_share[o], 3) if bshare is not None and overall_share[o] else None,
                          "sufficient_sample": len(rs) >= min_domain_repos})
            if k and len(rs) >= min_domain_repos and d != "unknown":
                present_in.append(d)
        eligible = [d for d, rs in repos_in_domain.items() if len(rs) >= min_domain_repos and d != "unknown"]
        cross.append({"operation": o, "domains_with_operation": len(present_in), "domains_considered": len(eligible),
                      "domains": ";".join(present_in), "overall_repo_prevalence": round(overall[o], 4)})
    cross.sort(key=lambda r: (-r["domains_with_operation"], -r["overall_repo_prevalence"]))
    return cells, cross
