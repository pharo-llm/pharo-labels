"""Sampling modes: all_repositories, random_sample, stratified_sample,
diversity_sample and manual_selection. All are seeded and every inclusion or
exclusion is returned with a reason so it can be recorded."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path
import random

STRATEGIES = ("all_repositories", "random_sample", "stratified_sample", "diversity_sample", "manual_selection")
ALIASES = {"all": "all_repositories", "random": "random_sample", "stratified": "stratified_sample",
           "diversity": "diversity_sample", "manual": "manual_selection"}


def _cap_owner(order: list[dict], size: int, max_per_owner: int) -> tuple[list[dict], list[tuple[dict, str]]]:
    chosen, skipped, per = [], [], Counter()
    for p in order:
        if len(chosen) >= size:
            skipped.append((p, "sample size reached"))
        elif max_per_owner and per[p["owner"]] >= max_per_owner:
            skipped.append((p, f"owner cap ({max_per_owner}) reached for {p['owner']}"))
        else:
            per[p["owner"]] += 1
            chosen.append(p)
    return chosen, skipped


def stratum_of(p: dict, strata: list[str]) -> str:
    return "|".join(str(p.get(s)) for s in strata)


def stratified(eligible: list[dict], size: int, strata: list[str], rng: random.Random, max_per_owner: int):
    groups = defaultdict(list)
    for p in eligible:
        groups[stratum_of(p, strata)].append(p)
    for g in groups.values():
        rng.shuffle(g)
    total = len(eligible)
    # proportional allocation (largest remainder) with at least one per stratum when possible
    quotas = {k: size * len(v) / total for k, v in groups.items()} if total else {}
    alloc = {k: max(1, int(q)) if groups[k] else 0 for k, q in quotas.items()}
    while sum(alloc.values()) > size and any(a > 1 for a in alloc.values()):
        k = max((k for k in alloc if alloc[k] > 1), key=lambda k: alloc[k] - quotas[k])
        alloc[k] -= 1
    rema = sorted(quotas, key=lambda k: -(quotas[k] - int(quotas[k])))
    i = 0
    while sum(alloc.values()) < min(size, total) and rema:
        k = rema[i % len(rema)]
        if alloc[k] < len(groups[k]):
            alloc[k] += 1
        i += 1
        if i > 10 * len(rema) + size:
            break
    chosen, skipped, per = [], [], Counter()
    for k, members in sorted(groups.items()):
        taken = 0
        for p in members:
            if taken < alloc[k] and not (max_per_owner and per[p["owner"]] >= max_per_owner):
                chosen.append(p)
                per[p["owner"]] += 1
                taken += 1
            else:
                skipped.append((p, f"stratum {k} quota ({alloc[k]}) filled" if taken >= alloc[k]
                                else f"owner cap ({max_per_owner}) reached for {p['owner']}"))
    return chosen[:size], skipped + [(p, "sample size reached") for p in chosen[size:]], alloc


def diversity(eligible: list[dict], size: int, attributes: list[str], rng: random.Random, max_per_owner: int):
    """Greedy max-coverage: repeatedly add the repository whose attribute values
    are least represented so far (sum of 1/(1+count)), ties broken randomly."""
    pool = list(eligible)
    rng.shuffle(pool)
    counts = {a: Counter() for a in attributes}
    per_owner = Counter()
    chosen = []
    while pool and len(chosen) < size:
        best, best_score = None, -1.0
        for p in pool:
            if max_per_owner and per_owner[p["owner"]] >= max_per_owner:
                continue
            s = sum(1.0 / (1 + counts[a][str(p.get(a))]) for a in attributes)
            if s > best_score:
                best, best_score = p, s
        if best is None:
            break
        chosen.append(best)
        pool.remove(best)
        per_owner[best["owner"]] += 1
        for a in attributes:
            counts[a][str(best.get(a))] += 1
    skipped = [(p, "not selected by diversity maximisation" if not (max_per_owner and per_owner[p["owner"]] >= max_per_owner)
                else f"owner cap ({max_per_owner}) reached") for p in pool]
    return chosen, skipped


def read_manual(path: Path) -> list[str]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [(r.get("full_name") or r.get("repository") or "").strip() for r in rows if (r.get("full_name") or r.get("repository"))]


def sample(eligible: list[dict], strategy: str, size: int, seed: int, cfg_sampling: dict, manual_file: Path | None = None):
    strategy = ALIASES.get(strategy, strategy)
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown sampling strategy {strategy!r}; choose one of {STRATEGIES}")
    rng = random.Random(seed)
    max_owner = int(cfg_sampling.get("max_per_owner") or 0)
    info = {"strategy": strategy, "seed": seed, "requested_size": size, "eligible": len(eligible)}
    if strategy == "all_repositories":
        order = sorted(eligible, key=lambda p: p["full_name"].lower())
        chosen = order[:size] if size else order
        skipped = [(p, "max_repositories reached") for p in order[len(chosen):]]
    elif strategy == "random_sample":
        order = list(eligible)
        rng.shuffle(order)
        chosen, skipped = _cap_owner(order, size, max_owner)
    elif strategy == "stratified_sample":
        chosen, skipped, alloc = stratified(eligible, size, cfg_sampling["strata"], rng, max_owner)
        info["allocation"] = alloc
    elif strategy == "diversity_sample":
        chosen, skipped = diversity(eligible, size, cfg_sampling["diversity_attributes"], rng, max_owner)
    else:
        if not manual_file:
            raise ValueError("manual_selection requires sampling.manual_selection_file (CSV with a full_name column)")
        wanted = {n.lower() for n in read_manual(manual_file)}
        chosen = [p for p in eligible if p["full_name"].lower() in wanted]
        skipped = [(p, "not in manual selection") for p in eligible if p["full_name"].lower() not in wanted]
        found = {p["full_name"].lower() for p in chosen}
        info["manual_not_eligible_or_missing"] = sorted(wanted - found)
    return chosen, skipped, info
