"""Sample size, method population and proportional draws.

Sample size: Cochran (1977, eq. 4.2) for a proportion, with the finite population correction
(eq. 4.3). With p = 0.5 it is the largest sample any proportion needs, so every category share
measured on the sample has at most the chosen margin of error at the chosen confidence.

Draws: stratified random sampling with the repository as stratum and proportional allocation
(Cochran 1977, ch. 5). Every round of the saturation loop draws again from the methods not drawn
yet, so the union of all rounds stays a self-weighting sample of the corpus."""
from __future__ import annotations

from collections import defaultdict
import math
import random
from statistics import NormalDist

from ..storage.database import Database


def z_value(confidence: float) -> float:
    return NormalDist().inv_cdf(1 - (1 - confidence) / 2)


def cochran(population: int, confidence: float = 0.95, margin: float = 0.05, p: float = 0.5) -> dict:
    """Sample size for a proportion, corrected for a finite population."""
    z = z_value(confidence)
    n0 = z * z * p * (1 - p) / (margin * margin)
    n = n0 / (1 + (n0 - 1) / population) if population else n0
    return {"population": population, "confidence": confidence, "margin_of_error": margin, "proportion": p,
            "z": round(z, 4), "n0": round(n0, 2), "n": min(population, math.ceil(n))}


def margin_for(n: int, population: int, confidence: float = 0.95, p: float = 0.5) -> float:
    """Margin of error a sample of n methods gives (the inverse of `cochran`)."""
    if n <= 0:
        return 1.0
    fpc = (population - n) / (population - 1) if population > 1 else 0.0
    return z_value(confidence) * math.sqrt(p * (1 - p) / n * fpc)


def quiet_run_length(min_share: float, confidence: float = 0.95) -> int:
    """Methods in a row without a new category before saturation is claimed: a task with
    prevalence ``min_share`` would have appeared in them with probability ``confidence``
    (unseen with probability (1 - p)^L <= 1 - confidence)."""
    if min_share <= 0:
        return 0
    return math.ceil(math.log(1 - confidence) / math.log(1 - min_share))


def wilson(k: int, n: int, confidence: float = 0.95, population: int | None = None) -> tuple[float, float]:
    """Wilson score interval for k successes in n, narrowed by the finite population correction."""
    if n == 0:
        return 0.0, 0.0
    z = z_value(confidence)
    if population and population > n:
        z *= math.sqrt((population - n) / (population - 1))
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def latest_snapshots_sql() -> str:
    return "SELECT MAX(snapshot_id) FROM repository_snapshots GROUP BY repo_id"


def population(db: Database, repo_ids: set[int], include_tests: bool = True) -> dict[int, list[int]]:
    """Method ids per repository: parsed, not copied from another repository, latest snapshot."""
    rows = db.query(
        "SELECT method_id, repo_id, is_test FROM methods WHERE parse_status='ok' AND duplicate_of_method_id IS NULL"
        f" AND snapshot_id IN ({latest_snapshots_sql()}) AND repo_id IN ({','.join(map(str, sorted(repo_ids))) or 'NULL'})"
        " ORDER BY repo_id, method_id")
    out = defaultdict(list)
    for r in rows:
        if include_tests or not r["is_test"]:
            out[r["repo_id"]].append(r["method_id"])
    return dict(out)


def draw(pop: dict[int, list[int]], size: int, seed: int, exclude: set[int] = frozenset()) -> list[int]:
    """`size` methods, every repository in proportion to its methods not drawn yet; the shares
    are rounded with the largest remainder method so exactly `size` are drawn."""
    left = {k: [m for m in v if m not in exclude] for k, v in pop.items()}
    left = {k: v for k, v in left.items() if v}
    total = sum(len(v) for v in left.values())
    size = min(size, total)
    if size <= 0:
        return []
    exact = {k: size * len(v) / total for k, v in left.items()}
    alloc = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - alloc[k]), k))[:size - sum(alloc.values())]:
        alloc[k] += 1
    rng = random.Random(seed)
    chosen = []
    for k in sorted(left):
        chosen += rng.sample(left[k], alloc[k])
    rng.shuffle(chosen)
    return chosen


def method_records(db: Database, ids: list[int]) -> dict[int, dict]:
    """What the labeller and the experts need about each method, including a GitHub link."""
    out = {}
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for r in db.query(
                "SELECT m.method_id, m.repo_id, m.package, m.class_name, m.side, m.selector, m.protocol, m.source,"
                " m.source_path, m.line, m.is_test, r.full_name, s.commit_sha FROM methods m"
                " JOIN repositories r ON r.repo_id=m.repo_id JOIN repository_snapshots s ON s.snapshot_id=m.snapshot_id"
                f" WHERE m.method_id IN ({','.join('?' * len(chunk))})", chunk):
            d = dict(r)
            d["is_test"] = bool(d["is_test"])
            d["signature"] = f"{d['class_name']}{' class' if d['side'] == 'class' else ''}>>{d['selector']}"
            d["url"] = (f"https://github.com/{d['full_name']}/blob/{d['commit_sha']}/{d['source_path']}"
                        + (f"#L{d['line']}" if d["line"] else "")) if d["commit_sha"] and d["source_path"] else None
            out[d["method_id"]] = d
    return out
