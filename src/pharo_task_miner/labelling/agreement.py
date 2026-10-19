"""Agreement and estimation statistics for several annotators (models or experts).

Agreement, nominal categories:

* ``krippendorff_alpha``: any number of annotators, missing answers allowed (Krippendorff 2004,
  *Content Analysis*, ch. 11). Reliable from 0.8, tentative conclusions from 0.667.
* ``fleiss_kappa`` (Fleiss 1971) and ``gwet_ac1`` (Gwet 2008) on the items every annotator
  answered. AC1 does not collapse when one category dominates (the "kappa paradox": high
  agreement, low kappa, Feinstein & Cicchetti 1990), which is the case of task categories.
* ``cohen_kappa`` for two annotators.

Estimation from a stratified expert sample drawn inside the classified sample (two-phase sampling
for stratification, Cochran 1977, ch. 12): stratum h holds N_h of the n classified methods, m_h of
them are checked by the experts, weight W_h = N_h / n.

* ``stratified_mean``: e.g. a model's accuracy, sum_h W_h * mean_h, with its variance.
* ``ppi_share``: the share of a category corrected with the experts' labels (prediction-powered
  inference, Angelopoulos et al., Science 2023): the models' share on all n methods plus the
  stratified mean of (expert label - model label) on the checked ones. Unbiased whatever the
  models' errors, with a valid interval.
"""
from __future__ import annotations

from collections import Counter
import math

from .sampling import z_value


def cohen_kappa(a: list, b: list) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if not n:
        return None
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def krippendorff_alpha(units: list[list]) -> float | None:
    """units: one list of answers per item (None = missing). Nominal metric."""
    coinc = Counter()
    for answers in units:
        vals = [v for v in answers if v is not None]
        m = len(vals)
        if m < 2:
            continue
        counts = Counter(vals)
        for c, nc in counts.items():
            for k, nk in counts.items():
                coinc[(c, k)] += nc * (nk - (c == k)) / (m - 1)
    n = sum(coinc.values())
    if n <= 1:
        return None
    marg = Counter()
    for (c, _), v in coinc.items():
        marg[c] += v
    do = sum(v for (c, k), v in coinc.items() if c != k) / n
    de = sum(marg[c] * marg[k] for c in marg for k in marg if c != k) / (n * (n - 1))
    return round(1 - do / de, 3) if de > 0 else None


def _complete(units: list[list]) -> list[list]:
    return [u for u in units if u and all(v is not None for v in u)]


def fleiss_kappa(units: list[list]) -> float | None:
    units = _complete(units)
    if not units:
        return None
    r = len(units[0])
    cats = sorted({v for u in units for v in u}, key=str)
    if r < 2:
        return None
    p_items = [(sum(c * c for c in Counter(u).values()) - r) / (r * (r - 1)) for u in units]
    pbar = sum(p_items) / len(units)
    pj = [sum(u.count(c) for u in units) / (len(units) * r) for c in cats]
    pe = sum(p * p for p in pj)
    return round((pbar - pe) / (1 - pe), 3) if pe < 1 else None


def gwet_ac1(units: list[list]) -> float | None:
    units = _complete(units)
    if not units:
        return None
    r = len(units[0])
    cats = sorted({v for u in units for v in u}, key=str)
    q = len(cats)
    if r < 2 or q < 2:
        return 1.0 if units and q == 1 else None
    pa = sum((sum(c * (c - 1) for c in Counter(u).values())) / (r * (r - 1)) for u in units) / len(units)
    pk = [sum(u.count(c) for u in units) / (len(units) * r) for c in cats]
    pe = sum(p * (1 - p) for p in pk) / (q - 1)
    return round((pa - pe) / (1 - pe), 3) if pe < 1 else None


def majority(answers: list) -> tuple[str | None, str]:
    """(the label at least half of the annotators give, or None; the agreement level)."""
    vals = [v for v in answers if v is not None]
    if not vals:
        return None, "none"
    (top, k), *rest = Counter(vals).most_common()
    if k == len(answers):
        return top, f"{k}/{len(answers)}"
    if k * 2 > len(answers) or (k * 2 == len(answers) and (not rest or rest[0][1] < k)):
        return top, f"{k}/{len(answers)}"
    return None, "none"


def _ci(est: float, var: float, conf: float) -> tuple[float, float]:
    half = z_value(conf) * math.sqrt(max(var, 0.0))
    return max(0.0, est - half), min(1.0, est + half)


def stratified_mean(strata: dict[str, dict], values: dict[str, list[float]], conf: float = 0.95) -> dict:
    """strata: name -> {"N": methods in the stratum, "m": checked}; values: name -> the checked
    values (0/1). Strata without values count with weight zero (reported as such)."""
    n = sum(s["N"] for s in strata.values())
    est, var, used = 0.0, 0.0, 0
    for h, s in strata.items():
        v = values.get(h) or []
        if not v or not n:
            continue
        w, m = s["N"] / n, len(v)
        mean = sum(v) / m
        s2 = sum((x - mean) ** 2 for x in v) / (m - 1) if m > 1 else mean * (1 - mean)
        est += w * mean
        var += w * w * s2 / m * max(0.0, 1 - m / s["N"])
        used += m
    covered = sum(s["N"] for h, s in strata.items() if values.get(h)) / n if n else 0
    est = est / covered if covered else 0.0
    lo, hi = _ci(est, var / (covered * covered) if covered else 0.0, conf)
    return {"estimate": round(est, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4), "checked": used}


def ppi_share(model_share: float, n: int, strata: dict[str, dict], diffs: dict[str, list[float]],
              conf: float = 0.95, population: int | None = None) -> dict:
    """model_share: the category's share among the n classified methods (model labels);
    diffs: per stratum, (expert says this category) - (models say it), 1/0/-1, on the checked."""
    rect, var_r = 0.0, 0.0
    for h, s in strata.items():
        d = diffs.get(h) or []
        if not d:
            continue
        w, m = s["N"] / n, len(d)
        mean = sum(d) / m
        s2 = sum((x - mean) ** 2 for x in d) / (m - 1) if m > 1 else 0.0
        rect += w * mean
        var_r += w * w * s2 / m * max(0.0, 1 - m / s["N"])
    est = min(1.0, max(0.0, model_share + rect))
    fpc = (population - n) / (population - 1) if population and population > n else 1.0
    var = est * (1 - est) / n * fpc + var_r
    lo, hi = _ci(est, var, conf)
    return {"share": round(est, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "model_share": round(model_share, 4), "correction": round(rect, 4)}
