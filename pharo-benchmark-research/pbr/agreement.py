"""Agreement and estimation statistics (phases 7, 8, 9).

Agreement between annotators (models or experts):
* ``krippendorff_alpha`` — any number of annotators, missing answers allowed, any distance: nominal for
  one label per item, MASI (Passonneau 2006) for sets of behaviours (multi-label). Reliable from 0.8,
  tentative from 0.667 (Krippendorff 2004, ch. 11). Same formulas as ../labeling-pharo-repos agreement.py,
  generalised to set-valued answers.
* ``fleiss_kappa`` (Fleiss 1971), ``gwet_ac1`` (Gwet 2008: robust when one category dominates),
  ``cohen_kappa`` (two annotators).
* ``agreement_level`` — full / partial / none for a set of answers.

Classification against human reference labels: ``multilabel_prf`` (micro / macro precision, recall, F1;
per label), ``exact_match``.

Estimation: ``wilson`` interval for a proportion, ``weighted_share`` (design-weighted share with a
linearisation variance, two-stage cluster design: repositories are the clusters), ``ppi_mean``
(prediction-powered inference, Angelopoulos et al., Science 2023: model predictions on a large sample
corrected with human labels on a small one), ``rule_of_three`` (upper bound when nothing is observed).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import math

Z95 = 1.959964


def z_value(conf: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(0.5 + conf / 2))


# ---------------------------------------------------------------------------------------------- distances
def nominal(a, b) -> float:
    return 0.0 if a == b else 1.0


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def masi(a, b) -> float:
    """MASI distance between two label sets: 1 - Jaccard × monotonicity (1 equal, 2/3 subset, 1/3 overlap, 0)."""
    a, b = frozenset(a), frozenset(b)
    if a == b:
        return 0.0
    j = jaccard(a, b)
    m = 2 / 3 if (a <= b or b <= a) else (1 / 3 if a & b else 0.0)
    return 1 - j * m


# ---------------------------------------------------------------------------------------------- agreement
def krippendorff_alpha(units: list[list], distance=nominal) -> float | None:
    """units: one list of answers per item (None = missing). ``distance``: nominal (labels) or masi (sets)."""
    pairable = [[v for v in u if v is not None] for u in units]
    pairable = [u for u in pairable if len(u) >= 2]
    n = sum(len(u) for u in pairable)
    if n <= 1:
        return None
    do = 0.0
    for u in pairable:
        m = len(u)
        do += sum(distance(u[i], u[j]) for i in range(m) for j in range(m) if i != j) / (m - 1)
    do /= n
    values = [v for u in pairable for v in u]
    # expected disagreement over every pair of pairable values (grouped by identical value for speed)
    counts = Counter(frozenset(v) if isinstance(v, (set, frozenset, list, tuple)) else v for v in values)
    keys = list(counts)
    de = 0.0
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i != j:
                de += counts[a] * counts[b] * distance(a, b)
    de /= n * (n - 1)
    if de == 0:
        return 1.0 if do == 0 else None
    return round(1 - do / de, 4)


def _complete(units):
    return [u for u in units if u and all(v is not None for v in u)]


def fleiss_kappa(units: list[list]) -> float | None:
    units = _complete(units)
    if not units or len(units[0]) < 2:
        return None
    r = len(units[0])
    cats = sorted({v for u in units for v in u}, key=str)
    p_items = [(sum(c * c for c in Counter(u).values()) - r) / (r * (r - 1)) for u in units]
    pbar = sum(p_items) / len(units)
    pe = sum((sum(u.count(c) for u in units) / (len(units) * r)) ** 2 for c in cats)
    return round((pbar - pe) / (1 - pe), 4) if pe < 1 else None


def gwet_ac1(units: list[list]) -> float | None:
    units = _complete(units)
    if not units:
        return None
    r = len(units[0])
    cats = sorted({v for u in units for v in u}, key=str)
    q = len(cats)
    if r < 2:
        return None
    if q < 2:
        return 1.0
    pa = sum(sum(c * (c - 1) for c in Counter(u).values()) / (r * (r - 1)) for u in units) / len(units)
    pk = [sum(u.count(c) for u in units) / (len(units) * r) for c in cats]
    pe = sum(p * (1 - p) for p in pk) / (q - 1)
    return round((pa - pe) / (1 - pe), 4) if pe < 1 else None


def cohen_kappa(a: list, b: list) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if not n:
        return None
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else (1.0 if po == 1 else None)


def agreement_level(sets: list) -> str:
    """full: every annotator gave the same set; partial: some behaviour shared by at least two
    annotators (or identical sets for two of three); none: no behaviour shared."""
    sets = [frozenset(s) for s in sets if s is not None]
    if len(sets) < 2:
        return "missing"
    if all(s == sets[0] for s in sets):
        return "full"
    for a, b in combinations(sets, 2):
        if a & b:
            return "partial"
    return "none"


def majority_set(sets: list, k: int | None = None) -> frozenset:
    """Behaviours given by a strict majority of the annotators (k = how many must agree)."""
    sets = [frozenset(s) for s in sets if s is not None]
    if not sets:
        return frozenset()
    k = k or (len(sets) // 2 + 1)
    c = Counter(x for s in sets for x in s)
    return frozenset(x for x, n in c.items() if n >= k)


def pairwise(units_by_annotator: dict[str, list], fn) -> dict:
    """fn(a_answers, b_answers) for every pair of annotators."""
    names = list(units_by_annotator)
    return {f"{a}~{b}": fn(units_by_annotator[a], units_by_annotator[b]) for a, b in combinations(names, 2)}


# ---------------------------------------------------------------------------------------------- vs reference
def multilabel_prf(gold: list, pred: list, labels: list | None = None) -> dict:
    """gold, pred: lists of label sets (same items). Micro and macro P/R/F1, exact match, per label."""
    tp, fp, fn = Counter(), Counter(), Counter()
    exact = 0
    n = 0
    for g, p in zip(gold, pred):
        if g is None or p is None:
            continue
        n += 1
        g, p = set(g), set(p)
        exact += g == p
        for x in p & g:
            tp[x] += 1
        for x in p - g:
            fp[x] += 1
        for x in g - p:
            fn[x] += 1
    labels = labels or sorted(set(tp) | set(fp) | set(fn))

    def prf(t, f_p, f_n):
        p = t / (t + f_p) if t + f_p else None
        r = t / (t + f_n) if t + f_n else None
        f = 2 * p * r / (p + r) if p and r else (0.0 if (p is not None and r is not None) else None)
        return p, r, f

    per = {}
    for x in labels:
        p, r, f = prf(tp[x], fp[x], fn[x])
        per[x] = {"precision": _r(p), "recall": _r(r), "f1": _r(f), "support": tp[x] + fn[x], "predicted": tp[x] + fp[x]}
    mp, mr, mf = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    supported = [v for v in per.values() if v["support"] > 0]
    macro = {k: _r(sum(v[k] or 0 for v in supported) / len(supported)) if supported else None
             for k in ("precision", "recall", "f1")}
    return {"n": n, "exact_match": _r(exact / n) if n else None,
            "micro": {"precision": _r(mp), "recall": _r(mr), "f1": _r(mf)}, "macro": macro, "per_label": per}


def _r(x, k=4):
    return None if x is None else round(float(x), k)


# ---------------------------------------------------------------------------------------------- estimation
def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = z_value(conf)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def rule_of_three(n: int, conf: float = 0.95) -> float:
    """Upper confidence bound of a proportion when 0 of n items show it: 1 - (1-conf)^(1/n) (≈ 3/n)."""
    return 1.0 if n == 0 else 1 - (1 - conf) ** (1 / n)


def weighted_share(y: list[float], w: list[float], clusters: list, strata: list | None = None,
                   conf: float = 0.95) -> dict:
    """Design-weighted share (ratio estimator sum w*y / sum w) with a linearisation (Taylor) variance
    that treats repositories as primary sampling units within strata (with-replacement approximation,
    conservative). y: 0/1 per sampled method; w: design weight (methods represented)."""
    import numpy as np
    y, w = np.asarray(y, float), np.asarray(w, float)
    if not len(y) or w.sum() == 0:
        return {"share": None, "ci_low": None, "ci_high": None, "n": 0}
    R = float((w * y).sum() / w.sum())
    z = w * (y - R) / w.sum()
    strata = strata if strata is not None else ["all"] * len(y)
    var = 0.0
    by = defaultdict(lambda: defaultdict(float))
    for zi, c, h in zip(z, clusters, strata):
        by[h][c] += zi
    for h, cl in by.items():
        totals = np.array(list(cl.values()))
        nh = len(totals)
        if nh > 1:
            var += nh / (nh - 1) * ((totals - totals.mean()) ** 2).sum()
    half = z_value(conf) * math.sqrt(max(var, 0))
    return {"share": round(R, 5), "ci_low": round(max(0.0, R - half), 5), "ci_high": round(min(1.0, R + half), 5),
            "n": int(len(y)), "se": round(math.sqrt(max(var, 0)), 5)}


def ppi_mean(f_unlabeled: list[float], f_labeled: list[float], y_labeled: list[float], conf: float = 0.95) -> dict:
    """Prediction-powered estimate of a mean (here a behaviour's share): mean of the model's predictions on
    the large sample + mean of (human - model) on the human-labelled sample; valid interval whatever the
    model's errors (Angelopoulos et al. 2023)."""
    import numpy as np
    fu, fl, yl = (np.asarray(x, float) for x in (f_unlabeled, f_labeled, y_labeled))
    if not len(fu) or not len(fl):
        return {"share": None}
    rect = yl - fl
    est = fu.mean() + rect.mean()
    var = fu.var(ddof=1) / len(fu) if len(fu) > 1 else 0.0
    var += rect.var(ddof=1) / len(fl) if len(fl) > 1 else 0.0
    half = z_value(conf) * math.sqrt(var)
    return {"share": round(float(min(1, max(0, est))), 5), "ci_low": round(float(max(0, est - half)), 5),
            "ci_high": round(float(min(1, est + half)), 5), "model_share": round(float(fu.mean()), 5),
            "correction": round(float(rect.mean()), 5), "n_labeled": int(len(fl)), "n_unlabeled": int(len(fu))}
