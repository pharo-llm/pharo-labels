"""Inter-annotator agreement for multi-label operation annotations.

* per-category Cohen's kappa for every annotator pair (binary presence);
* per-category Krippendorff's alpha (nominal, binary) across all annotators;
* mean item-level Jaccard agreement between annotator label sets.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations


def cohen_kappa(a: list[int], b: list[int]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1:
        return 1.0 if po == 1 else None
    return (po - pe) / (1 - pe)


def krippendorff_alpha_binary(units: list[list[int]]) -> float | None:
    """units: per item, the list of binary values given by its annotators."""
    pairable = [u for u in units if len(u) >= 2]
    if not pairable:
        return None
    o = defaultdict(float)  # coincidence matrix
    for u in pairable:
        m = len(u)
        for i, x in enumerate(u):
            for j, y in enumerate(u):
                if i != j:
                    o[(x, y)] += 1 / (m - 1)
    n0 = o[(0, 0)] + o[(0, 1)]
    n1 = o[(1, 0)] + o[(1, 1)]
    n = n0 + n1
    if n <= 1 or n0 == 0 or n1 == 0:
        return 1.0 if (o[(0, 1)] + o[(1, 0)]) == 0 else None
    do = o[(0, 1)] + o[(1, 0)]
    de = 2 * n0 * n1 / (n - 1)
    return 1 - do / de


def jaccard(a: set, b: set) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def agreement_report(decisions: dict[str, dict[str, set]], categories: list[str]) -> dict:
    """decisions: sample_id -> annotator -> set(operation ids)."""
    multi = {s: d for s, d in decisions.items() if len(d) >= 2}
    annotators = sorted({a for d in decisions.values() for a in d})
    per_cat = {}
    for c in categories:
        units = [[int(c in labels) for labels in d.values()] for d in multi.values()]
        if not any(any(u) for u in units):
            continue
        pairs = {}
        for x, y in combinations(annotators, 2):
            common = [s for s, d in multi.items() if x in d and y in d]
            if common:
                k = cohen_kappa([int(c in multi[s][x]) for s in common], [int(c in multi[s][y]) for s in common])
                pairs[f"{x}~{y}"] = None if k is None else round(k, 3)
        alpha = krippendorff_alpha_binary(units)
        per_cat[c] = {"items_with_label": sum(1 for u in units if any(u)), "krippendorff_alpha": None if alpha is None else round(alpha, 3),
                      "cohen_kappa": pairs}
    jac = [jaccard(*pair) for d in multi.values() for pair in combinations(list(d.values()), 2)]
    overall_units = [[frozenset(l) for l in d.values()] for d in multi.values()]
    exact = sum(1 for u in overall_units if len(set(u)) == 1)
    return {"annotators": annotators, "items": len(decisions), "items_multiply_annotated": len(multi),
            "exact_set_agreement": round(exact / len(multi), 3) if multi else None,
            "mean_jaccard": round(sum(jac) / len(jac), 3) if jac else None, "per_category": per_cat}
