"""Interval estimates and data-quality summaries.

Confidence intervals assume the analysed repositories/methods are a simple
random sample of the population of interest. The corpus is not such a sample
(GitHub search coverage, filters, stratification), so intervals describe
sampling variability only, not selection bias — reports state this explicitly.
"""
from __future__ import annotations

import math
import random
from statistics import NormalDist


def z_value(level: float) -> float:
    return NormalDist().inv_cdf(0.5 + level / 2)


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    z = z_value(level)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean(values: list[float], iterations: int, seed: int, level: float = 0.95) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(iterations))
    lo = means[int((1 - level) / 2 * iterations)]
    hi = means[min(iterations - 1, int((1 + level) / 2 * iterations))]
    return lo, hi
