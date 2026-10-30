"""Statistics of the measurement protocol: median, bootstrap CIs, noise and significance."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np


def _generator(rng: random.Random) -> np.random.Generator:
    return np.random.default_rng(rng.getrandbits(63))


def bootstrap_median(
    values: list[float], resamples: int, level: float, rng: random.Random
) -> tuple[float, float, float]:
    """Median and its percentile bootstrap CI."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("no samples")
    idx = _generator(rng).integers(0, x.size, size=(resamples, x.size))
    meds = np.median(x[idx], axis=1)
    lo, hi = np.quantile(meds, [(1 - level) / 2, (1 + level) / 2])
    return float(np.median(x)), float(lo), float(hi)


def bootstrap_rel_diff(
    a: list[float], b: list[float], resamples: int, level: float, rng: random.Random
) -> tuple[float, float, float]:
    """(median(b) - median(a)) / median(a) and its bootstrap CI (independent resampling)."""
    xa, xb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    gen = _generator(rng)
    ma = np.median(xa[gen.integers(0, xa.size, size=(resamples, xa.size))], axis=1)
    mb = np.median(xb[gen.integers(0, xb.size, size=(resamples, xb.size))], axis=1)
    rel = (mb - ma) / ma
    lo, hi = np.quantile(rel, [(1 - level) / 2, (1 + level) / 2])
    point = (float(np.median(xb)) - float(np.median(xa))) / float(np.median(xa))
    return point, float(lo), float(hi)


@dataclass(frozen=True)
class Verdict:
    significant: bool
    cis_disjoint: bool
    above_floor: bool
    threshold_rel: float


def significance(
    a_ci: tuple[float, float],
    b_ci: tuple[float, float],
    rel_diff: float,
    noise_floor_rel: float,
    factor: float,
) -> Verdict:
    disjoint = a_ci[1] < b_ci[0] or b_ci[1] < a_ci[0]
    threshold = factor * noise_floor_rel
    above = abs(rel_diff) > threshold
    return Verdict(disjoint and above, disjoint, above, threshold)


def noise_summary(medians: list[float]) -> dict[str, Any]:
    """Spread of repeated group medians of one kernel: CV and deviations from their median."""
    x = np.asarray(medians, dtype=np.float64)
    center = float(np.median(x))
    dev = np.abs(x - center) / center
    return {
        "groups": int(x.size),
        "median_of_medians_us": center,
        "mean_us": float(np.mean(x)),
        "std_us": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "cv": float(np.std(x, ddof=1) / np.mean(x)) if x.size > 1 else 0.0,
        "dev_median": float(np.median(dev)),
        "dev_p90": float(np.quantile(dev, 0.9)),
        "dev_max": float(np.max(dev)),
        "min_us": float(np.min(x)),
        "max_us": float(np.max(x)),
    }
