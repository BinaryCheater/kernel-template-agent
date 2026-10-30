"""Fitting the per-unit model to measurements (decision D244).

Loss: mean squared relative error. eff_u is searched per unit by coordinate descent on
log eff; for fixed eff, serial_coef and serial_const_us are a non-negative weighted
least-squares fit. An interaction gamma_uv * min(t_u, t_v) is added only when it passes
an F test (F > 4, at least 3 residual degrees of freedom, >= 10% lower loss).
Every fifth point (index % 5 == 4) is held out when there are at least five points; the
returned model is fitted on the rest and its hold-out error is reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from kta.core.errors import SchemaError
from kta.core.schemas.evaluation import Measurement, Prediction

if TYPE_CHECKING:
    from kta.models.unit_model import Interaction, UnitModel

EFF_MIN, EFF_MAX = 0.02, 2.0
SEARCH_ROUNDS = 60
F_THRESHOLD = 4.0
MIN_GAIN = 0.10
MIN_RESIDUAL_DOF = 3
HOLDOUT_EVERY = 5

Arr = NDArray[np.float64]


@dataclass(frozen=True)
class FitReport:
    eff: dict[str, float]
    serial_coef: float
    serial_const_us: float
    interactions: tuple[Interaction, ...]
    info: dict[str, Any]


@dataclass(frozen=True)
class _Data:
    units: list[str]
    t: Arr  # (n, units) peak times
    s: Arr  # (n,) serial peak times
    y: Arr  # (n,) measured medians


def _data(points: list[tuple[Prediction, Measurement]]) -> _Data:
    units = sorted({u for p, _ in points for u, t in p.per_unit.items() if t > 0})
    t = np.array([[p.per_unit.get(u, 0.0) for u in units] for p, _ in points], dtype=float)
    s = np.array([p.serial for p, _ in points], dtype=float)
    y = np.array([m.median_us for _, m in points], dtype=float)
    if np.any(y <= 0):
        raise SchemaError("measurements must have positive median_us")
    return _Data(units, t.reshape(len(points), len(units)), s, y)


def _times(d: _Data, log_eff: Arr) -> Arr:
    return d.t / np.exp(log_eff)[None, :]


def _nnls(cols: list[Arr], target: Arr, w: Arr) -> Arr:
    """Weighted least squares with the coefficients clamped at zero (few columns)."""
    active = list(range(len(cols)))
    coef = np.zeros(len(cols))
    while active:
        a = np.stack([cols[i] * w for i in active], axis=1)
        sol, *_ = np.linalg.lstsq(a, target * w, rcond=None)
        if np.all(sol >= 0):
            coef[active] = sol
            return coef
        active = [i for i, c in zip(active, sol, strict=True) if c > 0]
    return coef


def _evaluate(d: _Data, log_eff: Arr, pairs: list[tuple[int, int]]) -> tuple[float, Arr]:
    """Loss and the linear coefficients [serial_coef, const, gamma...] for fixed eff."""
    times = _times(d, log_eff)
    base = times.max(axis=1) if d.units else np.zeros(len(d.y))
    cols = [d.s, np.ones(len(d.y))] + [np.minimum(times[:, i], times[:, j]) for i, j in pairs]
    coef = _nnls(cols, d.y - base, 1.0 / d.y)
    pred = base + sum(c * col for c, col in zip(coef, cols, strict=True))
    return float(np.mean(((pred - d.y) / d.y) ** 2)), coef


def _search(d: _Data, start: Arr, pairs: list[tuple[int, int]]) -> tuple[Arr, float, Arr]:
    log_eff = np.clip(start.copy(), math.log(EFF_MIN), math.log(EFF_MAX))
    loss, coef = _evaluate(d, log_eff, pairs)
    step = math.log(2.0)
    for _ in range(SEARCH_ROUNDS):
        improved = False
        for k in range(len(d.units)):
            for delta in (step, -step):
                trial = log_eff.copy()
                trial[k] = min(max(trial[k] + delta, math.log(EFF_MIN)), math.log(EFF_MAX))
                trial_loss, trial_coef = _evaluate(d, trial, pairs)
                if trial_loss < loss - 1e-15:
                    log_eff, loss, coef, improved = trial, trial_loss, trial_coef, True
        if not improved:
            step /= 2
            if step < 1e-3:
                break
    return log_eff, loss, coef


def _interactions(
    d: _Data, log_eff: Arr, loss: float, coef: Arr
) -> tuple[list[tuple[int, int]], Arr, Arr, float, list[str]]:
    """Greedily add significant pair terms; returns pairs, eff, coef, loss and the basis."""
    pairs: list[tuple[int, int]] = []
    basis: list[str] = []
    n = len(d.y)
    candidates = [(i, j) for i in range(len(d.units)) for j in range(i + 1, len(d.units))]
    while candidates:
        k = len(d.units) + 2 + len(pairs) + 1
        dof = n - k
        if dof < MIN_RESIDUAL_DOF:
            break
        best: tuple[float, tuple[int, int], Arr, float, Arr] | None = None
        for pair in candidates:
            le, new_loss, new_coef = _search(d, log_eff, [*pairs, pair])
            f_stat = (loss - new_loss) / max(new_loss / dof, 1e-30)
            if best is None or f_stat > best[0]:
                best = (f_stat, pair, le, new_loss, new_coef)
        assert best is not None
        f_stat, pair, le, new_loss, new_coef = best
        if f_stat <= F_THRESHOLD or new_loss > (1 - MIN_GAIN) * loss:
            break
        basis.append(
            f"F={f_stat:.2f} > {F_THRESHOLD}, loss {loss:.4g} -> {new_loss:.4g}, n={n}, dof={dof}"
        )
        pairs.append(pair)
        candidates.remove(pair)
        log_eff, loss, coef = le, new_loss, new_coef
    return pairs, log_eff, coef, loss, basis


def errors(pred: list[float], meas: list[float]) -> dict[str, float | int | None]:
    """Median absolute relative error and pairwise ordering accuracy."""
    if not pred:
        return {"n": 0, "median_abs_rel_err": None, "ordering_accuracy": None}
    rel = sorted(abs(p - m) / m for p, m in zip(pred, meas, strict=True))
    agree = total = 0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            if meas[i] == meas[j]:
                continue
            total += 1
            agree += (pred[i] - pred[j]) * (meas[i] - meas[j]) > 0
    return {
        "n": len(pred),
        "median_abs_rel_err": float(np.median(rel)),
        "ordering_accuracy": agree / total if total else None,
    }


def fit_model(model: UnitModel, points: list[tuple[Prediction, Measurement]]) -> FitReport:
    from kta.models.unit_model import Interaction, UnitModel

    if not points:
        raise SchemaError("fit needs at least one (Prediction, Measurement) pair")
    holdout_idx = [i for i in range(len(points)) if len(points) >= HOLDOUT_EVERY and i % 5 == 4]
    train = [p for i, p in enumerate(points) if i not in holdout_idx]
    d = _data(train)
    start = np.array([math.log(model.eff.get(u, 1.0)) for u in d.units], dtype=float)
    log_eff, loss, coef = _search(d, start, [])
    pairs, log_eff, coef, loss, basis = _interactions(d, log_eff, loss, coef)
    eff = {**model.eff, **{u: float(math.exp(v)) for u, v in zip(d.units, log_eff, strict=True)}}
    inter = tuple(
        Interaction((d.units[i], d.units[j]), float(coef[2 + k]), basis[k])
        for k, (i, j) in enumerate(pairs)
    )
    fitted = FitReport(eff, float(coef[0]), float(coef[1]), inter, {})
    probe = UnitModel(
        target=model.target,
        eff=eff,
        serial_coef=fitted.serial_coef,
        serial_const_us=fitted.serial_const_us,
        interactions=inter,
        occupancy_knee=model.occupancy_knee,
    )

    def scores(sel: list[tuple[Prediction, Measurement]]) -> dict[str, float | int | None]:
        return errors(
            [probe.total_us(p.per_unit, p.serial) for p, _ in sel], [m.median_us for _, m in sel]
        )

    info: dict[str, Any] = {
        "loss": loss,
        "train": scores(train),
        "holdout": scores([points[i] for i in holdout_idx]),
        "holdout_indices": holdout_idx,
        "units": d.units,
        "rule": "D244",
    }
    return FitReport(eff, fitted.serial_coef, fitted.serial_const_us, inter, info)
