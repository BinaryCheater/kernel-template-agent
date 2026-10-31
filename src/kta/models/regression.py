"""Regression set and scoreboard of the hardware model (docs/design/02 §4 "Maintenance as a
software artefact"; D507–D510).

Every measured kernel becomes a regression case with several configurations, because what
the solver relies on is ranking. A model version is scored per case: Spearman rank
correlation of predicted against measured time, the median relative error of the total,
and the median relative error of each resource's busy time (measured busy = the resource's
ncu ``.pct_of_peak_sustained_elapsed`` times the measured duration). The scoreboard lists
ModelVersion records with these numbers; admission compares the hold-out cases with the
``model.version_admission`` knob.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kta.core.errors import NotSupported
from kta.core.ids import record_id
from kta.core.schemas import Actor, Record
from kta.core.schemas.common import Origin
from kta.core.schemas.evaluation import Measurement, Profile
from kta.core.schemas.hwmodel import (
    ModelVersion,
    RegressionCase,
    RegressionResult,
    RegressionSet,
)
from kta.models.l2 import L2Result
from kta.models.machine import MachineModel
from kta.store.base import Store

PCT = ".pct_of_peak_sustained_elapsed"


@dataclass(frozen=True)
class Observation:
    kernel: str
    configuration: Mapping[str, Any]
    predicted_us: float
    measured_us: float
    predicted_busy_us: Mapping[str, float] = field(default_factory=lambda: dict[str, float]())
    measured_busy_us: Mapping[str, float] = field(default_factory=lambda: dict[str, float]())
    holdout: bool = False
    prediction: str | None = None  # Report ID of the predicted timeline (l2_simulation)
    measurement: str | None = None


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1  # ties share the average rank
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation; None with fewer than two points or a constant side."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sx = sum((a - mx) ** 2 for a in rx)
    sy = sum((b - my) ** 2 for b in ry)
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True)) / (sx * sy) ** 0.5


def _rel(p: float, m: float) -> float:
    return abs(p - m) / m


def results(observations: Iterable[Observation]) -> list[RegressionResult]:
    """One RegressionResult per kernel, in kernel order."""
    by: dict[str, list[Observation]] = {}
    for o in observations:
        by.setdefault(o.kernel, []).append(o)
    out: list[RegressionResult] = []
    for kernel in sorted(by):
        obs = by[kernel]
        errors: dict[str, float] = {
            "total": statistics.median(_rel(o.predicted_us, o.measured_us) for o in obs)
        }
        for r in sorted({r for o in obs for r in o.measured_busy_us}):
            pairs = [
                (o.predicted_busy_us.get(r, 0.0), o.measured_busy_us[r])
                for o in obs
                if o.measured_busy_us.get(r, 0.0) > 0
            ]
            if pairs:
                errors[r] = statistics.median(_rel(p, m) for p, m in pairs)
        out.append(
            RegressionResult(
                kernel=kernel,
                predicted_us=[o.predicted_us for o in obs],
                measured_us=[o.measured_us for o in obs],
                rank_correlation=spearman(
                    [o.predicted_us for o in obs], [o.measured_us for o in obs]
                ),
                error_by_resource=errors,
                holdout=any(o.holdout for o in obs),
                predictions=[o.prediction or "" for o in obs],
                measurements=[o.measurement or "" for o in obs],
            )
        )
    return out


def summarise(res: Sequence[RegressionResult]) -> dict[str, float]:
    """Mean rank correlation and median errors, over all cases and over hold-out cases."""
    out: dict[str, float] = {}
    for prefix, rows in (("", res), ("holdout_", [r for r in res if r.holdout])):
        rho = [r.rank_correlation for r in rows if r.rank_correlation is not None]
        if rho:
            out[f"{prefix}rank_correlation_mean"] = statistics.fmean(rho)
        for key in sorted({k for r in rows for k in r.error_by_resource}):
            errs = [r.error_by_resource[key] for r in rows if key in r.error_by_resource]
            out[f"{prefix}error_median:{key}"] = statistics.median(errs)
    return out


def admit(summary: Mapping[str, float], thresholds: Mapping[str, float]) -> bool | None:
    """Hold-out decision; None when no hold-out case was scored (D509)."""
    rho = summary.get("holdout_rank_correlation_mean")
    err = summary.get("holdout_error_median:total")
    if rho is None or err is None:
        return None
    return rho >= thresholds["rank_min"] and err <= thresholds["rel_error_max"]


# ---- from the store -------------------------------------------------------------------


Predict = Callable[[RegressionCase, Mapping[str, Any]], tuple[L2Result, str | None]]


def measured_busy(profile: Profile, total_us: float, mm: MachineModel) -> dict[str, float]:
    """Measured busy time per resource from its first ncu counter's pct-of-peak metric."""
    out: dict[str, float] = {}
    for name, res in mm.resources.items():
        if res.ncu and f"{res.ncu[0]}{PCT}" in profile.metrics:
            out[name] = profile.metrics[f"{res.ncu[0]}{PCT}"] / 100.0 * total_us
    return out


def _profile_of(store: Store, mid: str) -> Profile | None:
    build = store.get(mid).refs.get("build")
    if not isinstance(build, str):
        return None
    recs = [store.get(i) for i in store.neighbors(build, rel="build", direction="in")]
    profiles = sorted((r for r in recs if r.kind == "Profile"), key=lambda r: r.created_at)
    return Profile.model_validate(profiles[-1].body) if profiles else None


def observe(store: Store, set_id: str, mm: MachineModel, predict: Predict) -> list[Observation]:
    """Observations for every measured configuration of a stored RegressionSet."""
    rs = RegressionSet.model_validate(store.get(set_id).body)
    out: list[Observation] = []
    for case in rs.cases:
        if not case.measurements:
            raise NotSupported(f"regression case {case.kernel}: no measurements yet")
        for config, mid in zip(case.configurations, case.measurements, strict=True):
            m = Measurement.model_validate(store.get(mid).body)
            l2, report = predict(case, config)
            profile = _profile_of(store, mid)
            out.append(
                Observation(
                    kernel=case.kernel,
                    configuration=config,
                    predicted_us=l2.total_us,
                    measured_us=m.median_us,
                    predicted_busy_us=l2.busy_us,
                    measured_busy_us=measured_busy(profile, m.median_us, mm) if profile else {},
                    holdout=case.holdout,
                    prediction=report,
                    measurement=mid,
                )
            )
    return out


def record_version(
    store: Store,
    body: ModelVersion,
    actor: Actor,
    now: datetime,
    refs: Mapping[str, str],
    origin: Origin = "real",
) -> str:
    return store.put(Record.new("ModelVersion", body, actor, now, refs=dict(refs), origin=origin))


def scoreboard(store: Store, target: str | None = None) -> list[dict[str, Any]]:
    """Every ModelVersion with its regression numbers, oldest first."""
    rows = []
    where = {"target": target} if target else None
    found = store.query("ModelVersion", where, order="created_at", limit=10_000)
    for rec in found:
        mv = ModelVersion.model_validate(rec.body)
        rows.append(
            {
                "id": record_id(rec),
                "label": mv.label,
                "target": mv.target,
                "hw_model": mv.hw_model,
                "arch": mv.arch,
                "simulators": mv.simulators,
                "change": mv.change,
                "reason": mv.reason,
                "admitted": mv.admitted,
                "summary": mv.summary,
                "results": [r.model_dump() for r in mv.results],
                "origin": rec.origin,
                "created_at": rec.created_at.isoformat(),
                "previous": rec.refs.get("previous"),
            }
        )
    return rows
