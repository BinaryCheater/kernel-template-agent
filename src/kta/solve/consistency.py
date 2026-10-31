"""The MIP <-> L2 consistency check (WP17 in WP19; docs/design/02 §5.2, D-T4; D551).

Translation correctness is the compiler's (the MIP objective equals ``projection_us`` at
every point); what remains is the simplification error of the projection against the
simulator. Without a GPU: sample feasible points of the subspace, evaluate the projection
and L2 (``kta.models.l2.simulate``), and require

- Spearman rho(MIP, L2) >= ``rank_min`` (knob ``model.mip_l2_check``), and
- MIP <= L2 (the optimistic bound) on all but a share ``violation_max`` of the points.

On failure the result names the effect with the largest deviation: the transaction
whose time differs most between the projection's and the simulator's pricing (summed
over the sample, e.g. a reservation table out of step), else, when every transaction
agrees, the simulator's idle category on its bottleneck that is largest where L2 exceeds
the projection most ("schedule:<resource>:<category>").

``l2_mm`` is the simulator's model; it defaults to the projection's own. The two differ
when the projection was compiled against another model version than the simulator runs.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from kta.formulation import Formulation, evaluate_effects
from kta.models import MachineModel
from kta.models.l1 import price_parts
from kta.models.l2 import simulate
from kta.models.regression import spearman
from kta.solve.enumerate import enumerate_space
from kta.solve.projection import projection_us

RANK_MIN = 0.5  # defaults of knob model.mip_l2_check (set_by X11)
VIOLATION_MAX = 0.05
SAMPLES = 128
REL_TOL = 1e-9  # MIP <= L2 up to rounding
MAX_NODES = 200_000  # enumeration budget for collecting feasible points


@dataclass
class ConsistencyResult:
    points: int
    rho: float | None
    violations: int  # points with MIP > L2
    violation_share: float
    passed: bool
    worst_effect: str | None  # set when the check fails
    deviation_us: dict[str, float] = field(default_factory=lambda: dict[str, float]())
    ratio_min: float | None = None  # L2 / MIP
    ratio_max: float | None = None
    notes: list[str] = field(default_factory=lambda: list[str]())

    def summary(self) -> dict[str, Any]:
        top = dict(sorted(self.deviation_us.items(), key=lambda kv: -kv[1])[:5])
        return {
            "points": self.points, "rho": self.rho, "violations": self.violations,
            "violation_share": self.violation_share, "passed": self.passed,
            "worst_effect": self.worst_effect, "deviation_us": top,
            "ratio_min": self.ratio_min, "ratio_max": self.ratio_max, "notes": self.notes,
        }  # fmt: skip


def _tx_times(
    f: Formulation, a: Mapping[str, Any], shape: Mapping[str, Any], mm: MachineModel
) -> dict[str, float]:
    per_tx = price_parts(evaluate_effects(f, a, shape, mm.hw), mm)[3]
    return {key: sum(times.values()) for key, _, times in per_tx}


def consistency_check(
    f: Formulation,
    shape: Mapping[str, Any],
    mm: MachineModel,
    *,
    l2_mm: MachineModel | None = None,
    rank_min: float = RANK_MIN,
    violation_max: float = VIOLATION_MAX,
    samples: int = SAMPLES,
    seed: int = 0,
) -> ConsistencyResult:
    """Check the projection of ``f`` (on ``mm``) against L2 (on ``l2_mm``) at ``shape``."""
    sim = l2_mm or mm
    found: list[dict[str, Any]] = []
    stats = enumerate_space(f, shape, mm.hw, found.append, max_nodes=MAX_NODES)
    notes = [] if stats.exhaustive else [f"enumeration budget {MAX_NODES} spent"]
    if len(found) > samples:
        found = random.Random(seed).sample(found, samples)
        notes.append(f"{samples} of {stats.feasible} feasible points sampled (seed {seed})")
    mip: list[float] = []
    l2: list[float] = []
    deviation: dict[str, float] = {}
    idle_at: list[tuple[float, str]] = []
    for a in found:
        r = simulate(evaluate_effects(f, a, shape, sim.hw), sim)
        if not r.feasible:
            continue
        p = projection_us(f, a, shape, mm)
        mip.append(p)
        l2.append(r.total_us)
        ours, theirs = _tx_times(f, a, shape, mm), _tx_times(f, a, shape, sim)
        for key in ours.keys() | theirs.keys():
            deviation[key] = deviation.get(key, 0.0) + abs(
                ours.get(key, 0.0) - theirs.get(key, 0.0)
            )
        idle = r.idle_us.get(r.bottleneck, {})
        if idle:
            cat = max(idle, key=lambda c: idle[c])
            idle_at.append((r.total_us / p if p else 0.0, f"schedule:{r.bottleneck}:{cat}"))
    if len(mip) < len(found):
        notes.append(f"{len(found) - len(mip)} points infeasible in L2 (no-launch wall)")
    n = len(mip)
    violations = sum(1 for x, y in zip(mip, l2, strict=True) if x > y * (1 + REL_TOL))
    share = violations / n if n else 0.0
    rho = spearman(mip, l2)
    passed = n >= 2 and rho is not None and rho >= rank_min and share <= violation_max
    worst = None
    if not passed:
        if deviation and max(deviation.values()) > 1e-9:
            worst = max(deviation, key=lambda k: (deviation[k], k))
        elif idle_at:
            worst = max(idle_at)[1]
        if rho is None:
            notes.append("rank correlation undefined (fewer than two points or a constant side)")
    ratios = [y / x for x, y in zip(mip, l2, strict=True) if x > 0]
    return ConsistencyResult(
        points=n, rho=rho, violations=violations, violation_share=share, passed=passed,
        worst_effect=worst, deviation_us=deviation,
        ratio_min=min(ratios, default=None), ratio_max=max(ratios, default=None), notes=notes,
    )  # fmt: skip
