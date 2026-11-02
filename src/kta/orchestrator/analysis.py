"""Bottleneck analysis of a judged round (SYSDEBUG-2, D733).

Judging and analysis are one step: after measuring, every bucket is read in a fixed
order and the result is one report that drives the next round.

1. Model alignment (the gate). The model's predictions must rank and size the measured
   candidates: no unit priced below its speed of light, predictions that are not all
   equal, rank correlation and median error inside the limits. A misaligned bucket says
   why (count error, missing term, rate or composition) and its next step is fixing the
   expression, not optimising the kernel.
2. Bottleneck (aligned buckets). The unit with the largest predicted busy time is the
   bottleneck; its work against the speed of light says whether the gap is inflation
   (more work than needed) or idle time (the unit waits).
3. Verification. The previous round's bottleneck and best time per bucket: progress, a
   moved bottleneck (also progress), or no effect. Directions follow the bottleneck: a
   bottleneck that does not move and does not improve counts as stuck.

Unit busy fractions come from the model's work at peak over the measured time; per-unit
counters (ncu) are not read yet.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

RANK_MIN = 0.6
REL_ERROR_MAX = 0.25
WORK_SLACK = 0.99
NEAR_BOUND = 1.10  # measured within 10% of the bound: the bucket is done
INFLATION = 1.20  # bottleneck work above 1.2x its minimum
IDLE = 0.80  # bottleneck busy below 80% of the measured time
PROGRESS = 0.02  # a best time 2% lower is progress (above the usual noise band)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out

    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if sx == 0 or sy == 0 else cov / (sx * sy)


def _bucket(b: Mapping[str, Any], prev: Mapping[str, Any] | None) -> dict[str, Any]:
    pairs = [(float(p), float(m)) for p, m in b.get("pairs", [])]
    best, bound = float(b["best_us"]), float(b["bound_us"])
    units = [u for u in b.get("units", []) if u.get("predicted_us") is not None]
    preds = [p for p, _ in pairs]
    ties = len(pairs) >= 2 and max(preds) - min(preds) <= 1e-9 * max(abs(max(preds)), 1.0)
    errors = sorted(abs(m / p - 1.0) for p, m in pairs if p > 0)
    median_err = errors[len(errors) // 2] if errors else None
    rho = None if ties else _spearman(preds, [m for _, m in pairs])
    low = [
        u["unit"]
        for u in units
        if u.get("bound_us") and float(u["predicted_us"]) < WORK_SLACK * float(u["bound_us"])
    ]
    meas = [m for _, m in pairs]
    spread_meas = max(meas) / min(meas) if meas and min(meas) > 0 else 1.0
    spread_pred = max(preds) / min(preds) if preds and min(preds) > 0 else 1.0
    misaligned: list[str] = []
    if low:
        misaligned.append("count")
    if ties or (spread_meas > 1.5 and spread_pred < 1.1):
        misaligned.append("missing_term")
    if median_err is not None and median_err > REL_ERROR_MAX and not low:
        misaligned.append("rate_or_composition")
    if rho is not None and rho < RANK_MIN and "missing_term" not in misaligned:
        misaligned.append("missing_term")
    aligned = not misaligned
    top = max(units, key=lambda u: float(u["predicted_us"]), default=None)
    unit = str(top["unit"]) if top else None
    busy = float(top["predicted_us"]) / best if top and best > 0 else None
    sol_u = float(top["bound_us"]) if top and top.get("bound_us") else None
    work_ratio = float(top["predicted_us"]) / sol_u if top and sol_u else None
    if best <= NEAR_BOUND * bound:
        status = "near_bound"
    elif not aligned:
        status = "model"
    elif work_ratio is not None and work_ratio > INFLATION:
        status = "inflation"
    elif busy is not None and busy < IDLE:
        status = "idle"
    else:
        status = "balanced"
    out: dict[str, Any] = {
        "name": b["name"],
        "best_us": best,
        "bound_us": bound,
        "pct_of_bound": 100.0 * bound / best if best > 0 else None,
        "aligned": aligned,
        "misaligned": misaligned,
        "rho": rho,
        "median_rel_error": median_err,
        "predictions_tie": ties,
        "below_sol_units": low,
        "measured_spread": spread_meas,
        "predicted_spread": spread_pred,
        "bottleneck": unit,
        "bottleneck_busy": busy,
        "bottleneck_work_ratio": work_ratio,
        "units": [
            {
                "unit": u["unit"],
                "sol_us": u.get("bound_us"),
                "predicted_us": u["predicted_us"],
                "busy": float(u["predicted_us"]) / best if best > 0 else None,
            }
            for u in units
        ],
        "status": status,
    }
    out["contradiction"] = _contradiction(out)
    out["next"] = _next(out)
    out["verification"] = _verify(out, prev)
    return out


def _pct(x: float | None) -> str:
    return "?" if x is None else f"{100 * x:.0f}%"


def _contradiction(a: Mapping[str, Any]) -> str:
    s, u = a["status"], a["bottleneck"]
    if s == "near_bound":
        return f"best {a['best_us']:.4g} us is within 10% of the bound {a['bound_us']:.4g} us"
    if s == "model":
        why = {
            "count": f"units {', '.join(a['below_sol_units'])} are priced below their speed of "
            "light (counts per tile or per step)",
            "missing_term": f"measured times differ {a['measured_spread']:.2f}x while "
            f"predictions differ {a['predicted_spread']:.2f}x (a resource is missing from the "
            "expression)",
            "rate_or_composition": "predictions are off by a median "
            f"{_pct(a['median_rel_error'])} (a rate or the way terms combine is wrong)",
        }
        return "the model does not match the measurements: " + "; ".join(
            why[k] for k in a["misaligned"]
        )
    if s == "inflation":
        return (
            f"{u} is the bottleneck and does {a['bottleneck_work_ratio']:.2f}x the necessary "
            f"work (busy {_pct(a['bottleneck_busy'])} of the measured time)"
        )
    if s == "idle":
        return (
            f"{u} is the bottleneck but busy only {_pct(a['bottleneck_busy'])} of the measured "
            "time: it waits (latency not hidden, synchronisation or low residency)"
        )
    return f"{u} is the bottleneck, busy {_pct(a['bottleneck_busy'])}, with little excess work"


def _next(a: Mapping[str, Any]) -> str:
    s, u = a["status"], a["bottleneck"]
    if s == "near_bound":
        return "done for this shape"
    if s == "model":
        return (
            "fix the search expression first (instances, missing transfer or storage terms, "
            "rates); do not optimise the kernel against a model that does not match"
        )
    if s == "inflation":
        return f"cut the work on {u}: more reuse per transfer (larger tiles, better layout)"
    if s == "idle":
        return f"keep {u} fed: overlap more (stages, prefetch), raise residency, fewer barriers"
    return f"reduce {u}'s work or move to a technique that uses a less loaded unit"


def _verify(a: Mapping[str, Any], prev: Mapping[str, Any] | None) -> dict[str, Any]:
    if not prev:
        return {"outcome": "first", "stuck": 0}
    improved = a["best_us"] < float(prev["best_us"]) * (1 - PROGRESS)
    moved = a["bottleneck"] != prev.get("bottleneck")
    if improved or moved:
        outcome = "progress" if improved else "bottleneck_moved"
        return {"outcome": outcome, "stuck": 0, "previous_best_us": prev["best_us"],
                "previous_bottleneck": prev.get("bottleneck")}  # fmt: skip
    return {
        "outcome": "no_effect",
        "stuck": int(prev.get("verification", {}).get("stuck", 0)) + 1,
        "previous_best_us": prev["best_us"],
        "previous_bottleneck": prev.get("bottleneck"),
    }


def analyze(
    buckets: Sequence[Mapping[str, Any]], previous: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The round's analysis: one entry per measured bucket and the report text."""
    before = {str(b["name"]): b for b in (previous or {}).get("buckets", [])}
    rows = [_bucket(b, before.get(str(b["name"]))) for b in buckets if b.get("pairs")]
    return {"buckets": rows, "text": report(rows)}


def report(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["## Analysis of the last round", ""]
    for a in rows:
        v = a["verification"]
        lines.append(
            f"- {a['name']}: best {a['best_us']:.4g} us, bound {a['bound_us']:.4g} us "
            f"({a['pct_of_bound']:.0f}% of bound); model "
            + ("aligned" if a["aligned"] else f"misaligned ({', '.join(a['misaligned'])})")
        )
        lines.append(f"  - main contradiction: {a['contradiction']}")
        lines.append(f"  - next: {a['next']}")
        if v["outcome"] != "first":
            lines.append(
                f"  - against the last round: {v['outcome']} (was {v['previous_best_us']:.4g} us"
                f" on {v['previous_bottleneck']})"
                + (f"; stuck {v['stuck']} rounds on this bottleneck" if v["stuck"] else "")
            )
    return "\n".join(lines)
