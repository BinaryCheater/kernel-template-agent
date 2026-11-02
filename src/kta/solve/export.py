"""The exported MIP, checked against direct pricing and shown as text (SYSDEBUG-2, D740).

The projection compiler (``projection.compile_and_solve``) turns the formulation into a
MIP. That translation is the system's: the agent writes the formulation, never the MIP.
``check_export`` compiles it for one shape and checks it against ``projection_us``, the
same L1 projection evaluated directly at a point:

1. every point the MIP returns prices to the MIP's objective there (translation);
2. no point of the same menus prices below the MIP's optimum (the MIP is a minimum);
3. when the MIP says infeasible, no point of the menus prices at all.

Points for 2 and 3 come from a bounded enumeration of the plan's menus. A failure is the
compiler's fault (``MipMismatch``): the round stops instead of searching a wrong problem.
``cost_terms`` and ``export_text`` say what the search expression is at the best point:
per resource, each transaction's count and amounts and the time they cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from kta.core.errors import KtaError
from kta.core.expr import ExprError
from kta.formulation import Formulation, evaluate_effects
from kta.models import MachineModel
from kta.models.l1 import check_walls, compose, price_parts
from kta.solve.enumerate import enumerate_space
from kta.solve.mip import NotExportable
from kta.solve.projection import compile_and_solve, menus_of, projection_us

CHECK_POINTS = 2000  # directly priced points per shape
VISIT_MAX = 50_000  # enumeration nodes when collecting them
SHOWN = 3  # best MIP points kept per shape
REL_TOL = 1e-6


class MipMismatch(KtaError):
    """The compiled MIP disagrees with direct pricing of the same formulation."""


@dataclass
class ExportCheck:
    shape: dict[str, Any]
    status: str  # MIP status, or "not exported"
    note: str = ""
    best: list[dict[str, Any]] = field(default_factory=lambda: list[dict[str, Any]]())
    errors: list[str] = field(default_factory=lambda: list[str]())
    checked: int = 0  # points priced directly
    exhaustive: bool = False
    outer: list[str] = field(default_factory=lambda: list[str]())
    rules: dict[str, int] = field(default_factory=lambda: dict[str, int]())
    terms: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "status": self.status,
            "note": self.note,
            "best": self.best,
            "errors": self.errors,
            "checked": self.checked,
            "exhaustive": self.exhaustive,
            "outer": self.outer,
            "rules": self.rules,
            "terms": self.terms,
        }


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(1.0, abs(a), abs(b))


def _direct(f: Formulation, a: Mapping[str, Any], shape: Mapping[str, Any], mm: MachineModel):
    try:
        return projection_us(f, a, shape, mm)
    except ExprError:
        return None  # undefined at this point (the MIP excludes it too)


def _menu_points(
    f: Formulation, shape: Mapping[str, Any], mm: MachineModel, inputs: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], bool]:
    """Feasible points of the plan's menus, spread over the enumeration order."""
    menus = menus_of(f, shape, mm.hw, inputs.get("fixed"), inputs.get("menus"))
    sub = replace(
        f,
        variables={
            x: v.model_copy(update={"domain": list(menus[x])}) for x, v in f.variables.items()
        },
    )
    found: list[dict[str, Any]] = []
    stats = enumerate_space(sub, shape, mm.hw, found.append, max_nodes=VISIT_MAX)
    step = max(1, len(found) // CHECK_POINTS)
    return found[::step][:CHECK_POINTS], stats.exhaustive and step == 1


def check_export(
    f: Formulation,
    shape: Mapping[str, Any],
    mm: MachineModel,
    inputs: Mapping[str, Any] | None = None,
    time_limit_s: float = 60.0,
) -> ExportCheck:
    """Compile ``f`` on ``shape`` with a plan's ``fixed``/``menus``/``outer`` and check the
    MIP against direct pricing; ``errors`` lists every disagreement."""
    inputs = dict(inputs or {})
    out = ExportCheck(dict(shape), "not exported")
    try:
        res = compile_and_solve(f, shape, mm, shortlist=SHOWN, time_limit_s=time_limit_s, **inputs)
    except NotExportable as exc:
        out.note = str(exc)  # the round enumerates instead; nothing to compare
        return out
    out.status, out.outer, out.rules = res.status, list(res.outer), dict(res.rules)
    for a, obj in zip(res.assignments, res.objective_us, strict=True):
        d = _direct(f, a, shape, mm)
        out.best.append({"assignment": a, "mip_us": obj, "direct_us": d})
        if d is None:
            out.errors.append(f"the MIP picks {a}, which direct pricing excludes")
        elif not _close(obj, d):
            out.errors.append(f"at {a} the MIP objective is {obj:.6g} us, direct {d:.6g} us")
    points, out.exhaustive = _menu_points(f, shape, mm, inputs)
    low: tuple[float, dict[str, Any]] | None = None
    for a in points:
        d = _direct(f, a, shape, mm)
        if d is None:
            continue
        out.checked += 1
        if low is None or d < low[0]:
            low = (d, a)
    if not res.assignments and low is not None:
        out.errors.append(
            f"the MIP is {res.status} but direct pricing prices {out.checked} points, "
            f"e.g. {low[1]} at {low[0]:.6g} us"
        )
    if res.assignments and res.first_optimal and low is not None:
        if low[0] < res.objective_us[0] and not _close(low[0], res.objective_us[0]):
            out.errors.append(
                f"{low[1]} prices {low[0]:.6g} us, below the MIP optimum "
                f"{res.objective_us[0]:.6g} us"
            )
    if res.assignments:
        out.terms = cost_terms(f, res.assignments[0], shape, mm)
    return out


def cost_terms(
    f: Formulation, a: Mapping[str, Any], shape: Mapping[str, Any], mm: MachineModel
) -> dict[str, Any]:
    """The search expression at ``a``: per resource, each transaction's count expression,
    amounts, values and time at peak, then the bounds the total is the max of."""
    effects = evaluate_effects(f, a, shape, mm.hw)
    walls = check_walls(effects.storage, mm)
    resident = [w.residency for w in walls if w.residency is not None]
    parts, unpriced, _, per_tx = price_parts(effects, mm, min(resident) if resident else None)
    composed = compose(parts)
    bound = {t.key: t for t in f.transactions}
    evald = {t.key: t for t in effects.txs}
    rows: list[dict[str, Any]] = []
    for key, kind, times in per_tx:
        bt, et = bound[key], evald[key]
        for res, us in sorted(times.items()):
            rows.append(
                {
                    "resource": res,
                    "item": bt.source,
                    "tx": bt.id,
                    "kind": kind,
                    "count": bt.count.src,
                    "count_value": et.count,
                    "amounts": {k: v.src for k, v in bt.amounts.items()},
                    "amount_values": dict(et.amounts),
                    "us": us,
                }
            )
    return {
        "assignment": dict(a),
        "rows": rows,
        "busy_us": dict(composed.busy_us),
        "bounds_us": dict(composed.bounds_us),
        "serial_us": composed.additive_us,
        "total_us": composed.total_us,
        "bottleneck": composed.bottleneck,
        "residency": min(resident) if resident else None,
        "unpriced": dict(unpriced),
    }


def _g(x: Any) -> str:
    return f"{x:.4g}" if isinstance(x, float) else str(x)


def _eq(src: str, value: Any) -> str:
    """``src = value``, or the value alone when the source is that number already."""
    v = _g(value)
    try:
        same = float(src) == float(value)
    except (TypeError, ValueError):
        same = False
    return v if same else f"{src} = {v}"


def terms_text(t: Mapping[str, Any]) -> list[str]:
    lines = [f"  cost at {t['assignment']}:"]
    by_res: dict[str, list[Mapping[str, Any]]] = {}
    for r in t["rows"]:
        by_res.setdefault(str(r["resource"]), []).append(r)
    for res, rows in sorted(by_res.items(), key=lambda kv: -sum(r["us"] for r in kv[1])):
        lines.append(f"    {res}: {sum(r['us'] for r in rows):.4g} us")
        for r in sorted(rows, key=lambda r: -r["us"]):
            amounts = "".join(
                f"; {k} = {_eq(v, r['amount_values'].get(k))}" for k, v in r["amounts"].items()
            )
            lines.append(
                f"      {r['item']}.{r['tx']} ({r['kind']}): count = "
                f"{_eq(r['count'], r['count_value'])}{amounts} -> {r['us']:.4g} us"
            )
    for name, us in sorted(t["bounds_us"].items(), key=lambda kv: -kv[1]):
        lines.append(f"    {name}: {us:.4g} us")
    if t["serial_us"]:
        lines.append(f"    serial (added): {t['serial_us']:.4g} us")
    if t["unpriced"]:
        lines.append(f"    unpriced classes (no time): {t['unpriced']}")
    lines.append(
        f"    total = max(resources, bounds) + serial = {t['total_us']:.4g} us; "
        f"bottleneck: {t['bottleneck']}"
        + (f"; {t['residency']} CTAs per SM fit" if t["residency"] else "")
    )
    return lines


def export_text(checks: list[ExportCheck]) -> str:
    """What the exported MIP is and finds, per shape, for the agent and the case page."""
    lines: list[str] = []
    for c in checks:
        lines.append(f"shape {c.shape}: MIP {c.status}" + (f" ({c.note})" if c.note else ""))
        if c.outer:
            lines.append(f"  enumerated outside the MIP: {c.outer}")
        lines.append(
            f"  checked against direct pricing at {c.checked} points"
            + (" (all points of the menus)" if c.exhaustive else "")
            + (": agrees" if not c.errors else ": DISAGREES (a system error, not the family's)")
        )
        lines += [f"  ! {e}" for e in c.errors]
        for i, b in enumerate(c.best):
            lines.append(f"  best {i + 1}: {b['mip_us']:.4g} us at {b['assignment']}")
        if c.terms:
            lines += terms_text(c.terms)
    return "\n".join(lines)
