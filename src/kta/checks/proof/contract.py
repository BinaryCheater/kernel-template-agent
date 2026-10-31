"""The contract conditions every Template has (paper §2.2): K1 and K3, plus the sanity of
the outline's environment assumptions.

- K1: for all p in Γ and x in X, C(p, x) implies each Pre. A ``same_as`` link is reported
  as confirmed only when the checker shows the linked constraint and the precondition are
  equivalent on the space; the link itself is never taken as evidence.
- K3: for every p in Γ some x in X satisfies C.
- Assumptions may name only p.*, hw.* and case.*; one naming a design variable would turn
  a design statement into an assumption and is refused. Γ together with the assumptions
  must be satisfiable, or every proof would be vacuous.
"""

from __future__ import annotations

import ast
from typing import Any

import z3

from kta.core.schemas.knowledge import Template
from kta.core.schemas.proof import Assumption

from . import interp
from .encode import Unsupported, parse
from .smt import run
from .space import Space
from .types import Draft

K3_MAX_DISJUNCTION = 2048  # design points expanded inline in the K3 query
K3_MAX_REPLAY = 200_000  # design points the interpreter may enumerate to confirm a K3 witness


def assumption_issues(t: Template, assumes: list[Assumption]) -> list[str]:
    design = {v.name for v in t.variables}
    issues: list[str] = []
    for a in assumes:
        if not a.expr:
            continue
        try:
            names = {n.id for n in ast.walk(parse(a.expr)) if isinstance(n, ast.Name)}
        except Unsupported as exc:
            issues.append(f"assumption {a.name}: {exc}")
            continue
        if names & design:
            issues.append(
                f"assumption {a.name} names design variables {sorted(names & design)}: "
                "an unproved design statement cannot become an assumption"
            )
    return issues


def vacuity(sp: Space, timeout_ms: int) -> Draft | None:
    """Γ ∧ assumptions ∧ X must have a point; otherwise every proof is vacuous."""
    q = run(z3.And(sp.domain, sp.gamma(), sp.assumptions()), {}, timeout_ms)
    if q.result == "sat":
        return None
    d = Draft("K3", "*", "unresolved", sp.describe(), groups=("gamma", "assumes", "X"))
    d.add(q.record())
    d.reason = (
        "Gamma with the outline's assumptions is empty: any proof would be vacuous"
        if q.result == "unsat"
        else f"vacuity check unknown: {q.reason}"
    )
    return d


def _witness(sp: Space, d: Draft, model: dict[str, Any], must: dict[str, bool]) -> None:
    point = sp.decode(model)
    agrees, values = interp.confirm(point, must)
    d.replay = {"interpreter": "kta.core.expr", "agrees": agrees, **values}
    if agrees:
        d.outcome, d.counterexample = "counterexample", point
    else:
        d.outcome = "unresolved"
        d.reason = "solver witness not confirmed by the independent interpreter"
        d.data["witness"] = point


def k1(sp: Space, timeout_ms: int, skip: set[tuple[str, str]] | None = None) -> list[Draft]:
    t = sp.template
    out: list[Draft] = []
    skip = skip or set()
    base = z3.And(sp.domain, sp.gamma(), sp.assumptions(), sp.constraints())
    by_id = {c.id: c for c in t.constraints}
    for pre in t.preconditions:
        if ("K1", pre.id) in skip:
            continue
        d = Draft(
            "K1",
            pre.id,
            "proved",
            "all " + sp.describe(),
            groups=("gamma", "assumes", "X", "C", f"pre:{pre.id}"),
        )
        try:
            q = run(z3.And(base, z3.Not(sp.tr.bool(pre.expr))), sp.decls(), timeout_ms)
        except Unsupported as exc:
            d.outcome, d.reason = "unresolved", f"unsupported construct: {exc}"
            out.append(d)
            continue
        d.add(q.record())
        if q.result == "unknown":
            d.outcome, d.reason = "unresolved", f"solver unknown: {q.reason}"
        elif q.result == "sat":
            must = {t.scope.predicate: True, pre.expr: False}
            must.update({c.expr: True for c in t.constraints})
            _witness(sp, d, q.model, must)
        if pre.same_as:
            d.data["same_as"] = _same_as(sp, pre.expr, by_id.get(pre.same_as), d, timeout_ms)
        out.append(d)
    return out


def _same_as(sp: Space, pre: str, c: Any, d: Draft, timeout_ms: int) -> dict[str, Any]:
    if c is None:
        return {"link": "missing constraint", "confirmed": False}
    q = run(
        z3.And(sp.domain, sp.gamma(), sp.assumptions(), sp.tr.bool(pre) != sp.tr.bool(c.expr)),
        sp.decls(),
        timeout_ms,
    )
    d.add(q.record())
    return {"constraint": c.id, "confirmed": q.result == "unsat", "solver": q.result}


def k3(sp: Space, timeout_ms: int) -> Draft:
    d = Draft("K3", "*", "proved", "all " + sp.describe(), groups=("gamma", "assumes", "X", "C"))
    pts = sp.finite_points(K3_MAX_DISJUNCTION)
    try:
        if pts is not None:
            none_fits = z3.And(*[z3.Not(sp.constraints(sp.fixed_env(x))) for x in pts])
            bad = z3.And(sp.gamma(), sp.assumptions(), none_fits)
            decls = {k: s.term for k, s in sp.free.items()}
        else:
            xs = [sp.env[v.name].term for v in sp.design()]
            body = z3.Implies(sp.domain, z3.Not(sp.constraints()))
            bad = z3.And(
                sp.gamma(),
                sp.assumptions(),
                z3.ForAll(xs, body) if xs else z3.Not(sp.constraints()),
            )
            decls = {k: s.term for k, s in sp.free.items()}
        q = run(bad, decls, timeout_ms)
    except Unsupported as exc:
        d.outcome, d.reason = "unresolved", f"unsupported construct: {exc}"
        return d
    d.add(q.record())
    if q.result == "unsat":
        return d
    if q.result == "unknown":
        d.outcome, d.reason = "unresolved", f"solver unknown: {q.reason}"
        return d
    p = sp.decode(q.model)
    replay = _replay_k3(sp, p)
    d.replay = replay
    if replay.get("agrees"):
        d.outcome, d.counterexample = "counterexample", p
    else:
        d.outcome = "unresolved"
        d.reason = replay.get("reason", "solver witness not confirmed by the interpreter")
        d.data["witness"] = p
    return d


def _replay_k3(sp: Space, p: dict[str, Any]) -> dict[str, Any]:
    """Enumerate X at ``p`` with the interpreter: the witness holds when no x fits C."""
    if interp.holds(sp.template.scope.predicate, p) is not True:
        return {"agrees": False, "reason": "witness p is not in Gamma by the interpreter"}
    pts = sp.finite_points(K3_MAX_REPLAY)
    if pts is None:
        return {
            "agrees": False,
            "reason": f"K3 witness p={p} needs |X| > {K3_MAX_REPLAY} points enumerated to replay",
        }
    n = 0
    for x in pts:
        n += 1
        point = {**p, **x}
        if all(interp.holds(c.expr, point) is True for c in sp.template.constraints):
            return {"agrees": False, "reason": f"interpreter found x={x} satisfying C", "points": n}
    return {"interpreter": "kta.core.expr", "agrees": True, "points": n}
