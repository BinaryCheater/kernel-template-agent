"""Γ evaluation and the step-by-step reuse check of a chain on a new case (22 §4)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kta.core.errors import SchemaError
from kta.core.expr import Expr, ExprError
from kta.core.ids import is_record_id
from kta.core.schemas.evaluation import Case
from kta.core.schemas.knowledge import Chain, Template
from kta.knowledge import pool
from kta.knowledge.chains import PARAM_PREFIX
from kta.knowledge.types import ReusePlan, ReuseStep, ScopeResult
from kta.store.base import Store


def case_views(case: Case) -> list[dict[str, Any]]:
    """One ``case.*`` namespace per shape in the distribution.

    Γ must hold for every shape the case covers, so it is evaluated once per shape with
    the shape's dimensions next to the case's scalar fields (decision D209).
    """
    base: dict[str, Any] = {
        "problem": case.problem,
        "objective": case.objective,
        "target": case.target,
    }
    shapes = [s.get("shape", {}) for s in case.shape_dist] or [{}]
    return [{**base, **shape} for shape in shapes]


def scope_holds(
    t: Template,
    case: Case,
    hw: Mapping[str, Any],
    params: Mapping[str, Any],
    bound: Mapping[str, Expr] | None = None,
    assignment: Mapping[str, Any] | None = None,
) -> ScopeResult:
    """Whether Γ of ``t`` holds; ``holds=None`` when a name it needs has no value.

    ``bound`` maps ``p.*`` names to binding expressions over family variables, ``case.*``
    and ``hw.*``; they are evaluated per shape with ``assignment`` (decision D286).
    """
    try:
        expr = Expr.parse(t.scope.predicate, set())
    except ExprError as exc:
        return ScopeResult(None, f"scope predicate does not parse: {exc}")
    unknown: list[str] = []
    for view in case_views(case):
        try:
            p = dict(params)
            for name, e in (bound or {}).items():
                p[name] = e.eval(dict(assignment or {}), hw=hw, case=view)
            value = expr.eval({}, hw=hw, p=p, case=view)
        except ExprError as exc:
            unknown.append(str(exc))
            continue
        if not isinstance(value, bool):
            return ScopeResult(None, f"scope predicate gave {value!r}, not a boolean")
        if not value:
            shape = {k: v for k, v in view.items() if k not in ("problem", "objective", "target")}
            return ScopeResult(False, f"{t.scope.predicate} is false for shape {shape}")
    if unknown:
        return ScopeResult(None, f"cannot evaluate {t.scope.predicate}: {unknown[0]}")
    return ScopeResult(True, f"{t.scope.predicate} holds")


def problem_params(store: Store, case: Case) -> dict[str, Any]:
    """Numeric and boolean parameters of the case's Problem (strings are descriptions).

    ``Case.problem`` is a Problem record ID (as measure and checks require) or a name.
    """
    if is_record_id(case.problem) and store.exists(case.problem):
        body = store.get(case.problem).body
    else:
        hit = pool.find_by_name(store, "Problem", case.problem)
        if hit is None:
            return {}
        body = hit[1].body
    params: dict[str, Any] = body.get("params", {})
    return {k: v for k, v in params.items() if isinstance(v, (int, float, bool))}


def item_params(base: Mapping[str, Any], bindings: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in bindings.items():
        if key.startswith(PARAM_PREFIX) and not isinstance(value, str):
            out[key[len(PARAM_PREFIX) :]] = value
    return out


def bound_params(
    bindings: Mapping[str, Any], assignment: Mapping[str, Any] | None
) -> dict[str, Expr]:
    """``p.*`` bindings that are expressions, parsed over the assignment's variables;
    empty without an assignment (Γ then sees only constant bindings)."""
    if assignment is None:
        return {}
    out: dict[str, Expr] = {}
    for key, value in bindings.items():
        if key.startswith(PARAM_PREFIX) and isinstance(value, str):
            try:
                out[key[len(PARAM_PREFIX) :]] = Expr.parse(value, set(assignment))
            except ExprError:
                continue  # left unknown: scope_holds reports the missing name
    return out


def _alternatives(
    store: Store, t: Template, case: Case, hw: Mapping[str, Any], params: Mapping[str, Any]
) -> list[str]:
    """Templates related to ``t`` by alternative_of (either direction) whose Γ is not false."""
    out: list[str] = []
    for rid, other in pool.templates(store):
        if other.name == t.name or not pool.usable(store, rid):
            continue
        linked = any(r.type == "alternative_of" and r.target == other.name for r in t.relations)
        linked |= any(r.type == "alternative_of" and r.target == t.name for r in other.relations)
        if linked and scope_holds(other, case, hw, params).holds is not False:
            out.append(pool.template_ref(other.name, other.version))
    return sorted(out)


def reuse_plan(
    store: Store,
    chain_id: str,
    case: Case,
    hw: Mapping[str, Any],
    params: Mapping[str, Any] | None = None,
    assignment: Mapping[str, Any] | None = None,
) -> ReusePlan:
    """Evaluate Γ of every chain step on ``case``: keep / invalid / unknown, with reasons.

    ``p.*`` comes from the case's Problem params, then ``params``, then the item's own
    ``p.<name>`` constant bindings, then (given an ``assignment`` of the family variables)
    its expression bindings evaluated per shape.
    """
    record = store.get(chain_id)
    if record.kind != "Chain":
        raise SchemaError(f"{chain_id} is a {record.kind}, not a Chain")
    chain = Chain.model_validate(record.body)
    base = {**problem_params(store, case), **(params or {})}
    steps: list[ReuseStep] = []
    for item in chain.items:
        hit = pool.find_template(store, item.template)
        if hit is None:
            steps.append(ReuseStep(item.key, item.template, "unknown", "Template not in the store"))
            continue
        t = hit[1]
        p = item_params(base, item.bindings)
        result = scope_holds(t, case, hw, p, bound_params(item.bindings, assignment), assignment)
        if result.holds is True:
            steps.append(ReuseStep(item.key, item.template, "keep", result.reason))
        elif result.holds is False:
            alts = _alternatives(store, t, case, hw, p)
            steps.append(ReuseStep(item.key, item.template, "invalid", result.reason, alts))
        else:
            steps.append(ReuseStep(item.key, item.template, "unknown", result.reason))
    return ReusePlan(chain_id, steps)
