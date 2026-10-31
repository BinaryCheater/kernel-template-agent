"""Chain validation: items, bindings, the ``after`` DAG and Template relations (22 §3.1)."""

from __future__ import annotations

from typing import Any

from kta.core.errors import SchemaError
from kta.core.expr import Expr, ExprError
from kta.core.schemas.knowledge import Chain, ChainItem, RangeSpec, Template, VarSpec
from kta.knowledge import pool
from kta.knowledge.learning import classify
from kta.knowledge.types import Issue
from kta.store.base import Store

# A binding key "p.<param>" binds a parameter of the refined transaction (decision D205).
PARAM_PREFIX = "p."


def _in_domain(v: VarSpec, value: int | bool) -> bool:
    d = v.domain
    if isinstance(d, RangeSpec):
        if isinstance(value, bool) or not d.lo <= value <= d.hi or (value - d.lo) % d.step:
            return False
        return not d.pow2 or (value > 0 and value & (value - 1) == 0)
    return any(type(x) is type(value) and x == value for x in d)


def _bindings(item: ChainItem, t: Template) -> list[Issue]:
    where = f"items.{item.key}.bindings"
    variables = {v.name: v for v in t.variables}
    issues = [
        Issue("unbound_variable", f"{item.template} variable {name!r} is not bound", where)
        for name in variables
        if name not in item.bindings
    ]
    for key, value in item.bindings.items():
        if key.startswith(PARAM_PREFIX):
            continue
        var = variables.get(key)
        if var is None:
            issues.append(
                Issue("unknown_binding", f"{item.template} has no variable {key!r}", where)
            )
        elif not isinstance(value, str) and not _in_domain(var, value):
            issues.append(Issue("binding_out_of_domain", f"{key}={value!r}", where))
    return issues


def _constant_checks(item: ChainItem, t: Template) -> list[Issue]:
    """Evaluate C and Pre when every variable they use is bound to a number or boolean.

    A false result is a warning: a reconstruction chain records what a kernel does, and
    a violated constraint says the Template does not describe that step exactly.
    """
    env = {k: v for k, v in item.bindings.items() if not isinstance(v, str)}
    p = {k[len(PARAM_PREFIX) :]: v for k, v in item.bindings.items() if k.startswith(PARAM_PREFIX)}
    declared = {v.name for v in t.variables}
    issues: list[Issue] = []
    # A precondition with same_as repeats a constraint verbatim; report it once.
    for spec in [*t.constraints, *(c for c in t.preconditions if c.same_as is None)]:
        try:
            expr = Expr.parse(spec.expr, declared)
            if not expr.free_vars <= env.keys() or any(
                not r.startswith("p.") for r in expr.attr_refs
            ):
                continue
            value: Any = expr.eval(env, p=p)
        except ExprError:
            continue  # unparsable expressions are reported by validate_template
        if value is False:
            issues.append(
                Issue(
                    "constraint_false",
                    f"{spec.id}: {spec.expr} is false for the bound values",
                    f"items.{item.key}",
                    "warning",
                )
            )
    return issues


def _dag(ch: Chain) -> list[Issue]:
    keys = [i.key for i in ch.items]
    issues = [
        Issue("duplicate_key", f"key {k!r} used twice", "items")
        for k in set(keys)
        if keys.count(k) > 1
    ]
    known = set(keys)
    for item in ch.items:
        issues += [
            Issue("unknown_after", f"{item.key} is after unknown key {a!r}", f"items.{item.key}")
            for a in item.after
            if a not in known
        ]
    deps = {i.key: {a for a in i.after if a in known} for i in ch.items}
    while deps:
        ready = [k for k, d in deps.items() if not d]
        if not ready:
            return [*issues, Issue("cycle", f"items {sorted(deps)} form a cycle in after", "items")]
        for k in ready:
            del deps[k]
        for d in deps.values():
            d.difference_update(ready)
    return issues


def _relations(ch: Chain, used: dict[str, Template]) -> list[Issue]:
    present = {t.name for t in used.values()}
    issues: list[Issue] = []
    for ref, t in sorted(used.items()):
        for r in t.relations:
            if r.type == "requires" and r.target not in present:
                issues.append(Issue("missing_required", f"{ref} requires {r.target}", "items"))
            if r.type == "excludes" and r.target in present and r.target != t.name:
                issues.append(Issue("excluded_pair", f"{ref} excludes {r.target}", "items"))
    return issues


def resolve_templates(
    ch: Chain, store: Store
) -> tuple[dict[str, tuple[str, Template]], list[Issue]]:
    """Template ref -> (record ID, Template) for every item; issues for unknown refs."""
    found: dict[str, tuple[str, Template]] = {}
    issues: list[Issue] = []
    for item in ch.items:
        if item.template in found:
            continue
        try:
            hit = pool.find_template(store, item.template)
        except SchemaError as exc:
            issues.append(Issue("bad_template_ref", str(exc), f"items.{item.key}"))
            continue
        if hit is None:
            issues.append(
                Issue(
                    "unknown_template", f"{item.template} is not in the store", f"items.{item.key}"
                )
            )
        else:
            found[item.template] = hit
    return found, issues


def validate_chain(ch: Chain, store: Store) -> list[Issue]:
    ch = Chain.model_validate(ch.model_dump())
    if not ch.items:
        return [Issue("empty_chain", "a chain has at least one item", "items")]
    found, issues = resolve_templates(ch, store)
    for item in ch.items:
        hit = found.get(item.template)
        if hit is not None:
            issues += _bindings(item, hit[1]) + _constant_checks(item, hit[1])
    issues += _dag(ch)
    issues += _relations(ch, {ref: t for ref, (_, t) in found.items()})
    return classify(issues)
