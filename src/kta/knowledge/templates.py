"""Template validation, duplicate detection and proposal (spec/modules/knowledge.md)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from kta.core.context import Context
from kta.core.errors import SchemaError
from kta.core.expr import Expr, ExprError
from kta.core.schemas import Actor, Record
from kta.core.schemas.knowledge import ExprSpec, RangeSpec, Template, VarSpec
from kta.knowledge import pool
from kta.knowledge.learning import classify
from kta.knowledge.types import Issue, errors_of
from kta.store.base import Store

# Author fields that must not be blank (docs/design/21 §1.2 lists what a Template needs).
_REQUIRED_TEXT = ("name", "summary", "backward_map")


def check_expr(
    src: str, declared: set[str], where: str, hw: Mapping[str, Any] | None
) -> tuple[Expr | None, list[Issue]]:
    """Parse one expression; unknown ``hw.*`` names are warnings when ``hw`` is given."""
    try:
        expr = Expr.parse(src, declared)
    except ExprError as exc:
        return None, [Issue("bad_expression", str(exc), where)]
    issues: list[Issue] = []
    if hw:
        for ref in sorted(expr.attr_refs):
            root, _, attr = ref.partition(".")
            if root == "hw" and attr not in hw:
                issues.append(
                    Issue(
                        "unknown_hw_param",
                        f"{ref} is not in the hardware contract",
                        where,
                        "warning",
                    )
                )
    return expr, issues


def _check_domain(v: VarSpec) -> list[Issue]:
    where = f"variables.{v.name}"
    d = v.domain
    if isinstance(d, RangeSpec):
        if d.lo > d.hi or d.step < 1:
            return [Issue("bad_domain", f"range lo={d.lo} hi={d.hi} step={d.step}", where)]
        return []
    if not d:
        return [Issue("bad_domain", "empty domain", where)]
    if len({repr(x) for x in d}) != len(d):
        return [Issue("bad_domain", "repeated values", where)]
    return []


def _fields(t: Template) -> list[Issue]:
    issues = [
        Issue("missing_field", f"{f} is empty", f)
        for f in _REQUIRED_TEXT
        if not str(getattr(t, f)).strip()
    ]
    for f, value in (
        ("object.type", t.object.type),
        ("object.description", t.object.description),
        ("program.description", t.program.description),
        ("scope.text", t.scope.text),
        ("signature.algorithm", t.signature.algorithm),
    ):
        if not value.strip():
            issues.append(Issue("missing_field", f"{f} is empty", f))
    if not t.variables:
        issues.append(
            Issue("missing_field", "a Template introduces at least one variable", "variables")
        )
    if not t.obligations:
        issues.append(Issue("missing_field", "the obligation list Σ is empty", "obligations"))
    if t.version < 1:
        issues.append(Issue("bad_version", "versions start at 1", "version"))
    if t.level_to < t.level_from:
        issues.append(Issue("bad_level", f"level_to {t.level_to} < level_from {t.level_from}"))
    return issues


def _unique(items: Iterable[str], kind: str, where: str) -> list[Issue]:
    seen: set[str] = set()
    issues: list[Issue] = []
    for item in items:
        if item in seen:
            issues.append(Issue(kind, f"{item!r} declared twice", where))
        seen.add(item)
    return issues


def _expressions(t: Template, hw: Mapping[str, Any] | None) -> list[Issue]:
    declared = {v.name for v in t.variables}
    issues: list[Issue] = []
    for v in t.variables:
        issues += _check_domain(v)
        if v.when:
            issues += check_expr(v.when, declared, f"variables.{v.name}.when", hw)[1]
    exprs: list[tuple[str, ExprSpec]] = [("constraints", c) for c in t.constraints]
    exprs += [("preconditions", c) for c in t.preconditions]
    for group, spec in exprs:
        issues += check_expr(spec.expr, declared, f"{group}.{spec.id}", hw)[1]
        if not spec.meaning.strip():
            issues.append(Issue("missing_field", "meaning is empty", f"{group}.{spec.id}"))
    for i, term in enumerate(t.cost):
        issues += check_expr(term.work, declared, f"cost[{i}].work", hw)[1]
    # Γ ranges over pattern/workload parameters and the hardware contract, not over X.
    issues += check_expr(t.scope.predicate, set(), "scope.predicate", hw)[1]
    return issues


def _same_as(t: Template) -> list[Issue]:
    by_id = {c.id: c for c in t.constraints}
    issues: list[Issue] = []
    for pre in t.preconditions:
        if pre.same_as is None:
            continue
        target = by_id.get(pre.same_as)
        where = f"preconditions.{pre.id}"
        if target is None:
            issues.append(Issue("bad_same_as", f"no constraint with id {pre.same_as!r}", where))
        elif _normal(target.expr) != _normal(pre.expr):
            # same_as claims K1 holds verbatim: the two texts must be the same predicate.
            issues.append(
                Issue("same_as_mismatch", f"{pre.expr!r} differs from {target.expr!r}", where)
            )
    issues += [
        Issue("bad_same_as", "same_as is only used in preconditions", f"constraints.{c.id}")
        for c in t.constraints
        if c.same_as is not None
    ]
    return issues


def _normal(src: str) -> str:
    return "".join(src.split())


def _all_exprs(t: Template) -> list[tuple[str, str, set[str]]]:
    """(where, source, declared names) of every expression in ``t``."""
    xs = {v.name for v in t.variables}
    out = [(f"variables.{v.name}.when", v.when, xs) for v in t.variables if v.when]
    out += [(f"constraints.{c.id}", c.expr, xs) for c in t.constraints]
    out += [(f"preconditions.{c.id}", c.expr, xs) for c in t.preconditions]
    out += [(f"cost[{i}].work", c.work, xs) for i, c in enumerate(t.cost)]
    out += [("scope.predicate", t.scope.predicate, set())]
    return out


def _pattern_params(t: Template, store: Store) -> list[Issue]:
    """``p.<name>`` must be a parameter of the refined or produced Patterns (warning, D217).

    Unparsable expressions are reported by ``_expressions``; unknown Patterns by
    ``_references``.
    """
    declared: set[str] = set()
    for name in {t.refines, *t.produces}:
        hit = pool.find_by_name(store, "Pattern", name)
        if hit is None:
            return []
        declared |= {str(p["name"]) for p in hit[1].body.get("params", [])}
    issues: list[Issue] = []
    for where, src, names in _all_exprs(t):
        try:
            refs = Expr.parse(src, names).attr_refs
        except ExprError:
            continue
        issues += [
            Issue(
                "undeclared_pattern_param",
                f"{r} is not a parameter of {sorted({t.refines, *t.produces})}",
                where,
                "warning",
            )
            for r in sorted(refs)
            if r.startswith("p.") and r[2:] not in declared
        ]
    return issues


def _references(t: Template, store: Store) -> list[Issue]:
    patterns = pool.names(store, "Pattern")
    issues = [
        Issue("unknown_pattern", f"Pattern {name!r} is not in the store", where)
        for where, name in [("refines", t.refines), *(("produces", p) for p in t.produces)]
        if name not in patterns
    ]
    issues += _requires(t.requires, store)
    known = pool.names(store, "Template") | {t.name}
    issues += [
        Issue("unknown_relation_target", f"{r.type} -> {r.target!r}: no such Template", "relations")
        for r in t.relations
        if r.target not in known
    ]
    return issues


def _requires(requires: list[str], store: Store) -> list[Issue]:
    """Each name must be a Capability; a legacy HwFeature name is accepted with a warning.

    The issue kind keeps its pre-D442 name ``unknown_hw_feature`` (Issue kinds are read by
    done-checks and locked acceptance tests).
    """
    capabilities = pool.names(store, "Capability")
    features = pool.names(store, "HwFeature")
    issues: list[Issue] = []
    for name in requires:
        if name in capabilities:
            continue
        if name in features:
            msg = f"{name!r} is an ISA feature (HwFeature), not an abstract Capability"
            issues.append(Issue("hw_feature_not_capability", msg, "requires", "warning"))
        else:
            msg = f"Capability {name!r} is not in the store"
            issues.append(Issue("unknown_hw_feature", msg, "requires"))
    return issues


def duplicates(t: Template, store: Store) -> list[Issue]:
    """Same refines, same object type, same variable names and cost units (21 §1.3)."""
    sig = (t.refines, t.object.type, {v.name for v in t.variables}, {c.unit for c in t.cost})
    issues: list[Issue] = []
    for _, other in pool.templates(store):
        if other.name == t.name:
            continue  # another version of the same Template is not a duplicate
        osig = (
            other.refines,
            other.object.type,
            {v.name for v in other.variables},
            {c.unit for c in other.cost},
        )
        if osig == sig:
            issues.append(
                Issue(
                    "possible_duplicate",
                    f"same refines/object/variables/cost units as {other.name}@{other.version}",
                    "",
                    "warning",
                )
            )
    return issues


def validate_template(
    t: Template, store: Store, hw: Mapping[str, Any] | None = None
) -> list[Issue]:
    """All issues of ``t`` against the pool in ``store``; no ``error`` issue means valid.

    ``hw`` is the hardware contract (HwModel.limits plus derived facts) that ``hw.*``
    names are checked against; pass ``None`` or ``{}`` to skip that check. Expressibility
    gaps of the language come back with severity ``gap`` (D560): counted, not refusing.
    """
    t = Template.model_validate(t.model_dump())
    issues = _fields(t)
    issues += _unique([v.name for v in t.variables], "duplicate_variable", "variables")
    issues += _unique(
        [c.id for c in [*t.constraints, *t.preconditions]],
        "duplicate_id",
        "constraints/preconditions",
    )
    issues += _unique([o.name for o in t.obligations], "duplicate_obligation", "obligations")
    issues += _expressions(t, hw)
    issues += _same_as(t)
    issues += _references(t, store)
    issues += _pattern_params(t, store)
    issues += duplicates(t, store)
    return classify(issues)


def propose_template(
    ctx: Context,
    store: Store,
    t: Template,
    sources: list[str],
    actor: Actor | None = None,
    hw: Mapping[str, Any] | None = None,
    validate_in: Store | None = None,
) -> str:
    """Validate and write ``t`` (status proposed, assurance declared); returns the record ID.

    ``validate_in`` (default ``store``) is the pool ``t`` is validated against, e.g. a
    PendingView holding Patterns and sibling Templates written in the same task (D383).

    Refuses when validation has errors, when no Source is given for a non-human actor,
    or when ``name@version`` already exists with a different body (a change needs a new
    version, decision D202). Writing the same body again returns the existing ID.
    """
    actor = actor or Actor(type="tool", name="kta knowledge propose")
    bad = errors_of(validate_template(t, validate_in or store, hw))
    if bad:
        raise SchemaError("; ".join(str(i) for i in bad))
    existing = pool.find_template(store, pool.template_ref(t.name, t.version))
    if existing is not None:
        if existing[1].model_dump() == t.model_dump():
            return existing[0]
        raise SchemaError(f"{t.name}@{t.version} exists with a different body; bump the version")
    refs: dict[str, str | list[str]] = {}
    if sources:
        refs["sources"] = sorted(set(sources))
    previous = pool.find_template(store, pool.template_ref(t.name, t.version - 1))
    if previous is not None:
        refs["previous"] = previous[0]
    return store.put(Record.new("Template", t, actor, ctx.clock.now(), refs=refs))
