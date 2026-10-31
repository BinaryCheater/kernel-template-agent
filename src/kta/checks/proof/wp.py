"""Weakest precondition on the declared menu (paper §4.3) and the relaxed Template it gives.

A model that decides safety point by point (``slot_rotation``) yields the safe set Safe
over the menu. Against it a precondition is ``exact`` (Pre = Safe), ``stronger`` (sound,
but excludes safe points: precondition loss Δpre ≥ 0) or ``unsound`` (admits an unsafe
point; K2 already reports the replayed counterexample). For a stronger Pre the search
looks for the simplest linear predicate a·v + c ≤ 0 equal to Safe on the menu, else the
explicit disjunction of the safe points, and re-checks the Template with it before the
result is recorded. Δpre itself is a cost question: the orchestrator re-solves with the
relaxed Template (``relaxed``) and the gap decomposition reads the difference.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kta.core.expr import Expr, ExprError
from kta.core.ids import record_id
from kta.core.schemas import Actor
from kta.core.schemas.knowledge import ExprSpec, Template
from kta.store.sqlite import SqliteStore

from . import records as rc
from .encode import Unsupported
from .prove import CHECKER, outline_for, template_ref
from .smt import DEFAULT_TIMEOUT_MS
from .template import check_template
from .types import Draft

COEFFS = (-2, -1, 0, 1, 2)
CONSTS = range(-8, 9)


@dataclass
class WeakestPre:
    pre_id: str
    relation: str  # exact | stronger | unsound
    current: str
    weaker: str | None
    table: list[dict[str, Any]]
    verified: str | None  # outcome of K2 with the weaker Pre


def _tables(t: Template, outline: Any, timeout_ms: int) -> tuple[list[str], list[dict[str, Any]]]:
    chk = check_template(t, outline, conditions=("K2",), timeout_ms=timeout_ms)
    tables = [d.data["table"] for d in chk.drafts if "table" in d.data]
    if not tables:
        raise Unsupported("no checker model of this Template decides safety point by point")
    names = [k for k in tables[0][0] if k not in ("pre", "safe")]
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for tab in tables:
        for row in tab:
            key = tuple(row[k] for k in names)
            cur = merged.setdefault(
                key, {**{k: row[k] for k in names}, "pre": row["pre"], "safe": True}
            )
            cur["safe"] = cur["safe"] and row["safe"] is True
    return names, list(merged.values())


def _fmt(names: Sequence[str], coeffs: Sequence[int], c: int) -> str:
    """a·v + c <= 0 written with one unit-coefficient variable on the left when possible."""
    lead = next((i for i, a in enumerate(coeffs) if a == 1), None)
    if lead is None:
        terms = " + ".join(f"{a} * {n}" for a, n in zip(coeffs, names, strict=True) if a)
        return f"{terms} + {c} <= 0" if terms else f"{c} <= 0"
    rhs: list[str] = []
    for a, n in zip(coeffs, names, strict=True):
        if n == names[lead] or not a:
            continue
        rhs.append(f"{'+' if -a > 0 else '-'} {'' if abs(a) == 1 else f'{abs(a)} * '}{n}")
    if c:
        rhs.append(f"{'+' if -c > 0 else '-'} {abs(c)}")
    right = " ".join(rhs).lstrip("+ ").strip()
    if right.startswith("- "):
        right = "-" + right[2:]
    return f"{names[lead]} <= {right or 0}"


def linear_fit(names: Sequence[str], table: list[dict[str, Any]]) -> str | None:
    best: tuple[int, int, str] | None = None
    for coeffs in itertools.product(COEFFS, repeat=len(names)):
        if not any(coeffs):
            continue
        for c in CONSTS:
            if all(
                (sum(a * int(r[n]) for a, n in zip(coeffs, names, strict=True)) + c <= 0)
                == r["safe"]
                for r in table
            ):
                key = (sum(1 for a in coeffs if a), sum(abs(a) for a in coeffs) + abs(c))
                if best is None or key < best[:2]:
                    best = (*key, _fmt(names, coeffs, c))
    return best[2] if best else None


def disjunction(names: Sequence[str], table: list[dict[str, Any]]) -> str:
    pts = [" and ".join(f"{n} == {r[n]!r}" for n in names) for r in table if r["safe"]]
    return " or ".join(f"({p})" for p in pts) if pts else "False"


def relaxed(t: Template, pre_id: str, weaker: str) -> Template:
    """``t`` with the precondition and the constraint it is linked to (same_as, or the
    textually equal one) replaced by ``weaker``: the formulation the weaker Pre admits."""
    pre = next(p for p in t.preconditions if p.id == pre_id)
    linked = {pre.same_as} if pre.same_as else {c.id for c in t.constraints if c.expr == pre.expr}
    pres = [p.model_copy(update={"expr": weaker}) if p.id == pre_id else p for p in t.preconditions]
    cons = [c.model_copy(update={"expr": weaker}) if c.id in linked else c for c in t.constraints]
    return t.model_copy(update={"preconditions": pres, "constraints": cons})


def search(t: Template, outline: Any, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> WeakestPre:
    names, table = _tables(t, outline, timeout_ms)
    pres = [p for p in t.preconditions if _vars(p, t) <= set(names)]
    if len(pres) != 1:
        raise Unsupported(
            f"need exactly one precondition over {names}, found {[p.id for p in pres]}"
        )
    pre = pres[0]
    unsafe = [r for r in table if r["pre"] and not r["safe"]]
    if unsafe:
        return WeakestPre(pre.id, "unsound", pre.expr, None, table, None)
    if all(r["pre"] == r["safe"] for r in table):
        return WeakestPre(pre.id, "exact", pre.expr, None, table, None)
    weaker = linear_fit(names, table) or disjunction(names, table)
    chk = check_template(
        relaxed(t, pre.id, weaker), outline, conditions=("K1", "K2"), timeout_ms=timeout_ms
    )
    k2 = [d.outcome for d in chk.drafts if d.condition == "K2" and "table" in d.data]
    verified = (
        "bounded"
        if k2 and all(o in ("bounded", "proved") for o in k2)
        else (k2[0] if k2 else "unresolved")
    )
    return WeakestPre(pre.id, "stronger", pre.expr, weaker, table, verified)


def _vars(p: ExprSpec, t: Template) -> set[str]:
    try:
        return set(Expr.parse(p.expr, {v.name for v in t.variables}).free_vars)
    except ExprError:
        return set()


def record_search(
    store: SqliteStore,
    template_id: str,
    now: datetime,
    *,
    actor: Actor = CHECKER,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> tuple[str, WeakestPre]:
    """Run ``search`` on a stored Template and write it as a ProofResult (condition wp)."""
    t = Template.model_validate(store.get(template_id).body)
    found = outline_for(store, template_ref(t))
    outline_id, outline = found if found else (None, None)
    w = search(t, outline, timeout_ms)
    outcome = {"exact": "bounded", "unsound": "counterexample"}.get(
        w.relation, w.verified or "unresolved"
    )
    d = Draft(
        "wp",
        w.pre_id,
        outcome,
        "the Template's declared menu of "
        + ", ".join(k for k in w.table[0] if k not in ("pre", "safe")),
    )
    d.data = {
        "relation": w.relation,
        "current": w.current,
        "weaker": w.weaker,
        "table": w.table,
        "verified": w.verified,
    }
    if w.relation == "unsound":
        d.counterexample = next(r for r in w.table if r["pre"] and not r["safe"])
        d.reason = "Pre admits an unsafe point (see the K2 counterexample and its replay)"
    g = rc.groups(t, outline, {})
    prem = {
        **{k: g[k] for k in ("gamma", "X", "C", "pre", "program", "assumes")},
        "condition": f"wp:{w.pre_id}",
        **rc.versions(),
    }
    rid = store.put(rc.result_record(d, template_id, "template", prem, actor, now, outline_id))
    return rid, w


def latest(store: SqliteStore, template_id: str) -> Mapping[str, Any] | None:
    """The newest live weakest-precondition result of a Template."""
    for rec in store.query(
        "ProofResult",
        where={"subject": template_id, "condition": "wp"},
        order="-created_at",
        limit=5,
    ):
        return rec.body if rc.alive(store, record_id(rec)) else None
    return None
