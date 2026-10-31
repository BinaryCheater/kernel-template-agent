"""Theorem 1 (paper §4.1) over a family's chain.

A chain is ``proved`` for its stated domain when

1. every step's K1 and K2 hold on the step's bound subspace (its chain bindings fix or
   alias its variables and bind its p.* to family expressions; K3 is not needed for
   soundness and is not rechecked here);
2. π maps feasible to feasible: every family assignment that satisfies the family's
   domains, glue and every step's bound constraints, for every case in the stated case
   domain, satisfies each step's Γ;
3. adjacent levels' environment assumptions are compatible: each step assumption is
   discharged by the bindings (an expression that holds on the feasible set), guaranteed
   by a step at a finer level, or declared as an assumption of the whole result.

Otherwise the chain takes the weakest step's level and names each gap. The theorem's
remaining premises (coarse executions terminate; the graph meets the workload contract)
are listed as assumptions of the whole result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import z3

from kta.core.ids import record_id
from kta.core.schemas import Actor, Record
from kta.core.schemas.knowledge import Chain, ChainItem, Family, Template, VarSpec
from kta.store.sqlite import SqliteStore

from . import records as rc
from .encode import Sym, Translator, Unsupported, declare, domain_values, free, infer
from .prove import CHECKER, Proved, find_template, outline_for, prove_template
from .smt import DEFAULT_TIMEOUT_MS, run
from .space import Space
from .types import RANK, Draft

THEOREM_ASSUMPTIONS = (
    "every maximal coarse execution terminates",
    "the computation graph meets the workload contract",
)


@dataclass
class FamilyEnv:
    syms: dict[str, Sym]
    domain: Any  # family variable domains, glue and the case domain

    def translate(self, src: str) -> Sym:
        term, kind = Translator(self.syms).value(src)
        return Sym(src, term, kind)


def family_env(
    fam: Family, chain: Chain, case_domain: Sequence[str], hw: Mapping[str, Any] | None
) -> FamilyEnv:
    srcs = [g.expr for g in fam.glue_constraints] + list(case_domain)
    values = [
        v for it in chain.items for v in it.bindings.values() if isinstance(v, str) and _is_expr(v)
    ]
    kinds = infer(srcs, values)
    syms: dict[str, Sym] = {}
    parts: list[Any] = []
    for v in fam.variables:
        s, dom = declare(v)
        syms[v.name] = s
        parts.append(dom)
    for name, (kind, lits) in kinds.items():
        if name in syms:
            continue
        root, _, attr = name.partition(".")
        if root == "hw" and hw is not None and attr in hw:
            syms[name] = (
                Sym(name, z3.IntVal(int(hw[attr])), "int")
                if not isinstance(hw[attr], bool)
                else Sym(name, z3.BoolVal(hw[attr]), "bool")
            )
        else:
            syms[name] = free(name, kind, lits)
    env = FamilyEnv(syms, z3.BoolVal(True))
    tr = Translator(syms)
    parts += [tr.bool(g.expr) for g in fam.glue_constraints]
    parts += [tr.bool(c) for c in case_domain]
    env.domain = z3.And(*parts)
    return env


def hw_attrs(hw: Mapping[str, Any] | None) -> dict[str, Sym]:
    """The Target's hardware values as constants (the stated domain names the Target)."""
    out: dict[str, Sym] = {}
    for k, v in (hw or {}).items():
        if isinstance(v, bool):
            out[f"hw.{k}"] = Sym(f"hw.{k}", z3.BoolVal(v), "bool")
        elif isinstance(v, int):
            out[f"hw.{k}"] = Sym(f"hw.{k}", z3.IntVal(v), "int")
        elif isinstance(v, float):
            out[f"hw.{k}"] = Sym(f"hw.{k}", z3.RealVal(v), "real")
    return out


def _is_expr(s: str) -> bool:
    try:
        compile(s, "<binding>", "eval")
    except SyntaxError:
        return False
    return True


def step_binding(
    t: Template, it: ChainItem, env: FamilyEnv, fam_vars: Mapping[str, VarSpec]
) -> tuple[dict[str, Any], dict[str, Sym], dict[str, list[Any]]]:
    """(restrict, attrs, menus) of one chain item."""
    restrict: dict[str, Any] = {}
    attrs: dict[str, Sym] = {}
    menus: dict[str, list[Any]] = {}
    tvars = {v.name: v for v in t.variables}
    for k, val in it.bindings.items():
        if k.startswith("p."):
            attrs[k] = (
                env.translate(val)
                if isinstance(val, str)
                else (
                    Sym(k, z3.BoolVal(val), "bool")
                    if isinstance(val, bool)
                    else Sym(k, z3.IntVal(int(val)), "int")
                )
            )
            continue
        spec = tvars.get(k)
        if spec is None:
            continue
        if isinstance(val, str) and val in fam_vars:
            fv = fam_vars[val]
            vals = domain_values(fv)
            if declare(spec)[0].kind == "enum":
                if vals is None:
                    raise Unsupported(f"{k} bound to {val}: structure variable with an open domain")
                menus[k] = vals
            else:
                restrict[k] = env.syms[val]
                if vals is not None:
                    menus[k] = vals
        elif isinstance(val, str) and isinstance(spec.domain, list) and val in spec.domain:
            restrict[k] = val
        elif isinstance(val, str):
            restrict[k] = Sym(val, env.translate(val).term, "int")
        else:
            restrict[k] = val
    return restrict, attrs, menus


def _assumption_gaps(
    steps: list[tuple[ChainItem, Template, Any, Space | None]],
    declared: Sequence[str],
    feasible: Any,
    timeout_ms: int,
) -> tuple[list[dict[str, Any]], list[str], list[Any]]:
    rows: list[dict[str, Any]] = []
    gaps: list[str] = []
    queries: list[Any] = []
    for it, t, outline, sp in steps:
        for a in outline.assumes if outline else []:
            how = None
            if a.expr and sp is not None:
                q = run(z3.And(feasible, sp.domain, z3.Not(sp.tr.bool(a.expr))), {}, timeout_ms)
                queries.append(q.record())
                if q.result == "unsat":
                    how = "discharged by the chain bindings"
            if how is None:
                below = [
                    s_it.key for s_it, s_t, s_o, _ in steps
                    if s_o and a.name in s_o.guarantees and s_t.level_from >= t.level_to
                ]  # fmt: skip
                if below:
                    how = f"guaranteed by {below}"
                elif a.name in declared:
                    how = "declared assumption of the whole result"
            rows.append(
                {"step": it.key, "assumption": a.name, "statement": a.statement, "how": how}
            )
            if how is None:
                gaps.append(
                    f"composition: assumption {a.name} of {it.key} is neither guaranteed "
                    "below nor declared"
                )
    return rows, gaps, queries


def prove_family(
    store: SqliteStore,
    family_id: str,
    now: datetime,
    *,
    case_domain: Sequence[str] = (),
    declared: Sequence[str] = (),
    hw: Mapping[str, Any] | None = None,
    actor: Actor = CHECKER,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> Proved:
    rec = store.get(family_id)
    fam = Family.model_validate(rec.body)
    chain_id = rec.refs.get("chain")
    if not isinstance(chain_id, str):
        raise Unsupported("family has no chain")
    chain = Chain.model_validate(store.get(chain_id).body)
    env = family_env(fam, chain, case_domain, hw)
    fam_vars = {v.name: v for v in fam.variables}
    steps: list[tuple[ChainItem, Template, Any, Space | None]] = []
    step_summaries: list[str] = []
    gaps: list[str] = []
    levels: list[str | None] = []
    bound: list[tuple[str, Space]] = []
    for it in chain.items:
        tid = find_template(store, it.template)
        t = Template.model_validate(store.get(tid).body)
        found = outline_for(store, it.template)
        outline = found[1] if found else None
        try:
            restrict, attrs, menus = step_binding(t, it, env, fam_vars)
            attrs = {**hw_attrs(hw), **attrs}
            sp = Space(
                t,
                list(outline.assumes) if outline else [],
                restrict,
                attrs,
                menus,
                env.domain,
                f"{it.key}:",
            )
        except Unsupported as exc:
            gaps.append(f"step {it.key}: bindings not encodable ({exc})")
            levels.append(None)
            steps.append((it, t, outline, None))
            continue
        p = prove_template(
            store, tid, now, actor=actor, subject=family_id, scope="chain", part=f"step:{it.key}",
            restrict=restrict, attrs=attrs, menus=menus, extra=env.domain, ns=f"{it.key}:",
            conditions=("K1", "K2"), timeout_ms=timeout_ms,
        )  # fmt: skip
        step_summaries.append(p.summary)
        levels.append(p.level)
        gaps += [f"step {it.key} ({it.template}): {g}" for g in p.gaps]
        steps.append((it, t, outline, sp))
        bound.append((it.key, sp))
    feasible = z3.And(env.domain, *[sp.constraints() for _, sp in bound])
    drafts = _feasibility(bound, feasible, timeout_ms)
    rows, agaps, aq = _assumption_gaps(steps, declared, feasible, timeout_ms)
    gaps += agaps
    gaps += [
        f"composition {d.obligation}: {d.outcome} ({d.reason})"
        for d in drafts
        if d.outcome != "proved"
    ]
    compat = Draft(
        "composition",
        "assumptions",
        "proved" if not agaps else "counterexample",
        "chain environment assumptions",
        data={
            "rows": rows,
            "theorem": list(THEOREM_ASSUMPTIONS),
            "case_domain": list(case_domain),
            "declared": list(declared),
        },
        reason="; ".join(agaps),
    )
    for q in aq:
        compat.add(q)
    drafts.append(compat)
    v = rc.versions()
    recs: list[Record] = []
    for d in drafts:
        prem = {
            "family": rc.h(rec.body),
            "chain": rc.h(chain.model_dump(mode="json")),
            "case_domain": rc.h(list(case_domain)),
            "declared": rc.h(list(declared)),
            "condition": f"composition:{d.obligation}",
            **v,
        }
        recs.append(rc.result_record(d, family_id, "chain", prem, actor, now, None))
    if any(lv is None for lv in levels) or any(d.outcome != "proved" for d in drafts):
        level = None
    else:
        level = min((lv for lv in levels if lv), key=lambda x: RANK[x], default=None)
    ids = store.put_many(recs)
    summary = rc.summary_record(
        family_id,
        "chain",
        level,
        list(zip(ids, recs, strict=True)),
        [],
        gaps,
        actor,
        now,
        steps=step_summaries,
    )
    sid = store.put(summary)
    return Proved(sid, level, ids, 0, sum(len(d.queries) for d in drafts), gaps, None)  # type: ignore[arg-type]


def _feasibility(bound: list[tuple[str, Space]], feasible: Any, timeout_ms: int) -> list[Draft]:
    out: list[Draft] = []
    for key, sp in bound:
        d = Draft(
            "composition",
            f"feasible:{key}",
            "proved",
            f"every feasible family design and case: Gamma of {key} holds",
        )
        try:
            q = run(z3.And(feasible, sp.domain, z3.Not(sp.gamma())), sp.decls(), timeout_ms)
        except Unsupported as exc:
            d.outcome, d.reason = "unresolved", str(exc)
            out.append(d)
            continue
        d.add(q.record())
        if q.result == "sat":
            d.outcome, d.counterexample = "counterexample", sp.decode(q.model)
            d.reason = f"a feasible design leaves the scope of {key}"
        elif q.result == "unknown":
            d.outcome, d.reason = "unresolved", q.reason
        out.append(d)
    return out


def latest_family_level(store: SqliteStore, family_id: str) -> str | None:
    s = rc.latest_summary(store, family_id)
    return s.body["level"] if s else None


__all__ = ["latest_family_level", "prove_family", "record_id"]
