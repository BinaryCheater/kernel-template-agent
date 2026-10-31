"""Once per candidate (paper §4.2): the solver's assignment lies in C and its context in Γ.

Because a proved Template's constraints imply its preconditions (K1), checking a design
reduces to evaluating each step's C and Γ at the assignment under the chain bindings. The
evaluation uses the expression interpreter, not the solver's encoding, so it also catches
an encoding that admitted a point outside C. A candidate of a family without a proved or
bounded composition may still be measured (exploration); its result is never used in a
correctness claim (``claims: exploration``).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kta.core.expr import Expr, ExprError
from kta.core.ids import record_id
from kta.core.schemas import Actor
from kta.core.schemas.knowledge import ChainItem, Template
from kta.store.sqlite import SqliteStore

from . import records as rc
from .prove import CHECKER
from .types import Draft


@dataclass
class CandidateCheck:
    ok: bool
    failures: list[str] = field(default_factory=lambda: list[str]())
    unknown: list[str] = field(default_factory=lambda: list[str]())  # not evaluable here
    steps: list[dict[str, Any]] = field(default_factory=lambda: list[dict[str, Any]]())
    seconds: float = 0.0


def _eval(
    src: str,
    env: Mapping[str, Any],
    p: Mapping[str, Any],
    case: Mapping[str, Any],
    hw: Mapping[str, Any],
) -> Any:
    return Expr.parse(src, set(env)).eval(env, hw=hw, p=p, case=case)


def _bound(
    it: ChainItem,
    t: Template,
    assignment: Mapping[str, Any],
    case: Mapping[str, Any],
    hw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    env: dict[str, Any] = {}
    p: dict[str, Any] = {}
    names = {v.name for v in t.variables}
    for k, val in it.bindings.items():
        if isinstance(val, str) and val not in {
            str(x)
            for v in t.variables
            if v.name == k and isinstance(v.domain, list)
            for x in v.domain
        }:
            got = _eval(val, assignment, {}, case, hw)
        else:
            got = val
        if k.startswith("p."):
            p[k[2:]] = got
        elif k in names:
            env[k] = got
    return env, p


def check_candidate(
    steps: Sequence[tuple[ChainItem, Template]],
    assignment: Mapping[str, Any],
    case: Mapping[str, Any],
    hw: Mapping[str, Any],
) -> CandidateCheck:
    t0 = time.perf_counter()
    out = CandidateCheck(True)
    for it, t in steps:
        row: dict[str, Any] = {"step": it.key, "template": it.template}
        try:
            env, p = _bound(it, t, assignment, case, hw)
            row["scope"] = bool(_eval(t.scope.predicate, env, p, case, hw))
            row["violated"] = [c.id for c in t.constraints if not _eval(c.expr, env, p, case, hw)]
            row["pre_false"] = [
                x.id for x in t.preconditions if not _eval(x.expr, env, p, case, hw)
            ]
        except ExprError as exc:
            # an unbound workload parameter: not a violation, but no claim either
            row["unknown"] = str(exc)
            out.unknown.append(f"{it.key}: {exc}")
            out.steps.append(row)
            continue
        if not row["scope"]:
            out.failures.append(f"{it.key}: context outside Gamma ({t.scope.predicate})")
        out.failures += [f"{it.key}: constraint {c} false" for c in row["violated"]]
        out.failures += [f"{it.key}: precondition {c} false" for c in row["pre_false"]]
        out.ok = out.ok and row["scope"] and not row["violated"] and not row["pre_false"]
        out.steps.append(row)
    out.seconds = time.perf_counter() - t0
    return out


def claims_for(family_level: str | None, chk: CandidateCheck) -> str:
    sound = chk.ok and not chk.unknown and family_level in ("proved", "bounded")
    return "correctness" if sound else "exploration"


def record_candidate(
    store: SqliteStore,
    candidate_id: str,
    family_id: str,
    chk: CandidateCheck,
    assignment: Mapping[str, Any],
    case: Mapping[str, Any],
    now: datetime,
    actor: Actor = CHECKER,
) -> str:
    summary = rc.latest_summary(store, family_id)
    level = summary.body["level"] if summary else None
    d = Draft(
        "candidate",
        "*",
        "proved" if chk.ok else "counterexample",
        f"assignment {dict(assignment)} on shape {dict(case)}",
        seconds=chk.seconds,
        reason="; ".join(chk.failures),
        data={
            "claims": claims_for(level, chk),
            "unknown": chk.unknown,
            "family_level": level,
            "family_summary": record_id(summary) if summary else None,
            "steps": chk.steps,
        },
    )
    if not chk.ok:
        d.counterexample = {"assignment": dict(assignment), "case": dict(case)}
    prem = {
        "family": family_id,
        "assignment": rc.h(dict(assignment)),
        "case": rc.h(dict(case)),
        "condition": "candidate",
        **rc.versions(),
    }
    return store.put(rc.result_record(d, candidate_id, "candidate", prem, actor, now, None))
