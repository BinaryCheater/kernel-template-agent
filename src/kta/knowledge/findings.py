"""Findings and couplings: empirical knowledge with scope (D-T9, D444, D447).

A Finding unifies the legacy Regime and Negative kinds. Legacy records are read as
Findings (``legacy=True``) without being rewritten:

- Regime(structure, context, holds, how) -> kind ``regime``, subject = structure,
  scope.workload = context, basis ``proved`` or ``observed``;
- Negative(subject, condition, effect) -> kind ``negative``, claim = effect,
  scope.condition = condition, basis ``observed``.

A Finding scoped to other Targets than the one asked about is a *prior* there (02 §2:
a chip change demotes findings to priors); ``effective_status`` says so without writing
anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from kta.core.schemas import Record
from kta.core.schemas.knowledge import Coupling, Finding, Scope
from kta.knowledge import pool
from kta.store.base import Store

PRIOR = "prior"


@dataclass(frozen=True)
class FindingRef:
    id: str
    body: Finding
    status: str
    legacy: bool = False


@dataclass(frozen=True)
class CouplingRef:
    id: str
    body: Coupling
    status: str


def as_finding(rec: Record) -> Finding:
    """The Finding view of a Finding, Regime or Negative record."""
    b = rec.body
    if rec.kind == "Finding":
        return Finding.model_validate(b)
    if rec.kind == "Regime":
        verb = "holds" if b["holds"] else "fails"
        return Finding(
            claim=f"structure {b['structure']} {verb} in {b['context']}",
            kind="regime",
            subject=str(b["structure"]),
            scope=Scope(workload=str(b["context"])),
            holds=bool(b["holds"]),
            basis="proved" if b["how"] == "proved" else "observed",
        )
    if rec.kind == "Negative":
        return Finding(
            claim=str(b["effect"]),
            kind="negative",
            subject=str(b["subject"]),
            scope=Scope(condition=str(b["condition"])),
            basis="observed",
        )
    raise ValueError(f"{rec.kind} is not a Finding kind")


def in_scope(scope: Scope, target: str | None) -> bool:
    return target is None or not scope.targets or target in scope.targets


def effective_status(status: str, scope: Scope, target: str | None) -> str:
    """``prior`` when the claim was established on other Targets only."""
    return status if in_scope(scope, target) else PRIOR


def findings(
    store: Store,
    subject: str | None = None,
    kind: str | None = None,
    target: str | None = None,
    usable_only: bool = True,
) -> list[FindingRef]:
    """Findings (new and legacy) about ``subject``; status is ``prior`` out of scope."""
    out: list[FindingRef] = []
    for k in ("Finding", "Regime", "Negative"):
        for rid, rec in pool.with_ids(pool.records(store, k, order="created_at")):
            if usable_only and not pool.usable(store, rid):
                continue
            f = as_finding(rec)
            if subject is not None and f.subject != subject:
                continue
            if kind is not None and f.kind != kind:
                continue
            status = effective_status(pool.status_of(store, rid)[0], f.scope, target)
            out.append(FindingRef(rid, f, status, legacy=k != "Finding"))
    return out


def couplings(
    store: Store, member: str | None = None, target: str | None = None
) -> list[CouplingRef]:
    """Usable couplings that include ``member`` (any member kind) and apply on ``target``.

    Consumers: the formulation opens coupled variables together; the hardware model
    takes coupled resources as interaction candidates (02 §8.4).
    """
    out: list[CouplingRef] = []
    for rid, rec in pool.with_ids(pool.records(store, "Coupling", order="created_at")):
        if not pool.usable(store, rid):
            continue
        c = Coupling.model_validate(rec.body)
        if member is not None and member not in {m.name for m in c.members}:
            continue
        if not in_scope(c.scope, target):
            continue
        out.append(CouplingRef(rid, c, pool.status_of(store, rid)[0]))
    return out
