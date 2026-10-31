"""Proof records of the verification loop (WP24; paper §2.2, §4; D631, D632).

- ``ProofOutline`` is the proof side of a Template (Σ in the paper): the relation R, the
  rank ρ, the environment assumptions it makes and the ones it guarantees, and for each
  obligation the checker model that discharges it. Seeded by hand or proposed by an agent
  (T3); it never sets a level by itself.
- ``ProofResult`` is one checker outcome for one condition of one subject, with the
  premises it rests on. Outcomes: ``proved`` (full declared domain), ``bounded`` (a bounded
  model or a bounded enumeration), ``counterexample`` (replayed by the independent
  interpreter), ``unresolved`` (timeout, unknown, unsupported construct; never a proof).
- ``ProofSummary`` folds the results of one check of one subject into a level. Assurance
  ``bounded`` and ``proved`` are read only from these records (kta.store.assurance).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from kta.core.schemas.common import Body

Outcome = Literal["proved", "bounded", "counterexample", "unresolved"]
# K1 C => Pre; K2 the four local conditions; K3 C satisfiable on Γ; encoding: the solver's
# encoding implies C; composition: Theorem 1 over a chain; candidate: the per-candidate
# check; wp: the weakest-precondition search. Open in practice; these are the ones written.
Condition = Literal["K1", "K2", "K3", "encoding", "composition", "candidate", "wp"]
ProofScope = Literal["template", "chain", "candidate"]
LocalCondition = Literal["initialisation", "step", "progress", "completion", "index"]


class Assumption(Body):
    name: str  # e.g. "bounded_arbitration"
    statement: str
    # An expression over p.* / hw.* only (never a design variable): an assumption about
    # the environment narrows the stated domain; it cannot stand in for an obligation.
    expr: str | None = None


class ObligationModel(Body):
    obligation: str  # the Template's obligation name
    condition: LocalCondition  # which local condition of paper §4.1 it discharges
    model: str  # checker model name (kta.checks.proof.models.MODELS, an open table)
    params: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())
    note: str | None = None


class ProofOutline(Body):
    template: str  # "<name>@<version>"
    version: int = 1
    relation: str  # R: fine state <-> coarse state
    rank: str  # ρ: work left, a natural number
    assumes: list[Assumption] = Field(default_factory=lambda: list[Assumption]())
    guarantees: list[str] = Field(default_factory=lambda: list[str]())  # assumption names
    models: list[ObligationModel] = Field(default_factory=lambda: list[ObligationModel]())
    note: str | None = None


class Query(Body):
    sha256: str  # of the SMT-LIB2 text, or of the enumeration description
    result: str  # unsat | sat | unknown | enumerated
    seconds: float


class ProofResult(Body):
    subject: str  # Template / Chain / Candidate record ID
    scope: ProofScope
    condition: Condition
    obligation: str  # obligation, precondition id or "*"
    outcome: Outcome
    domain: str  # the stated domain the outcome covers
    premises: dict[str, str]  # name -> content hash (template, pre, gamma, groups, versions)
    premise_key: str  # sha256 of the premises: equal keys may reuse the certificate
    queries: list[Query] = Field(default_factory=lambda: list[Query]())
    seconds: float = 0.0
    counterexample: dict[str, Any] | None = None
    replay: dict[str, Any] | None = None  # the independent interpreter's verdict
    reason: str = ""
    data: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())


class ProofSummary(Body):
    subject: str
    scope: ProofScope
    part: str = ""  # "" the whole subject; "step:<key>" one chain step checked under its bindings
    level: Literal["proved", "bounded"] | None  # None: neither (stays tested / declared)
    checked: list[dict[str, Any]]  # condition, obligation, outcome, domain, result, reused
    unchecked: list[dict[str, Any]]  # obligation, method, reason
    gaps: list[str]  # why the level is not higher, one line each
    premise_key: str
    checker: str
    solver: str
