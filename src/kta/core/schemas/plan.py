"""FormulationPlan, task header and pull records (WP19; docs/design/02 §5.3, §5.4).

A FormulationPlan is the structured product of the agent's formulation decisions F1–F7
(D-T18). The deterministic default is the same object (``author: default``), so plans
are diffed and scored against it (D-T19). A Pull is one ``kta`` knowledge fetch of a
worker, classified against the versioned access-discipline table (D-T20).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from kta.core.schemas.common import Body

Scalar = int | str | bool


class Citation(Body):
    """Why a non-default decision was taken: the structure it invokes (a Structure name,
    paper Table 1) and the records it cites (Finding, Measurement, Coupling, ...)."""

    structure: str | None = None
    evidence: list[str] = Field(default_factory=lambda: list[str]())


class PlanStep(Body):
    """F1: one chain item."""

    key: str
    template: str  # "<name>@<version>"
    applies_to: str = ""


class PlanBinding(Body):
    """F2: a chain item's Template variable bound to a family variable or a constant."""

    item: str
    var: str
    to: Scalar


class VarStatus(Body):
    """F3 status and F4 menu of one family variable."""

    name: str
    status: Literal["open", "fixed", "outer"] = "open"
    value: Scalar | None = None  # the fixed value
    menu: list[Scalar] | None = None  # None: the declared domain
    structure: str | None = None
    evidence: list[str] = Field(default_factory=lambda: list[str]())


class TableTerm(Body):
    """F5: an effect tabulated over a few variables."""

    effect: str  # transaction / storage key or constraint label
    over: list[str]
    by: Literal["compiler", "agent"] = "compiler"


class GlueSpec(Body):
    """F6: a family-specific constraint that belongs to no step (and why)."""

    id: str
    expr: str
    why: str


class DispatchPlan(Body):
    """F6: runtime variables and shape buckets (default: one bucket)."""

    runtime_vars: list[str] = Field(default_factory=lambda: list[str]())
    buckets: list[dict[str, Any]] = Field(default_factory=lambda: list[dict[str, Any]]())


class FormulationPlan(Body):
    family: str  # "<name>@<version>"
    author: Literal["default", "agent", "human"]
    chain_source: str  # family | retrieved:<chain id> | enumerated | agent
    chain: list[PlanStep]  # F1
    bindings: list[PlanBinding] = Field(default_factory=lambda: list[PlanBinding]())  # F2
    variables: list[VarStatus] = Field(default_factory=lambda: list[VarStatus]())  # F3, F4
    table_terms: list[TableTerm] = Field(default_factory=lambda: list[TableTerm]())  # F5
    glue: list[GlueSpec] = Field(default_factory=lambda: list[GlueSpec]())  # F6
    dispatch: DispatchPlan = Field(default_factory=DispatchPlan)  # F6
    generator: str | None = None  # F7: the generator's record / blob ID, None = none new
    # Citations of non-default sections other than variables: chain, bindings, table_terms,
    # glue, dispatch, generator.
    citations: dict[str, Citation] = Field(default_factory=lambda: dict[str, Citation]())

    def var(self, name: str) -> VarStatus | None:
        return next((v for v in self.variables if v.name == name), None)


class TaskHeader(Body):
    """The one-screen header of a case-loop task (02 §5.4) and the situation it states."""

    text: str
    situation: str  # a row id of the access-discipline table
    discipline: str  # the table version in force
    case: str | None = None  # Case record ID
    plan: str | None = None  # default FormulationPlan record ID


class Pull(Body):
    """One knowledge fetch of a worker; outside the table is allowed and logged."""

    task: str
    command: str  # show | card | why | query pool | query tmpl | query chain | src
    target: str  # record ID or query text
    kind: str | None = None  # kind of the record fetched
    situation: str  # the task header's situation ("" when the task has no header)
    row: str | None  # the table row allowing this pull; None = outside the table
    discipline: str
