"""Result types of the knowledge functions (module-internal dataclasses, conventions §3.1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from kta.core.schemas.knowledge import Chain, Family

Severity = Literal["error", "warning"]
Verdict = Literal["keep", "invalid", "unknown"]


@dataclass(frozen=True)
class Issue:
    """One finding of a validator.

    ``error`` issues make the object invalid; ``warning`` issues are hints that the
    author must answer (``possible_duplicate`` is a warning, knowledge.md "查重是提示").
    """

    kind: str
    message: str
    where: str = ""
    severity: Severity = "error"

    def __str__(self) -> str:
        loc = f" [{self.where}]" if self.where else ""
        return f"{self.severity} {self.kind}{loc}: {self.message}"


def errors_of(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.severity == "error"]


@dataclass(frozen=True)
class TxDescription:
    """A transaction seen in a kernel or a case, to be matched against the Pattern table.

    ``category`` is an optional guess among the schema categories; ``params`` are the
    parameter names (and values) the describer could state.
    """

    text: str
    level: int | None = None
    category: str | None = None
    params: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())


@dataclass(frozen=True)
class ScopeResult:
    holds: bool | None  # None: the predicate could not be evaluated
    reason: str


@dataclass(frozen=True)
class ReuseStep:
    key: str
    template: str
    verdict: Verdict
    reason: str
    alternatives: list[str] = field(default_factory=lambda: list[str]())


@dataclass(frozen=True)
class ReusePlan:
    chain_id: str
    steps: list[ReuseStep]

    def by_verdict(self, verdict: Verdict) -> list[ReuseStep]:
        return [s for s in self.steps if s.verdict == verdict]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Card:
    """At most five lines generated from evidence; author prose is labelled as such."""

    id: str
    kind: str
    name: str
    lines: list[str]
    status: str
    assurance: str
    scope_unknown: bool = False

    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass(frozen=True)
class FamilyPackage:
    path: Path
    family: Family
    chain: Chain
    edge_cases: list[dict[str, Any]]
    readme: str
    gen_path: Path
    gen_function: str


@dataclass(frozen=True)
class CoverageReport:
    snapshot_id: str | None
    reconstructions: int
    discovery_observed: list[int]  # in Reconstruction time order
    discovery_mean: list[float]
    discovery_low: list[int]
    discovery_high: list[int]
    discovery_permutations: int
    templates_observed: int
    singletons: int  # f1
    doubletons: int  # f2
    chao1: float
    good_turing: float | None  # None when no occurrences
    occurrences: int
    references: list[dict[str, Any]]
    unmatched_by_level: dict[str, int]
    unmatched_by_pattern: dict[str, int]
    other_patterns: int
    hw_pattern_grid: dict[str, dict[str, dict[str, int]]]
    assurance: dict[str, int]
    families_per_template: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
