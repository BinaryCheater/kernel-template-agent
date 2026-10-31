"""Learning-loop tables and rules (docs/design/02 §8; WP20).

The tables are data in ``learning.yaml`` next to this file: the admission classes of
validator Issues, system-level features, cost tiers L0-L4 and their admission rules,
source classes, reference priority, the performance core, ablation defaults, the four
SOTA coverage levels and the O12 blind-reconstruction protocol. Code here only reads
them; an entry that is not listed counts as ``other``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from kta.core.errors import ConfigError
from kta.core.schemas.knowledge import ReferencePriority
from kta.knowledge.types import Issue

TABLE = Path(__file__).with_name("learning.yaml")
IssueClass = Literal["error", "warning", "language_gap"]
SYSTEM_GUESS = "system"  # an unmatched transaction's pattern_guess that declares it


@cache
def tables() -> dict[str, Any]:
    data = yaml.safe_load(TABLE.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ConfigError(f"{TABLE} is not a version-1 learning table")
    return cast(dict[str, Any], data)


# ---- admit -------------------------------------------------------------------------


def issue_class(issue: Issue) -> IssueClass:
    """The admission class of one validator Issue (table ``admission``, D560)."""
    for row in tables()["admission"]:
        if row["kind"] == issue.kind and re.search(str(row.get("message", "")), issue.message):
            return cast(IssueClass, row["class"])
    if issue.severity == "gap":
        return "language_gap"
    return "error" if issue.severity == "error" else "warning"


def classify(issues: Iterable[Issue]) -> list[Issue]:
    """``issues`` with expressibility gaps of the language moved to severity ``gap``."""
    return [
        Issue(i.kind, i.message, i.where, "gap") if issue_class(i) == "language_gap" else i
        for i in issues
    ]


# ---- system-level features ---------------------------------------------------------


@cache
def _feature_patterns() -> dict[str, list[re.Pattern[str]]]:
    return {
        name: [re.compile(p, re.I) for p in pats]
        for name, pats in tables()["system_features"].items()
    }


def system_feature(tx: Mapping[str, Any]) -> str | None:
    """The system-level feature an unmatched transaction is (02 §8.2), or None.

    ``pattern_guess: system`` without a matching description is ``other``.
    """
    text = str(tx.get("description", ""))
    for name, pats in _feature_patterns().items():
        if any(p.search(text) for p in pats):
            return name
    return "other" if tx.get("pattern_guess") == SYSTEM_GUESS else None


# ---- tiers, source classes, priority -----------------------------------------------


@dataclass(frozen=True)
class ReferenceFacts:
    """What the tier rules read about one reference (all from records or the caller)."""

    name: str
    target: str
    selected: bool = True
    runnable_on: tuple[str, ...] = ()
    unreproduced_steps: int = 0  # chain steps no family has reproduced
    strongest_baseline: bool = False  # strongest baseline of some in-scope case
    reproduced: bool = False  # level 2 reached (error within epsilon)
    evaluation_set: bool = False  # chosen for O12


TIER_RULES: dict[str, Callable[[ReferenceFacts], bool]] = {
    "selected": lambda f: f.selected,
    "runnable_on_target": lambda f: f.target in f.runnable_on,
    "unreproduced_step_or_strongest_baseline": lambda f: (
        f.target in f.runnable_on and (f.unreproduced_steps > 0 or f.strongest_baseline)
    ),
    "reproduced": lambda f: f.reproduced,
    "evaluation_set": lambda f: f.evaluation_set and f.reproduced,
}


def tiers(facts: ReferenceFacts) -> list[str]:
    """The cost tiers L0-L4 this reference enters (table ``tiers``, D562)."""
    out: list[str] = []
    for row in tables()["tiers"]:
        rule = TIER_RULES.get(str(row["enter"]))
        if rule is None:
            raise ConfigError(f"tier {row['id']}: unknown rule {row['enter']!r}")
        if rule(facts):
            out.append(str(row["id"]))
    return out


def source_class(name: str) -> str:
    """``name`` if the table lists it, else ``other`` (open table)."""
    return name if name in tables()["source_classes"] else "other"


def stages(name: str) -> list[str]:
    """Pipeline stages that apply to a source class (02 §8.3)."""
    return [str(s) for s in tables()["source_classes"][source_class(name)]["stages"]]


def priority(p: ReferencePriority | None) -> float:
    """cases_affected × gap_closable × uncertainty × Π(1 + boost) (02 §8.3, D564)."""
    if p is None:
        return 0.0
    boosts: dict[str, float] = tables()["priority"]["boosts"]
    value = p.cases_affected * p.gap_closable * p.uncertainty
    for reason in p.reasons:
        value *= 1.0 + float(boosts.get(reason, 0.0))
    return value


# ---- performance core --------------------------------------------------------------


@dataclass(frozen=True)
class PerformanceCore:
    """What reproduction must match (02 §8.4, D565); not a drop-in replacement."""

    reference: str
    configuration: dict[str, Any]  # one dtype / code path
    hot_loop: list[str]  # chain item keys of the hot loop and its pipeline
    host_inputs: str  # "fixed": host planners replaced by fixed inputs
    excluded: dict[str, list[str]]  # system feature -> unmatched transaction descriptions
    tolerance: str  # the reference's own correctness contract
    epsilon_rel: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def performance_core(
    reference: str,
    reconstruction: Mapping[str, Any],
    chain: Mapping[str, Any],
    configuration: Mapping[str, Any],
    tolerance: str,
) -> PerformanceCore:
    """The performance core of a decomposed reference: every chain item, with the
    system features of its unmatched transactions excluded by declaration."""
    spec = tables()["performance_core"]
    excluded: dict[str, list[str]] = {}
    for tx in reconstruction.get("unmatched_transactions", []):
        feature = system_feature(tx)
        if feature is not None:
            excluded.setdefault(feature, []).append(str(tx.get("description", ""))[:160])
    return PerformanceCore(
        reference=reference,
        configuration=dict(configuration),
        hot_loop=[str(i["key"]) for i in chain.get("items", [])],
        host_inputs=str(spec["host_inputs"]),
        excluded=excluded,
        tolerance=tolerance,
        epsilon_rel=float(spec["epsilon_rel"]),
        note=str(spec["note"]),
    )


# ---- SOTA levels and O12 -----------------------------------------------------------


@dataclass(frozen=True)
class SotaLevel:
    level: int
    name: str
    measured_by: str
    references: int  # references at this level or above
    of: int  # references considered
    pending: str | None = None  # why it cannot be counted yet
    names: list[str] = field(default_factory=lambda: list[str]())


def sota_level_rows() -> list[dict[str, Any]]:
    return [dict(r) for r in tables()["sota_levels"]]


def o12_protocol() -> dict[str, Any]:
    """The O12 blind-reconstruction protocol (01 §6, D568): defined, not run."""
    return dict(tables()["o12"])


def ablation_settings() -> dict[str, Any]:
    return dict(tables()["ablation"])
