"""The one status ladder of empirical knowledge and its evidence rules (D-open-1, D446).

Empirical items (Findings, Couplings, legacy Regimes and Negatives, and later model terms
and chain performance) move along one ladder; Templates keep their assurance ladder
(22 §6) in addition. The step names are the existing ``EvidenceStatus`` values, so every
stored StatusChange stays valid; 02 §2's finding names are aliases:

| step | alias | evidence rule |
| --- | --- | --- |
| proposed | observed, claimed | fields complete, with a source (on creation) |
| reproduced | replicated | >= MIN_REPLICATIONS distinct real evidence records |
| calibrated | explained | resolvable ``explained_by``, or a Report with ``ci_low``/``ci_high`` |
| holdout_validated | — | a Report with ``data.holdout`` and ``data.agrees`` both true |

Invalidation is not a step: counter-evidence demotes one step (``promotion.demote``),
dependent records become ``stale``, and only a human rejects.

Thresholds here are defaults until the controls registry (WP21) holds them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from kta.core.schemas import Record
from kta.store.base import Store

LADDER: tuple[str, ...] = ("proposed", "reproduced", "calibrated", "holdout_validated")
ALIASES: Mapping[str, str] = {
    "observed": "proposed",
    "claimed": "proposed",
    "replicated": "reproduced",
    "explained": "calibrated",
}
RANK: Mapping[str, int] = {s: i for i, s in enumerate(LADDER)}
EMPIRICAL_KINDS = frozenset({"Finding", "Coupling", "Regime", "Negative"})
EVIDENCE_KINDS = frozenset(
    {
        "Measurement",
        "Check",
        "Profile",
        "Report",
        "ExperimentRun",
        "Reconstruction",
        "Solution",
        "Finding",
    }
)
# Default (registry knob once WP21 merges): distinct evidence records for "reproduced".
MIN_REPLICATIONS = 2


def step_name(name: str) -> str:
    return ALIASES.get(name, name)


def evidence_records(store: Store, ids: Iterable[str]) -> tuple[list[Record], list[str]]:
    recs: list[Record] = []
    problems: list[str] = []
    for rid in dict.fromkeys(ids):
        if not store.exists(rid):
            problems.append(f"evidence {rid[:12]} is not in the store")
            continue
        rec = store.get(rid)
        if rec.kind not in EVIDENCE_KINDS:
            problems.append(f"evidence {rid[:12]} is a {rec.kind}, not an evidence kind")
        elif rec.origin != "real":
            problems.append(f"evidence {rid[:12]} has origin {rec.origin}, not real")
        else:
            recs.append(rec)
    return recs, problems


def _report_data(rec: Record) -> Mapping[str, Any]:
    return rec.body.get("data", {}) if rec.kind == "Report" else {}


def _explained(store: Store, subject: Record) -> bool:
    names = subject.body.get("explained_by") or []
    # A model-term name ("HwModel:<id>#<term>") is resolved by its record part.
    ids = [str(n).split(":", 1)[-1].split("#", 1)[0] for n in names]
    return bool(ids) and all(store.exists(i) for i in ids)


def rule_problems(
    store: Store, subject: Record, to_status: str, evidence: Iterable[str]
) -> list[str]:
    """Why ``subject`` may not move to ``to_status`` with ``evidence``; [] means it may."""
    to_status = step_name(to_status)
    if to_status not in RANK or to_status == "proposed":
        return [f"{to_status!r} is not a promotion step of the ladder"]
    if subject.kind not in EMPIRICAL_KINDS:
        return [f"{subject.kind} is not empirical knowledge (templates keep assurance)"]
    cited = list(evidence)
    prior = subject.refs.get("evidence", [])
    cited += [prior] if isinstance(prior, str) else list(prior)
    recs, problems = evidence_records(store, cited)
    if to_status == "reproduced" and len(recs) < MIN_REPLICATIONS:
        problems.append(
            f"reproduced needs {MIN_REPLICATIONS} real evidence records, has {len(recs)}"
        )
    if to_status == "calibrated":
        ci = any("ci_low" in _report_data(r) and "ci_high" in _report_data(r) for r in recs)
        if not (_explained(store, subject) or ci):
            problems.append("calibrated needs a resolvable explained_by or a Report with a CI")
    if to_status == "holdout_validated":
        held = [r for r in recs if _report_data(r).get("holdout") is True]
        if not any(_report_data(r).get("agrees") is True for r in held):
            problems.append("holdout_validated needs a hold-out Report with agrees = true")
    return problems
