"""Promotion and demotion of empirical knowledge (02 §2 "Promotion and demotion", D448).

Raw evidence is immutable; only StatusChange records and Reports are written.

- ``promote``: a checker moves a Finding or Coupling up the ladder when the step's
  evidence rule (``ladder.rule_problems``) holds; a Report ``promotion`` records the
  rule and the evidence, and the StatusChange cites it.
- ``demote``: counter-evidence moves the item one step down (actor ``system``, the same
  actor that marks records stale) and invalidation propagates along dependency edges
  (``kta.store.stale.propagate``) so everything resting on it is re-verified. At the
  bottom of the ladder the counter-evidence is recorded and the item waits for a human;
  only a human rejects.
- ``review``: a deterministic scan that proposes promotions out of the finding layer
  (into the hardware model, into a Template's Γ or a Structure's regime, repeated
  ``other`` into a new Pattern or Template). It writes one Report ``promotion_review``
  and changes nothing else: those promotions are new versions written by a worker or
  a human.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

from kta.core.errors import PermissionDenied
from kta.core.schemas import Actor, EvidenceStatus, Record, RefValue
from kta.core.schemas.process import Report, StatusChange
from kta.knowledge import findings as fnd
from kta.knowledge import ladder, pool
from kta.store.sqlite import SqliteStore
from kta.store.stale import propagate

CHECKER = Actor(type="checker", name="knowledge.promotion")
SYSTEM = Actor(type="system", name="knowledge.promotion")
# Defaults (registry knobs once WP21 merges).
MIN_TARGETS_FOR_SCOPE = 2  # a finding holding on this many Targets is proposed for Γ/regime
MIN_REPEATS_FOR_PATTERN = 3  # gaps naming the same missing pattern/template guess


@dataclass(frozen=True)
class Demotion:
    subject: str
    from_status: str
    to_status: str | None  # None: at the bottom, recorded for a human
    report: str
    stale: list[str] = field(default_factory=lambda: list[str]())


def _report(
    store: SqliteStore,
    title: str,
    subject: str,
    summary: str,
    data: dict[str, Any],
    refs: dict[str, RefValue],
    now: datetime,
    actor: Actor,
) -> str:
    body = Report(title=title, subject=subject, summary=summary, data=data)
    return store.put(Record.new("Report", body, actor, now, refs=refs))


def promote(
    store: SqliteStore, subject: str, to_status: str, evidence: list[str], now: datetime
) -> str:
    """Promote ``subject`` one or more steps; returns the StatusChange ID.

    Raises PermissionDenied when the evidence rule does not hold (the problems are in
    the message) or the store's status rules refuse the move.
    """
    target = ladder.step_name(to_status)
    rec = store.get(subject)
    problems = ladder.rule_problems(store, rec, target, evidence)
    if problems:
        raise PermissionDenied(f"cannot promote {subject[:12]} to {target}: " + "; ".join(problems))
    current = store.status(subject)[0]
    rid = _report(
        store,
        "promotion",
        subject,
        f"{current} -> {target}",
        {"from": current, "to": target, "evidence": sorted(set(evidence))},
        {"evidence": sorted(set(evidence))} if evidence else {},
        now,
        CHECKER,
    )
    change = StatusChange(
        subject=subject,
        to_status=cast(EvidenceStatus, target),
        reason=f"evidence rule held (report {rid[:12]})",
    )
    return store.put(Record.new("StatusChange", change, CHECKER, now, refs={"report": rid}))


def demote(
    store: SqliteStore, subject: str, counter_evidence: list[str], reason: str, now: datetime
) -> Demotion:
    """One step down on counter-evidence, then stale propagation to dependents."""
    _, problems = ladder.evidence_records(store, counter_evidence)
    if problems or not counter_evidence:
        raise PermissionDenied(
            f"demotion of {subject[:12]} needs real counter-evidence: "
            + ("; ".join(problems) or "none given")
        )
    current = store.status(subject)[0]
    rank = ladder.RANK.get(current)
    lower = ladder.LADDER[rank - 1] if rank else None
    rid = _report(
        store,
        "demotion",
        subject,
        f"{current} -> {lower or 'needs a human'}: {reason}",
        {
            "from": current,
            "to": lower,
            "reason": reason,
            "counter_evidence": sorted(set(counter_evidence)),
        },
        {"evidence": sorted(set(counter_evidence))},
        now,
        SYSTEM,
    )
    if lower is None:
        return Demotion(subject, current, None, rid)
    change = StatusChange(
        subject=subject,
        to_status=cast(EvidenceStatus, lower),
        reason=f"counter-evidence (report {rid[:12]}): {reason}",
    )
    store.put(Record.new("StatusChange", change, SYSTEM, now, refs={"report": rid}))
    stale = propagate(store, subject, f"demoted: {reason}", now)
    return Demotion(subject, current, lower, rid, stale)


def _into_model(store: SqliteStore) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in fnd.findings(store):
        model_terms = [e for e in f.body.explained_by if e.startswith("HwModel:")]
        if model_terms and ladder.RANK.get(f.status, -1) >= ladder.RANK["holdout_validated"]:
            out.append({"finding": f.id, "into": "hardware_model", "terms": model_terms})
    return out


def _into_scope(store: SqliteStore) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, bool], set[str]] = defaultdict(set)
    members: dict[tuple[str, str, bool], list[str]] = defaultdict(list)
    for f in fnd.findings(store):
        if f.body.subject is None or ladder.RANK.get(f.status, -1) < ladder.RANK["reproduced"]:
            continue
        key = (f.body.subject, f.body.kind, f.body.holds)
        groups[key] |= set(f.body.scope.targets)
        members[key].append(f.id)
    out: list[dict[str, Any]] = []
    for (subject, kind, holds), targets in sorted(groups.items()):
        if len(targets) >= MIN_TARGETS_FOR_SCOPE:
            into = "structure_regime" if kind == "regime" else "template_scope"
            out.append(
                {
                    "subject": subject,
                    "kind": kind,
                    "holds": holds,
                    "into": into,
                    "targets": sorted(targets),
                    "findings": members[(subject, kind, holds)],
                }
            )
    return out


def _new_patterns(store: SqliteStore) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for rec in pool.records(store, "Gap"):
        if rec.body.get("missing") in ("pattern", "template", "other"):
            counts[str(rec.body.get("pattern_guess") or "other")] += 1
    return [
        {"pattern_guess": g, "gaps": n, "into": "new_pattern_or_template"}
        for g, n in sorted(counts.items())
        if n >= MIN_REPEATS_FOR_PATTERN
    ]


def review(store: SqliteStore, now: datetime) -> str:
    """Write a Report ``promotion_review`` with every promotion candidate; returns its ID."""
    data = {
        "into_model": _into_model(store),
        "into_scope": _into_scope(store),
        "new_patterns": _new_patterns(store),
        "thresholds": {
            "min_targets_for_scope": MIN_TARGETS_FOR_SCOPE,
            "min_repeats_for_pattern": MIN_REPEATS_FOR_PATTERN,
        },
    }
    n = sum(len(v) for k, v in data.items() if k != "thresholds")
    return _report(store, "promotion_review", "pool", f"{n} candidate(s)", data, {}, now, SYSTEM)
