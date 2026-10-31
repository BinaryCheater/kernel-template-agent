"""Effective assurance (WP24, D634): ``bounded`` and ``proved`` come only from proof records.

``declared`` and ``tested`` keep coming from StatusChanges (space verification writes
``tested``, kta.checks.full). A StatusChange that claims ``bounded`` or ``proved`` without
a proof record is ignored here: the level is read from the newest ProofSummary of the
subject (its whole-subject summary, ``part == ""``), and only while that summary is live
(not stale or rejected). When a premise changes and the summary goes stale, the subject
falls back to what the StatusChanges support, so a level can go down; the StatusChange
history itself is never rewritten.
"""

from __future__ import annotations

from typing import Any

from kta.core.ids import record_id
from kta.core.schemas import Assurance
from kta.store.sqlite import SqliteStore
from kta.store.status import ASSURANCE_RANK

DEAD = frozenset({"stale", "rejected", "deprecated", "suspect"})
EVIDENCE_LEVELS = ("declared", "tested")


def _from_status_changes(store: SqliteStore, subject: str) -> Assurance:
    best: Assurance = "declared"
    for rec in store.query("StatusChange", where={"subject": subject}, limit=1000):
        a = rec.body.get("to_assurance")
        if a in EVIDENCE_LEVELS and ASSURANCE_RANK[a] > ASSURANCE_RANK[best]:
            best = a
    return best


def proof_summary(store: SqliteStore, subject: str) -> dict[str, Any] | None:
    """The newest whole-subject ProofSummary (with its id and liveness), or None."""
    rows = store.query(
        "ProofSummary", where={"subject": subject, "part": ""}, order="-created_at", limit=1
    )
    if not rows:
        return None
    rec = rows[0]
    rid = record_id(rec)
    status = store.status(rid)[0]
    return {"id": rid, "status": status, "live": status not in DEAD, **rec.body}


def effective(store: SqliteStore, subject: str) -> Assurance:
    """The assurance a reader may rely on: max(evidence level, live proof level)."""
    base = _from_status_changes(store, subject)
    s = proof_summary(store, subject)
    if s is None or not s["live"] or s.get("level") is None:
        return base
    level: Assurance = s["level"]
    return level if ASSURANCE_RANK[level] > ASSURANCE_RANK[base] else base
