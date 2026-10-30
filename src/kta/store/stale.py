"""Stale propagation along the dependency graph (docs/design/20 §6).

When a record is rejected, or a Target, HwModel, Contract or checker version gets a
new record, everything that depends on the old one (reverse refs, transitively) is
marked stale by a system StatusChange and needs re-verification. Who calls this is
the orchestrator's decision; the store only walks the graph (decision D150).
"""

from __future__ import annotations

from collections import deque
from datetime import datetime

from kta.core.schemas import Actor, Record
from kta.core.schemas.process import StatusChange
from kta.store.sqlite import SqliteStore

SYSTEM = Actor(type="system", name="stale.propagate")
# A dependent with one of these statuses keeps it: stale must not hide a human verdict.
KEEP = frozenset({"rejected", "deprecated", "stale"})


def dependents(store: SqliteStore, id: str) -> list[str]:
    """Records that reference ``id`` directly or indirectly, excluding StatusChanges."""
    seen: set[str] = {id}
    order: list[str] = []
    frontier = deque([id])
    while frontier:
        node = frontier.popleft()
        for src in store.neighbors(node, direction="in"):
            if src in seen:
                continue
            seen.add(src)
            if store.kind_of(src) == "StatusChange":
                continue
            order.append(src)
            frontier.append(src)
    return order


def propagate(store: SqliteStore, id: str, reason: str, now: datetime) -> list[str]:
    """Mark every dependent of ``id`` stale; returns the IDs that need re-verification."""
    marked: list[str] = []
    changes: list[Record] = []
    for dep in dependents(store, id):
        if store.status(dep)[0] in KEEP:
            continue
        body = StatusChange(subject=dep, to_status="stale", reason=f"{reason} (via {id[:12]})")
        changes.append(Record.new("StatusChange", body, SYSTEM, now))
        marked.append(dep)
    store.put_many(changes)
    return marked


def reject(store: SqliteStore, id: str, reason: str, human: Actor, now: datetime) -> list[str]:
    """A human rejects ``id``; its dependents become stale."""
    body = StatusChange(subject=id, to_status="rejected", reason=reason)
    store.put(Record.new("StatusChange", body, human, now))
    return propagate(store, id, f"rejected: {reason}", now)
