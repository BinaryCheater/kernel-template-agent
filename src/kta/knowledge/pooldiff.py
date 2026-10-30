"""Pool diff between two PoolSnapshots (E14, D449).

What entered and left the pool, which named objects changed version, and how the status
of the members both snapshots share moved between the two snapshot times (promotions,
demotions, stale marks). Status at a time is folded from the StatusChanges written up to
that time, so the diff of two old snapshots stays the same later.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from kta.core.errors import SchemaError
from kta.core.schemas import Record
from kta.core.schemas.process import StatusChange
from kta.store.base import Store
from kta.store.status import fold


@dataclass(frozen=True)
class Entry:
    id: str
    kind: str
    label: str


@dataclass(frozen=True)
class Change:
    kind: str
    label: str
    old: str
    new: str


@dataclass(frozen=True)
class StatusMove:
    id: str
    kind: str
    label: str
    old: str
    new: str


@dataclass(frozen=True)
class PoolDiff:
    a: str
    b: str
    a_at: str
    b_at: str
    counts_a: dict[str, int]
    counts_b: dict[str, int]
    added: list[Entry] = field(default_factory=lambda: list[Entry]())
    removed: list[Entry] = field(default_factory=lambda: list[Entry]())
    changed: list[Change] = field(default_factory=lambda: list[Change]())
    status: list[StatusMove] = field(default_factory=lambda: list[StatusMove]())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def label(rec: Record) -> str:
    """A short human name; named kinds by name (``name@version`` for versioned ones)."""
    b = rec.body
    if "name" in b and "version" in b:
        return f"{b['name']}@{b['version']}"
    for key in ("name", "claim", "workload", "title", "description", "subject", "notes"):
        if b.get(key):
            text = " ".join(str(b[key]).split())
            return text if len(text) <= 80 else text[:79] + "…"
    return rec.kind


def _identity(rec: Record) -> str | None:
    """Name shared by versions of one object; None when the kind has no stable name."""
    name = rec.body.get("name")
    return f"{rec.kind}:{name}" if isinstance(name, str) else None


def _snapshot(store: Store, sid: str) -> tuple[list[str], datetime]:
    rec = store.get(sid)
    if rec.kind != "PoolSnapshot":
        raise SchemaError(f"{sid[:12]} is a {rec.kind}, not a PoolSnapshot")
    return list(rec.body["members"]), rec.created_at


def status_at(store: Store, rid: str, at: datetime) -> str:
    changes = store.query("StatusChange", {"subject": rid}, order="created_at", limit=10_000)
    upto = [StatusChange.model_validate(c.body) for c in changes if c.created_at <= at]
    return fold(upto)[0]


def diff(store: Store, a: str, b: str) -> PoolDiff:
    ma, at_a = _snapshot(store, a)
    mb, at_b = _snapshot(store, b)
    recs = {rid: store.get(rid) for rid in {*ma, *mb}}
    only_a = [r for r in ma if r not in set(mb)]
    only_b = [r for r in mb if r not in set(ma)]
    by_name_a = {k: r for r in only_a if (k := _identity(recs[r])) is not None}
    by_name_b = {k: r for r in only_b if (k := _identity(recs[r])) is not None}
    paired = set(by_name_a) & set(by_name_b)
    changed = [
        Change(recs[by_name_b[k]].kind, label(recs[by_name_b[k]]), by_name_a[k], by_name_b[k])
        for k in sorted(paired)
    ]
    moved = {by_name_a[k] for k in paired} | {by_name_b[k] for k in paired}
    status: list[StatusMove] = []
    for rid in sorted(set(ma) & set(mb)):
        old, new = status_at(store, rid, at_a), status_at(store, rid, at_b)
        if old != new:
            status.append(StatusMove(rid, recs[rid].kind, label(recs[rid]), old, new))

    def entries(ids: list[str]) -> list[Entry]:
        out = [Entry(r, recs[r].kind, label(recs[r])) for r in ids if r not in moved]
        return sorted(out, key=lambda e: (e.kind, e.label, e.id))

    return PoolDiff(
        a=a,
        b=b,
        a_at=at_a.isoformat(),
        b_at=at_b.isoformat(),
        counts_a=dict(Counter(recs[r].kind for r in ma)),
        counts_b=dict(Counter(recs[r].kind for r in mb)),
        added=entries(only_b),
        removed=entries(only_a),
        changed=changed,
        status=status,
    )
