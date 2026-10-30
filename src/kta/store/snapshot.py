"""Pool snapshots (docs/design/20 §5): a PoolSnapshot record plus a CAS on ``current_pool``."""

from __future__ import annotations

from datetime import datetime

from kta.core.errors import KtaError
from kta.core.schemas import Actor, Record, RefValue
from kta.core.schemas.process import PoolSnapshot
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps

CURRENT_POOL = "current_pool"


class SnapshotConflict(KtaError):
    """``current_pool`` moved since the parent snapshot was read."""


def snapshot(
    store: SqliteStore,
    ops: SqliteOps,
    members: list[str],
    parent: str | None,
    note: str,
    actor: Actor,
    now: datetime,
) -> str:
    """Write a snapshot of ``members`` on top of ``parent`` and make it current.

    The record is written first (records are immutable and harmless if unused); the
    pointer moves only if it still points at ``parent``. On a conflict the caller
    re-reads the current pool and decides; nothing is merged silently.
    """
    refs: dict[str, RefValue] = {"members": sorted(set(members))}
    if parent is not None:
        refs["previous"] = parent
    body = PoolSnapshot(members=sorted(set(members)), note=note)
    sid = store.put(Record.new("PoolSnapshot", body, actor, now, refs=refs))
    if not ops.set_pointer(CURRENT_POOL, parent, sid):
        raise SnapshotConflict(
            f"current_pool is {ops.get_pointer(CURRENT_POOL)}, expected {parent}; "
            f"snapshot {sid[:12]} was written but not made current"
        )
    return sid
