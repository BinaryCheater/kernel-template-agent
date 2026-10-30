"""Versions of the controls registry's values (WP21, D481).

A version is a ControlsVersion record; the ops pointer ``controls:current`` names the one
in force (compare-and-set, so two writers cannot both win). ``current`` only reads.
``ensure`` is called where a run starts: it writes version 1 from the definitions'
initial values when none exists, and a system version when the definitions gained or
lost knobs, so the version a run records always holds every value it could read.
``propose`` is the settings page's write: a human author, a reason, the version the
author read. Records are never rewritten; a new version names its predecessor in
``refs.previous``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from kta.core.controls import Change, Controls, baseline, definitions, upgrade
from kta.core.errors import ConfigError, PermissionDenied
from kta.core.ids import record_id
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import ControlsVersion
from kta.store.base import Ops, Store

POINTER = "controls:current"
SYSTEM = Actor(type="system", name="kta.store.controls")


def _read(store: Store, ops: Ops) -> tuple[Record | None, str | None]:
    rid = ops.get_pointer(POINTER)
    return (None, None) if rid is None else (store.get(rid), rid)


def current(store: Store, ops: Ops) -> Controls:
    """The values in force; the definitions' initial values when nothing is stored yet.

    ``version_id`` is None when nothing is stored or the stored version lacks knobs the
    definitions now have (``ensure`` writes the upgrade).
    """
    rec, rid = _read(store, ops)
    if rec is None:
        return baseline()
    body = ControlsVersion.model_validate(rec.body)
    values, added, dropped = upgrade(body.values)
    stored = None if added or dropped else rid
    return Controls(values, body.version, stored, body.definitions_hash)


def _write(
    store: Store,
    ops: Ops,
    base: Controls,
    new: Mapping[str, Any],
    actor: Actor,
    reason: str,
    at: datetime,
) -> Controls:
    previous = ops.get_pointer(POINTER)
    old = Controls(base.values) if previous is not None else Controls({})
    changes = [c.to_dict() for c in old.diff(Controls(dict(new)))]
    version = base.version + 1 if previous is not None else 1
    body = ControlsVersion(
        version=version,
        definitions_hash=definitions().hash,
        values=dict(new),
        changes=changes,
        reason=reason,
    )
    refs: dict[str, str | list[str]] = {"previous": previous} if previous else {}
    rid = store.put(Record.new("ControlsVersion", body, actor, at, refs=refs))
    if not ops.set_pointer(POINTER, previous, rid):
        raise ConfigError("the controls changed concurrently; read the current version again")
    return Controls(dict(new), version, rid, body.definitions_hash)


def ensure(store: Store, ops: Ops, at: datetime) -> Controls:
    """The values in force, stored: writes version 1 or a definitions upgrade if needed."""
    snap = current(store, ops)
    if snap.version_id is not None:
        return snap
    rec, _ = _read(store, ops)
    if rec is None:
        reason = "version 1: initial values of the registry definitions"
    else:
        _, added, dropped = upgrade(rec.body["values"])
        reason = (
            f"registry definitions changed: added {added or 'none'}, dropped {dropped or 'none'}"
        )
    try:
        return _write(store, ops, snap, snap.values, SYSTEM, reason, at)
    except ConfigError:
        return current(store, ops)  # another process wrote the same upgrade first


def propose(
    store: Store,
    ops: Ops,
    changes: Mapping[str, Any],
    author: Actor,
    reason: str,
    at: datetime,
    expected: str | None,
) -> Controls:
    """Write a new version with ``changes`` (knob id -> value) by a human ``author``.

    ``expected`` is the version the author read; a different current version is refused,
    so nobody overwrites a change they have not seen. Invariant knobs are refused
    (PermissionDenied), invalid values and no-op proposals are ConfigError.
    """
    if author.type != "human" or not author.name.strip():
        raise PermissionDenied("a controls version is written by a named human")
    if not reason.strip():
        raise ConfigError("a controls version needs a reason")
    if not changes:
        raise ConfigError("no changes given")
    base = ensure(store, ops, at)
    if expected != base.version_id:
        raise ConfigError(
            f"the current controls version is {base.version_id} (v{base.version}), "
            f"not {expected}: reload and propose again"
        )
    new = base.with_changes(changes)
    if not base.diff(new):
        raise ConfigError("the proposed values equal the current ones")
    return _write(store, ops, base, new.values, author, reason.strip(), at)


def history(store: Store, limit: int = 100) -> list[tuple[str, Record]]:
    """Stored versions, newest first, with their record IDs."""
    recs = store.query("ControlsVersion", order="-created_at", limit=limit)
    return sorted(((record_id(r), r) for r in recs), key=lambda p: -int(p[1].body["version"]))


def changes_of(rec: Record) -> list[Change]:
    return [Change(str(c["knob"]), c.get("old"), c.get("new")) for c in rec.body["changes"]]
