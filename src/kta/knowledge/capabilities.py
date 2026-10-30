"""Capabilities a Target provides, and whether a Template can run there (D-T8, D-T27).

A Template ``requires`` abstract Capabilities. A Target's support of a Capability comes
from, in order:

1. the newest CapabilityDecl record for (target, capability) (documented or probed);
2. otherwise, inferred from the ISA features that realise it (``Capability.realised_by``):
   ``yes`` when a HwFeature probe says yes for any of them, ``no`` when every one says no;
3. otherwise ``unknown``.

A name in ``requires`` that is no Capability but a legacy HwFeature (records written
before D442) is answered by that HwFeature's ``support`` table directly.

Only ``yes`` makes a Template available. Unavailable Templates are not dropped from the
pool: retrieval filters them and coverage counts them as "known, unavailable here"
(``no``) or "unconfirmed here" (``unknown``), D445.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from kta.core.schemas.hardware import Capability, CapabilityDecl, HwFeature
from kta.knowledge import pool
from kta.store.base import Store

Support = Literal["yes", "no", "unknown"]


@dataclass(frozen=True)
class CapabilityState:
    name: str
    status: Support
    basis: str  # documented | probed | inferred | legacy_hw_feature | none
    params: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    record: str | None = None  # the deciding record, when there is one


@dataclass(frozen=True)
class Availability:
    available: bool
    missing: list[str]  # required names whose status is "no"
    unconfirmed: list[str]  # required names whose status is "unknown"


def capabilities(store: Store) -> dict[str, tuple[str, Capability]]:
    """Capability name -> (record ID, body) of the newest record per name."""
    out: dict[str, tuple[str, Capability]] = {}
    for rid, rec in pool.with_ids(pool.records(store, "Capability", order="created_at")):
        out[str(rec.body["name"])] = (rid, Capability.model_validate(rec.body))
    return out


def _decls(store: Store, target: str) -> dict[str, tuple[str, CapabilityDecl]]:
    out: dict[str, tuple[str, CapabilityDecl]] = {}
    recs = pool.records(store, "CapabilityDecl", {"target": target}, order="created_at")
    for rid, rec in pool.with_ids(recs):
        out[str(rec.body["capability"])] = (rid, CapabilityDecl.model_validate(rec.body))
    return out


def _features(store: Store) -> dict[str, tuple[str, HwFeature]]:
    """HwFeature name -> newest record; probe results are written after the catalog."""
    out: dict[str, tuple[str, HwFeature]] = {}
    for rid, rec in pool.with_ids(pool.records(store, "HwFeature", order="created_at")):
        out[str(rec.body["name"])] = (rid, HwFeature.model_validate(rec.body))
    return out


def _inferred(
    name: str, cap: Capability, features: dict[str, tuple[str, HwFeature]], target: str
) -> CapabilityState:
    seen = [features[f][1].support.get(target, "unknown") for f in cap.realised_by if f in features]
    if "yes" in seen:
        return CapabilityState(name, "yes", "inferred")
    if seen and all(s == "no" for s in seen) and len(seen) == len(cap.realised_by):
        return CapabilityState(name, "no", "inferred")
    return CapabilityState(name, "unknown", "none")


def target_capabilities(
    store: Store, target: str, names: Iterable[str] | None = None
) -> dict[str, CapabilityState]:
    """Support of each Capability (default: every known one, plus ``names``) on ``target``."""
    caps = capabilities(store)
    decls = _decls(store, target)
    features = _features(store)
    wanted = list(dict.fromkeys([*caps, *(names or [])]))
    out: dict[str, CapabilityState] = {}
    for name in wanted:
        if name in decls:
            rid, d = decls[name]
            out[name] = CapabilityState(name, d.status, d.basis, dict(d.params), rid)
        elif name in caps:
            out[name] = _inferred(name, caps[name][1], features, target)
        elif name in features:
            rid, f = features[name]
            out[name] = CapabilityState(
                name, f.support.get(target, "unknown"), "legacy_hw_feature", {}, rid
            )
        else:
            out[name] = CapabilityState(name, "unknown", "none")
    return out


def availability(
    requires: Iterable[str], target: str | None, states: dict[str, CapabilityState]
) -> Availability:
    """Whether everything in ``requires`` is ``yes``; no target means nothing is confirmed."""
    needed = list(dict.fromkeys(requires))
    if target is None:
        return Availability(not needed, [], needed)
    missing = [n for n in needed if n in states and states[n].status == "no"]
    unconfirmed = [n for n in needed if n not in states or states[n].status == "unknown"]
    return Availability(not missing and not unconfirmed, missing, unconfirmed)


def template_availability(
    store: Store, requires: Iterable[str], target: str | None
) -> Availability:
    needed = list(requires)
    if target is None or not needed:
        return availability(needed, target, {})
    return availability(needed, target, target_capabilities(store, target, needed))
