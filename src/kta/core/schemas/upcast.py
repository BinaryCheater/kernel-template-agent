"""Upcasters: stored bodies of an older schema version -> the current shape (D440).

Records are immutable and their ID is the hash of the stored JSON, so a schema change
never rewrites stored records. ``Record`` keeps the stored body (for hashing and
transfer) and exposes the upcast body; readers see one shape. Each function maps
version n to n + 1 of one kind; ``KindSpec.upcasters[n - 1]`` is the n -> n + 1 step.

Data a newer shape no longer holds stays in the stored body: a v1 Template's
``program.fragments`` and ``expected_sass`` are read back as a legacy Realisation by
``kta.knowledge.realisations`` from ``Record.stored_body``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from kta.core.schemas.effects import upcast_template_body
from kta.core.schemas.levels import map_level

Upcaster = Callable[[dict[str, Any]], dict[str, Any]]


def _level(value: Any) -> Any:
    return map_level(value) if isinstance(value, int) else value


def template_v1(body: dict[str, Any]) -> dict[str, Any]:
    """WP14: levels 0-5 -> 1-7; requires_hw -> requires; code moves to Realisations."""
    out = copy.deepcopy(body)
    out["level_from"] = _level(out.get("level_from"))
    out["level_to"] = _level(out.get("level_to"))
    out["requires"] = out.pop("requires_hw", [])
    out.setdefault("probed_on", [])
    out.pop("expected_sass", None)
    program = dict(out.get("program") or {})
    program.pop("fragments", None)
    out["program"] = program
    return out


def pattern_v1(body: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(body)
    out["level"] = _level(out.get("level"))
    return out


def gap_v1(body: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(body)
    if out.get("level") is not None:
        out["level"] = _level(out["level"])
    return out


def reconstruction_v1(body: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(body)
    txs: list[dict[str, Any]] = out.get("unmatched_transactions") or []
    for tx in txs:
        tx["level"] = _level(tx.get("level"))
    return out


def template_v2_effects(body: dict[str, Any]) -> dict[str, Any]:
    """Template 2 -> 3 (WP15, D461): schema-1 ``cost`` becomes ``effects``."""
    return upcast_template_body(body)


def identity(body: dict[str, Any]) -> dict[str, Any]:
    """A version step that only added optional fields (HwModel 2, Prediction 2; D461)."""
    return body
