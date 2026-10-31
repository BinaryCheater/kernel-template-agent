"""Checker models: the open table a ProofOutline names per obligation.

Each model turns one obligation of a Template into SMT queries or a bounded enumeration
and replays every witness. ``GROUPS`` lists the premise groups a model's result rests on
(paper §4.4: constraints kept as named groups, so a result depends only on what it uses);
C is absent from most, so strengthening the constraints keeps those certificates.
"""

from __future__ import annotations

from .banks import bank_map, queue_progress
from .base import ModelFn, ModelInput
from .order import async_readiness, event_order, resident_capacity
from .partition import partition_index
from .slots import slot_progress, slot_rotation
from .tiles import tile_index

MODELS: dict[str, ModelFn] = {
    "tile_index": tile_index,
    "slot_rotation": slot_rotation,
    "slot_progress": slot_progress,
    "partition_index": partition_index,
    "event_order": event_order,
    "resident_capacity": resident_capacity,
    "async_readiness": async_readiness,
    "bank_map": bank_map,
    "queue_progress": queue_progress,
}

_K2 = ("gamma", "assumes", "X", "pre", "program")
GROUPS: dict[str, tuple[str, ...]] = {name: _K2 for name in MODELS}
GROUPS["resident_capacity"] = ("gamma", "assumes", "X", "C")

__all__ = ["GROUPS", "MODELS", "ModelFn", "ModelInput"]
