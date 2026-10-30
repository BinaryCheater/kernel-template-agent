"""Hardware units of GB10 (the ``Unit`` names used by Templates, models and HwFeatures).

An open list: a resource that fits none of these is recorded as ``other`` and counted.
"""

from __future__ import annotations

UNITS: tuple[str, ...] = (
    "dram",  # LPDDR5X, unified with the CPU
    "l2",
    "smem",
    "lsu",
    "tma",
    "tensor",
    "fma",
    "sfu",
    "issue",
    "launch",
    "other",
)
