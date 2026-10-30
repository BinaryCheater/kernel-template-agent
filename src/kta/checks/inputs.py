"""Input distributions of the checkers: the shared base table plus adversarial ones.

Open table (AGENTS.md principle 1): a Contract or a numeric risk may name any
distribution; one missing here raises NotSupported rather than being skipped.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kta.measure.inputs import BASE, Generator


def _cancellation(d: tuple[int, ...], dtype: str, g: np.random.Generator) -> np.ndarray[Any, Any]:
    """Neighbouring pairs nearly cancel: sums lose most significant bits (contractions)."""
    x = g.standard_normal(d) * 8.0
    flat = x.reshape(-1)
    n = flat.size - flat.size % 2
    flat[1:n:2] = -flat[0:n:2] + g.standard_normal(n // 2) * 1e-2
    return x


def _dominant_tail(d: tuple[int, ...], dtype: str, g: np.random.Generator) -> np.ndarray[Any, Any]:
    """The last entry along the last axis dominates (e.g. attention's last token)."""
    x = g.standard_normal(d)
    if len(d) > 0 and d[-1] > 0:
        x[..., -1] = 8.0
    return x


TABLE: dict[str, Generator] = {
    **BASE,
    "cancellation": _cancellation,
    "dominant_tail": _dominant_tail,
}
