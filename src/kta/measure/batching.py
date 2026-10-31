"""Measurement batching windows (11 §4.2, WP23 D608).

Draining a unified-memory host for an exclusive measurement costs the wait for every
running job (on this and on other control planes) to finish. A window pays that drain
once: ``MeasureWindow.run`` takes the Executor's quiet lease on one GPU, which also holds
the host lock exclusive on that worker, and runs every item inside it. Jobs the items
submit from this thread carry the lease holder, so their gpu-exclusive jobs run in the
window without asking again. Which items form a window is the orchestrator's choice
(``kta.orchestrator.stagequeue.plan``: ``measure.batch_window`` max_items and window_s).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Protocol, TypeVar

log = logging.getLogger("kta.measure.batching")
T = TypeVar("T")
R = TypeVar("R")
WINDOW_MAX_S = 600  # the Executor's lease maximum (11 §5)


class Leasing(Protocol):
    def quiet_lease(
        self, worker: str, device: str, holder: str, max_s: int = 600
    ) -> AbstractContextManager[None]: ...


class MeasureWindow:
    def __init__(self, ex: Leasing, worker: str, device: str, holder: str) -> None:
        self.ex, self.worker, self.device, self.holder = ex, worker, device, holder

    def run(self, items: Sequence[T], fn: Callable[[T], R]) -> list[R]:
        if not items:
            return []
        with self.ex.quiet_lease(self.worker, self.device, self.holder, WINDOW_MAX_S):
            log.info(
                "measurement window %s on %s:%s: %d items",
                self.holder,
                self.worker,
                self.device,
                len(items),
            )
            return [fn(item) for item in items]
