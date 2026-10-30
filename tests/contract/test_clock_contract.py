from __future__ import annotations

from collections.abc import Callable

import pytest

from fakes.clock import ManualClock
from kta.core.context import Clock, SystemClock

CLOCKS: list[Callable[[], Clock]] = [SystemClock, ManualClock]


@pytest.mark.contract
@pytest.mark.parametrize("make", CLOCKS, ids=lambda f: f.__name__)
def test_clock_is_utc_and_monotone(make: Callable[[], Clock]) -> None:
    clock = make()
    first = clock.now()
    second = clock.now()
    offset = first.utcoffset()
    assert offset is not None and offset.total_seconds() == 0
    assert second >= first
