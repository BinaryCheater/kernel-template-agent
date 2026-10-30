from __future__ import annotations

from pathlib import Path

import pytest

from fakes.clock import ManualClock
from kta.core.config import config_from_mapping
from kta.core.context import Context
from kta.core.registry import Registry


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def ctx(tmp_path: Path, clock: ManualClock) -> Context:
    return Context(
        env="test",
        config=config_from_mapping({}),
        data_dir=tmp_path / "data",
        registry=Registry(),
        clock=clock,
    )
