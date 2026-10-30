from __future__ import annotations

from pathlib import Path

import pytest

from fakes.clock import ManualClock
from kta.core.config import config_from_mapping
from kta.core.context import Context
from kta.core.registry import Registry
from kta.store import CodeState, SqliteOps, SqliteStore, open_ops, open_store
from storehelp import TEST_COMMIT


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


@pytest.fixture
def store(ctx: Context) -> SqliteStore:
    return open_store(ctx, CodeState(TEST_COMMIT, False))


@pytest.fixture
def ops(ctx: Context) -> SqliteOps:
    return open_ops(ctx)
