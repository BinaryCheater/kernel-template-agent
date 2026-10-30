from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from kta.core.config import config_from_mapping
from kta.core.context import Context
from kta.core.errors import ConfigError, PermissionDenied
from kta.core.registry import Registry, build_registry, resolve


def fake_factory(ctx: Context) -> str:
    return "double"


def _src_factory() -> Any:
    # A factory whose module is inside src/kta, as real implementations are.
    from kta.core.context import SystemClock

    return SystemClock


def test_outside_src_must_declare_fixture() -> None:
    reg = Registry()
    with pytest.raises(ConfigError, match="fixture=True"):
        reg.register("clock", "manual", fake_factory)


def test_src_implementation_cannot_be_fixture() -> None:
    reg = Registry()
    with pytest.raises(ConfigError, match="never fixtures"):
        reg.register("clock", "system", _src_factory(), fixture=True)


def test_duplicate_registration_rejected() -> None:
    reg = Registry()
    reg.register("clock", "system", _src_factory())
    with pytest.raises(ConfigError, match="twice"):
        reg.register("clock", "system", _src_factory())


def test_selection_uses_config_then_single_real(ctx: Context) -> None:
    reg = Registry()
    reg.register("clock", "system", _src_factory())
    reg.register("clock", "manual", fake_factory, fixture=True)
    test_ctx = dataclasses.replace(ctx, registry=reg)
    assert reg.select("clock", ctx.config).name == "system"
    chosen = dataclasses.replace(
        test_ctx, config=config_from_mapping({"impl": {"clock": "manual"}})
    )
    assert resolve("clock", chosen) == "double"


def test_unknown_or_ambiguous_selection_fails(ctx: Context) -> None:
    reg = Registry()
    with pytest.raises(ConfigError):
        reg.select("clock", ctx.config)
    with pytest.raises(ConfigError, match="not registered"):
        reg.select("clock", config_from_mapping({"impl": {"clock": "nope"}}))


@pytest.mark.parametrize("env", ["dev", "main"])
def test_fixture_refused_in_real_envs(ctx: Context, env: str) -> None:
    reg = Registry()
    reg.register("clock", "manual", fake_factory, fixture=True)
    real_ctx = dataclasses.replace(
        ctx, env=env, registry=reg, config=config_from_mapping({"impl": {"clock": "manual"}})
    )
    with pytest.raises(PermissionDenied):
        resolve("clock", real_ctx)


def test_build_registry_only_sees_src() -> None:
    reg = build_registry()
    for interface in reg.interfaces():
        for impl in reg.implementations(interface):
            assert impl.module.startswith("kta.") and not impl.fixture
