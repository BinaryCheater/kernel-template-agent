"""A CLI noise probe must honour the same versioned quiet policy as the orchestrator."""

from dataclasses import replace

from kta.cli.measure_cmds import _setup
from kta.core.context import Context
from kta.core.registry import build_registry, resolve
from kta.core.schemas import Actor, Record
from kta.core.schemas.hardware import Target
from kta.store import controls


def test_cli_measure_reads_stored_controls_on_each_invocation(ctx: Context) -> None:
    ctx = replace(ctx, registry=build_registry())
    store, ops = resolve("store", ctx), resolve("ops", ctx)
    target = Target(
        name="gb10",
        gpu="GB10",
        sm="sm_121",
        driver="test",
        cuda="13.0",
        ptxas="13.0",
        image="none",
    )
    rec = Record.new("Target", target, Actor(type="human", name="test"), ctx.clock.now())
    written_target = store.put(rec)
    before = controls.ensure(store, ops, ctx.clock.now())
    thresholds = {**before.quiet_thresholds, "temp_c_max": 73.0}
    changed = controls.propose(
        store,
        ops,
        {"measure.quiet_thresholds": thresholds},
        Actor(type="human", name="test"),
        "regression: a calibrated policy",
        ctx.clock.now(),
        before.version_id,
    )
    measurer, _, _, target_id = _setup(ctx)
    assert target_id == written_target
    assert measurer.thresholds == changed.quiet_thresholds
    assert measurer.thresholds["temp_c_max"] == 73.0
    assert ctx.controls.quiet_thresholds["temp_c_max"] != 73.0
