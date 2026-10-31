"""Evaluation pause, resume, failure visibility and partial progress are system contracts."""

from kta.core.context import Context
from kta.orchestrator import control, step
from kta.orchestrator.pipeline import Services
from kta.orchestrator.solve_evaluate import checkpoint, each_top
from kta.orchestrator.transition import Loop
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps


def test_paused_evaluation_keeps_completed_results_and_launches_nothing(
    ctx: Context,
    store: SqliteStore,
    ops: SqliteOps,
) -> None:
    svc = Services(ctx, store, ops)
    data = {"top": {"s0": ["done", "pending"]}, "checks": {"done": "result"}}
    ops.put_loop("L", "solve", "case", "checking", data)
    control.set_paused(ops, "loop:L", True, "inspect", "test", ctx.clock.now())
    loop = Loop("L", "solve", "case", "checking", data)

    def must_not_run(*args: object) -> str:
        raise AssertionError("paused evaluation launched work")

    assert each_top(svc, loop, must_not_run, "checks") == {"done": "result"}
    held = step(
        Loop(
            "L",
            "solve",
            "case",
            "checking",
            data,
            {"checks": {"done": "result"}, "failed": [], "pending": True},
        ),
        ctx,
    )
    assert held.to is None and held.data["checks"] == {"done": "result"}
    assert "checkpointed" in held.note
    # A filtered measurement view must not overwrite the full selected set in ops.
    filtered = Loop("L", "solve", "case", "checking", {**data, "top": {"s0": ["done"]}})
    checkpoint(svc, filtered, measurements={"done": "measured"})
    saved = ops.get_loop("L")["data"]
    assert saved["top"] == data["top"] and saved["checks"] == data["checks"]
    assert saved["measurements"] == {"done": "measured"}
    control.set_paused(ops, "loop:L", False, "resume", "test", ctx.clock.now())
    complete = Loop(
        "L", "solve", "case", "checking", {"top": {"s0": ["done"]}, "checks": {"done": "result"}}
    )
    assert each_top(svc, complete, must_not_run, "checks") == {"done": "result"}


def test_blocking_an_inflight_step_retains_newer_checkpoints(
    ctx: Context,
    store: SqliteStore,
    ops: SqliteOps,
) -> None:
    from kta.core.errors import SchemaError
    from kta.orchestrator.runner import Runner

    svc = Services(ctx, store, ops)
    data = {"top": {"s0": ["done", "next"]}}
    ops.put_loop("L", "solve", "case", "checking", data)
    old = Loop("L", "solve", "case", "checking", data)
    checkpoint(svc, old, checks={"done": "result"})
    Runner(svc, "test")._block(old, SchemaError("broken metadata"))  # pyright: ignore[reportPrivateUsage]
    saved = ops.get_loop("L")["data"]
    assert saved["checks"] == {"done": "result"}
    assert saved["blocked"]["error"] == "SchemaError"
