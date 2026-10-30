from __future__ import annotations

import pytest

from fakes.clock import ManualClock
from kta.core.errors import ConfigError
from kta.core.ids import UlidFactory
from kta.core.trace import (
    TRACE_PARENT_VAR,
    child_environ,
    current_span_id,
    new_tracer,
    span,
    tracing,
)
from kta.store.sqlite_ops import SqliteOps


def test_spans_nest_and_close_in_the_ops_table(ops: SqliteOps, clock: ManualClock) -> None:
    tracer = new_tracer(ops, clock, UlidFactory(clock), {})
    with tracing(tracer):
        assert current_span_id() is None
        with span("loop_step", "solve T1", "process:orchestrator", task_id="t1") as outer:
            clock.advance(1)
            with span("compile", "nvcc gemm@3", "worker:spark:gpu") as inner:
                inner.set(bytes=123)
                assert current_span_id() == inner.span_id
                assert child_environ() == {TRACE_PARENT_VAR: f"{tracer.trace_id}:{inner.span_id}"}
                clock.advance(2)
            assert current_span_id() == outer.span_id
    rows = {r.name: r for r in ops.spans(trace_id=tracer.trace_id)}
    outer_row, inner_row = rows["solve T1"], rows["nvcc gemm@3"]
    assert inner_row.parent_id == outer_row.span_id and outer_row.parent_id is None
    assert outer_row.status == inner_row.status == "ok"
    assert inner_row.attrs == {"bytes": 123}
    assert outer_row.attrs == {"task_id": "t1"}
    assert outer_row.end_ts is not None and inner_row.end_ts is not None
    assert (outer_row.end_ts - outer_row.start_ts).total_seconds() == 3


def test_failed_span_records_status_and_error(ops: SqliteOps, clock: ManualClock) -> None:
    tracer = new_tracer(ops, clock, UlidFactory(clock), {})
    with tracing(tracer), pytest.raises(RuntimeError), span("check", "numeric", "gpu:spark"):
        raise RuntimeError("boom")
    (row,) = ops.spans(trace_id=tracer.trace_id)
    assert row.status == "fail" and "boom" in row.attrs["error"]


def test_child_process_continues_the_parent_trace(ops: SqliteOps, clock: ManualClock) -> None:
    env = {TRACE_PARENT_VAR: "01TRACE:01PARENT"}
    tracer = new_tracer(ops, clock, UlidFactory(clock), env)
    with tracing(tracer), span("task", "kta measure", "process:kta"):
        pass
    (row,) = ops.spans(trace_id="01TRACE")
    assert row.parent_id == "01PARENT"


def test_span_without_tracer_or_with_bad_lane_fails(ops: SqliteOps, clock: ManualClock) -> None:
    with pytest.raises(ConfigError, match="tracer"), span("task", "x", "process:kta"):
        pass
    tracer = new_tracer(ops, clock, UlidFactory(clock), {})
    with tracing(tracer):
        with pytest.raises(ConfigError, match="lane"), span("task", "x", "somewhere"):
            pass
        with pytest.raises(ConfigError, match="kind"), span("Bad Kind", "x", "human"):
            pass
    with pytest.raises(ConfigError, match=TRACE_PARENT_VAR):
        new_tracer(ops, clock, UlidFactory(clock), {TRACE_PARENT_VAR: "no-colon"})
