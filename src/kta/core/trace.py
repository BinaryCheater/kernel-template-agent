"""Spans: start, end, parent/child links, contextvar propagation (spec/conventions.md §4.1).

``span()`` takes no Context, so the destination comes from a bound ``Tracer``
(``with tracing(tracer): ...``). Using ``span()`` with no tracer bound is an error: a
system event that silently goes nowhere would hide work from the timeline.

``KTA_TRACE_PARENT`` passes the parent to child ``kta`` processes as
``<trace_id>:<span_id>`` (decision D148).
"""

from __future__ import annotations

import re
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol

from kta.core.errors import ConfigError
from kta.core.ids import UlidFactory

SpanStatus = Literal["ok", "fail", "cancelled", "running"]
TRACE_PARENT_VAR = "KTA_TRACE_PARENT"
# Open table of kinds (lower_snake_case); lanes use the prefixes of conventions §4.1.
KIND_RE = re.compile(r"^[a-z][a-z0-9_]*$")
LANE_RE = re.compile(r"^(?:(?:process|agent|gpu):\S+|worker:\S+:\S+|human)$")


@dataclass(frozen=True)
class SpanStart:
    span_id: str
    parent_id: str | None
    trace_id: str
    kind: str
    lane: str
    name: str
    start_ts: datetime
    attrs: dict[str, Any]


class SpanSink(Protocol):
    """Where spans go (the Ops ``spans`` table in production)."""

    def start_span(self, span: SpanStart) -> None: ...

    def end_span(
        self, span_id: str, end_ts: datetime, status: SpanStatus, attrs: Mapping[str, Any]
    ) -> None: ...


class _ClockLike(Protocol):
    def now(self) -> datetime: ...


@dataclass(frozen=True)
class Tracer:
    sink: SpanSink
    clock: _ClockLike
    trace_id: str
    ids: UlidFactory
    root_parent: str | None = None  # parent span from KTA_TRACE_PARENT


@dataclass
class SpanHandle:
    """Yielded by ``span()``; ``set`` adds attributes written at the end of the span."""

    span_id: str
    trace_id: str
    attrs: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)


_tracer: ContextVar[Tracer | None] = ContextVar("kta_tracer", default=None)
_current: ContextVar[SpanHandle | None] = ContextVar("kta_span", default=None)


def parse_trace_parent(value: str | None) -> tuple[str, str] | None:
    """(trace_id, span_id) from a ``KTA_TRACE_PARENT`` value; None when unset."""
    if not value:
        return None
    trace_id, sep, span_id = value.partition(":")
    if not sep or not trace_id or not span_id:
        raise ConfigError(f"{TRACE_PARENT_VAR} must be '<trace_id>:<span_id>', got {value!r}")
    return trace_id, span_id


def new_tracer(
    sink: SpanSink, clock: _ClockLike, ids: UlidFactory, environ: Mapping[str, str]
) -> Tracer:
    """A tracer that continues the parent's trace from ``KTA_TRACE_PARENT``, else a new one."""
    parent = parse_trace_parent(environ.get(TRACE_PARENT_VAR))
    if parent is None:
        return Tracer(sink, clock, ids.new(), ids)
    return Tracer(sink, clock, parent[0], ids, root_parent=parent[1])


@contextmanager
def tracing(tracer: Tracer) -> Generator[Tracer]:
    token = _tracer.set(tracer)
    try:
        yield tracer
    finally:
        _tracer.reset(token)


def current_tracer() -> Tracer | None:
    """The bound tracer, so work handed to another thread can keep the caller's trace."""
    return _tracer.get()


def current_span_id() -> str | None:
    handle = _current.get()
    return handle.span_id if handle else None


def child_environ() -> dict[str, str]:
    """Environment entries that make a child ``kta`` process attach to the current span."""
    handle = _current.get()
    if handle is None:
        return {}
    return {TRACE_PARENT_VAR: f"{handle.trace_id}:{handle.span_id}"}


@contextmanager
def span(kind: str, name: str, lane: str, **attrs: Any) -> Generator[SpanHandle]:
    tracer = _tracer.get()
    if tracer is None:
        raise ConfigError("span() needs a bound tracer (with tracing(tracer): ...)")
    if not KIND_RE.match(kind):
        raise ConfigError(f"span kind {kind!r} must be lower_snake_case")
    if not LANE_RE.match(lane):
        raise ConfigError(f"span lane {lane!r} must be process:/agent:/worker:h:pool/gpu:/human")
    parent = _current.get()
    handle = SpanHandle(tracer.ids.new(), tracer.trace_id)
    parent_id = parent.span_id if parent else tracer.root_parent
    start = SpanStart(
        handle.span_id, parent_id, tracer.trace_id, kind, lane, name, tracer.clock.now(), attrs
    )
    tracer.sink.start_span(start)
    token = _current.set(handle)
    status: SpanStatus = "ok"
    try:
        yield handle
    except (KeyboardInterrupt, GeneratorExit):
        status = "cancelled"
        raise
    except BaseException as exc:
        status = "cancelled" if type(exc).__name__ == "CancelledError" else "fail"
        handle.set(error=f"{type(exc).__name__}: {exc}"[:500])
        raise
    finally:
        _current.reset(token)
        tracer.sink.end_span(handle.span_id, tracer.clock.now(), status, handle.attrs)
