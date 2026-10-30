"""Store and Ops interfaces (spec/modules/core-store.md "接口", docs/design/20 §10)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from kta.core.schemas import Assurance, EvidenceStatus, Record
from kta.core.trace import SpanStart, SpanStatus

Direction = Literal["out", "in"]


@dataclass(frozen=True)
class Hit:
    id: str
    kind: str
    title: str
    snippet: str
    score: float  # higher is better (negated BM25)


@dataclass(frozen=True)
class Graph:
    root: str
    nodes: dict[str, str]  # id -> kind
    edges: list[tuple[str, str, str]] = field(default_factory=lambda: list[tuple[str, str, str]]())


@dataclass(frozen=True)
class Task:
    id: str
    card_id: str
    card_type: str
    state: str
    priority: int
    lease_owner: str | None
    lease_until: datetime | None
    attempts: int
    work_unit: str


@dataclass(frozen=True)
class OpsEvent:
    seq: int
    at: datetime
    table_name: str
    row_id: str
    change: dict[str, Any]


@dataclass(frozen=True)
class SpanRow:
    span_id: str
    parent_id: str | None
    trace_id: str
    kind: str
    lane: str
    name: str
    start_ts: datetime
    end_ts: datetime | None
    status: str
    attrs: dict[str, Any]


class Store(Protocol):
    def put(self, record: Record) -> str: ...

    def put_many(self, records: Iterable[Record]) -> list[str]: ...

    def get(self, id: str) -> Record: ...

    def exists(self, id: str) -> bool: ...

    def query(
        self,
        kind: str,
        where: Mapping[str, Any] | None = None,
        order: str | None = None,
        limit: int = 100,
    ) -> list[Record]: ...

    def search(self, text: str, kinds: list[str] | None = None, limit: int = 10) -> list[Hit]: ...

    def neighbors(
        self, id: str, rel: str | None = None, direction: Direction = "out"
    ) -> list[str]: ...

    def lineage(self, id: str, depth: int = 10) -> Graph: ...

    def status(self, id: str) -> tuple[EvidenceStatus, Assurance | None]: ...

    def put_blob(self, data: bytes | Path, media: str) -> str: ...

    def blob_path(self, sha256: str) -> Path: ...


class Ops(Protocol):
    def enqueue(self, card_id: str, priority: int, work_unit: str) -> str: ...

    def lease(self, worker: str, kinds: list[str], seconds: int) -> Task | None: ...

    def renew(self, task_id: str, worker: str, seconds: int) -> bool: ...

    def finish(self, task_id: str, state: str) -> None: ...

    def acquire(self, lock: str, holder: str, seconds: int) -> bool: ...

    def release(self, lock: str, holder: str) -> None: ...

    def get_pointer(self, name: str) -> str | None: ...

    def set_pointer(self, name: str, expected: str | None, value: str) -> bool: ...

    def set_budget(self, scope: str, limit: Mapping[str, float]) -> None: ...

    def reserve(self, scope: str, amount: Mapping[str, float]) -> bool: ...

    def settle(
        self, scope: str, reserved: Mapping[str, float], used: Mapping[str, float]
    ) -> None: ...

    def events(self, since: int) -> list[OpsEvent]: ...

    # Workers (execution, decision D192).
    def heartbeat(self, name: str, host: str, healthy: bool) -> None: ...

    def worker_health(self, name: str) -> bool | None: ...

    def record_event(self, table: str, row_id: str, change: Mapping[str, Any]) -> None: ...

    # SpanSink (kta.core.trace) and span reads for the timeline.
    def start_span(self, span: SpanStart) -> None: ...

    def end_span(
        self, span_id: str, end_ts: datetime, status: SpanStatus, attrs: Mapping[str, Any]
    ) -> None: ...

    def spans(
        self, trace_id: str | None = None, since: datetime | None = None, limit: int = 1000
    ) -> list[SpanRow]: ...
