"""The Executor: queue, worker selection by labels, pools, GPU leases, result transfer.

Scheduling state lives in this object (one Executor per control-plane process,
decision D184). Each submitted job gets a thread that waits for a slot, ships missing
input blobs, runs the job through the worker's transport, and brings stdout, stderr
and outputs back into the store by hash.

GPU-exclusive lease per (worker, device): while an exclusive job or a ``quiet_lease``
waits for or holds it, no new gpu-shared or cpu job starts on that worker (they are
deferred, and the deferral is logged); the lease is taken once running shared and cpu
jobs drain. It is also held in the ops ``locks`` table, so two control-plane processes
never measure on one GPU at once. A lease older than ``max_s`` is force-released and
the release is written to ``ops_events``.

Host lock (WP23, D603): on a worker with ``host_lock`` (a unified-memory compute host)
every cpu and gpu-shared job runs inside the host's readers–writer lock taken shared, a
self-leased gpu-exclusive job takes it exclusive, and ``quiet_lease`` holds it exclusive
for the whole window through a ``hold`` process on the worker, so control planes of
other processes or modes drain too (``kta.execution.hostpolicy``). Lock waits are
``lock_wait`` spans on lane ``host:<worker>``. cpu jobs on a ``unified_memory`` worker are
also limited by the memory cap of the controls knob ``execution.unified_memory_cap``.

``cancel`` (decision D420) takes a queued job out of the queue and stops a running one
on its worker through the ``cancel`` worker command (process group, container); either
way ``wait`` then raises ``ExecutionError("cancelled")``.
"""

from __future__ import annotations

import contextvars
import logging
import mimetypes
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

from kta.core.context import Context
from kta.core.errors import ConfigError, ExecutionError, NotSupported
from kta.core.ids import UlidFactory
from kta.core.jobs import GPU_POOLS, Job, JobResult
from kta.core.logging import bind
from kta.core.trace import SpanStart, current_span_id, current_tracer, new_tracer, span, tracing
from kta.execution import hostpolicy
from kta.execution.base import HeldHostLock, WorkerClient, WorkerConfig
from kta.execution.registry import (
    Registration,
    client_for,
    label_matches,
    latest_registration,
    worker_configs,
)
from kta.store.base import Ops, Store

log = logging.getLogger("kta.execution.executor")
LEASE_MAX_S = 600
POLL_S = 0.5
_lease_holder: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "kta_lease_holder", default=None
)


@dataclass
class _Lease:
    holder: str
    started: float
    max_s: int
    timer: threading.Timer | None = None


@dataclass
class _Worker:
    config: WorkerConfig
    client: WorkerClient
    reg: Registration
    healthy: bool = True
    running: dict[str, int] = field(default_factory=lambda: dict[str, int]())
    leases: dict[str, _Lease] = field(default_factory=lambda: dict[str, _Lease]())
    waiting_exclusive: dict[str, int] = field(default_factory=lambda: dict[str, int]())
    held: dict[str, HeldHostLock] = field(default_factory=lambda: dict[str, HeldHostLock]())
    mem_gb: float = 0.0  # reserved by running cpu jobs (unified-memory cap)

    @property
    def name(self) -> str:
        return self.config.name

    def limit(self, pool: str) -> int:
        return int(self.reg.worker.labels.get(f"pool.{pool}", "0"))

    def busy(self, *pools: str) -> int:
        return sum(self.running.get(p, 0) for p in pools)

    def quiet_requested(self) -> bool:
        return bool(self.leases) or any(self.waiting_exclusive.values())


@dataclass
class _Slot:
    worker: _Worker
    pool: str
    device: str | None
    lease_owner: str | None = None  # set when this job took the lease itself
    mem_gb: float = 0.0


@dataclass
class _Pending:
    job_id: str
    job: Job
    holder: str | None
    submitted: float
    done: threading.Event = field(default_factory=threading.Event)
    result: JobResult | None = None
    error: BaseException | None = None
    deferred_by: str | None = None
    warned_unmatched: bool = False
    cancelled: bool = False
    slot: _Slot | None = None  # set once the job left the queue


class JobExecutor:
    def __init__(self, ctx: Context, store: Store, ops: Ops, commit: str) -> None:
        self.ctx = ctx
        self.store = store
        self.ops = ops
        self.commit = commit
        self.ids = UlidFactory(ctx.clock)
        self.cond = threading.Condition()
        self.jobs: dict[str, _Pending] = {}
        self.queue_timeout_s = float(ctx.config.section("executor").get("queue_timeout_s", 3600))
        self.mem_cap = ctx.controls.unified_memory_cap
        self.exclusive_wait_s = float(ctx.controls.batch_window["exclusive_wait_s"])
        self.workers: list[_Worker] = []
        for config in worker_configs(ctx).values():
            reg = latest_registration(store, config.name)
            if reg is None:
                continue
            if reg.worker.commit != commit:
                log.warning(
                    "worker %s registered with commit %s, control plane runs %s: not used",
                    config.name,
                    reg.worker.commit[:12],
                    commit[:12],
                )
                continue
            healthy = ops.worker_health(config.name) is not False
            client = client_for(ctx, config, commit)
            self.workers.append(_Worker(config, client, reg, healthy))
        self._tracer = new_tracer(ops, ctx.clock, self.ids, {})

    # ---- public API --------------------------------------------------------------

    def submit(self, job: Job) -> str:
        job_id = self.ids.new()
        pending = _Pending(job_id, job, _lease_holder.get(), time.monotonic())
        with self.cond:
            self.jobs[job_id] = pending
        caller = contextvars.copy_context()
        thread = threading.Thread(
            target=caller.run, args=(self._drive, pending), name=f"kta-job-{job_id}", daemon=True
        )
        thread.start()
        return job_id

    def wait(self, job_id: str) -> JobResult:
        pending = self.jobs.get(job_id)
        if pending is None:
            raise ConfigError(f"unknown job {job_id}")
        pending.done.wait()
        if pending.error is not None:
            raise pending.error
        if pending.result is None:
            raise ExecutionError("infra", f"job {job_id} finished without a result")
        return pending.result

    def run(self, job: Job) -> JobResult:
        return self.wait(self.submit(job))

    def cancel(self, job_id: str) -> None:
        """Stop a job; ``wait`` then raises ExecutionError("cancelled").

        A queued job leaves the queue; a running one is killed on its worker. A finished
        job is left as it is, and cancelling twice is harmless.
        """
        with self.cond:
            p = self.jobs.get(job_id)
            if p is None:
                raise ConfigError(f"unknown job {job_id}")
            if p.done.is_set():
                return
            p.cancelled = True
            slot = p.slot
            self.cond.notify_all()
        if slot is None:
            log.info("job %s cancelled while queued", job_id)
            return
        try:
            stopped = slot.worker.client.cancel(job_id)
        except (ExecutionError, NotSupported) as exc:
            # The job still ends as cancelled; its slot frees when the worker returns.
            log.warning("job %s: cancel did not reach %s: %s", job_id, slot.worker.name, exc)
            return
        log.info(
            "job %s cancelled on %s (%s)",
            job_id,
            slot.worker.name,
            "process killed" if stopped else "process not started or already ended",
        )

    def pending(self) -> list[dict[str, Any]]:
        """Jobs still waiting for a worker, with the reason (for the pending view)."""
        with self.cond:
            return [
                {
                    "job_id": p.job_id,
                    "kind": p.job.kind,
                    "requires": p.job.requires,
                    "waiting_s": time.monotonic() - p.submitted,
                    "reason": p.deferred_by or "no matching worker",
                }
                for p in self.jobs.values()
                if not p.done.is_set()
            ]

    def read_blob(self, sha256: str) -> bytes:
        return self.store.blob_path(sha256).read_bytes()

    def put_blob(self, data: bytes, media: str) -> str:
        return self.store.put_blob(data, media)

    def lease_held(self, worker: str) -> bool:
        with self.cond:
            return any(w.leases for w in self.workers if w.name == worker)

    def worker(self, name: str) -> Registration:
        for w in self.workers:
            if w.name == name:
                return w.reg
        raise ConfigError(f"worker {name!r} is not registered for commit {self.commit[:12]}")

    @contextmanager
    def quiet_lease(
        self, worker: str, device: str, holder: str, max_s: int = LEASE_MAX_S
    ) -> Generator[None]:
        w = next((x for x in self.workers if x.name == worker), None)
        if w is None or device not in w.reg.worker.devices:
            raise ConfigError(f"no device {device} on registered worker {worker!r}")
        with self.cond:
            w.waiting_exclusive[device] = w.waiting_exclusive.get(device, 0) + 1
            try:
                while not self._try_lease(w, device, holder, max_s):
                    self.cond.wait(POLL_S)
            finally:
                w.waiting_exclusive[device] -= 1
        token = _lease_holder.set(holder)
        try:
            with self._tracing():
                self._hold_host(w, device, holder, max_s)
                with span(
                    "lock_hold", f"gpu lease {worker}:{device}", f"gpu:{worker}", holder=holder
                ):
                    yield
        finally:
            _lease_holder.reset(token)
            self._release_host(w, device)
            with self.cond:
                self._release_lease(w, device, holder, forced=False)

    def _hold_host(self, w: _Worker, device: str, holder: str, max_s: int) -> None:
        """The host lock exclusive for the window (other control planes drain too)."""
        if not w.config.host_lock:
            return
        with span("lock_wait", f"host lock {w.name} for {holder}", f"host:{w.name}") as sp:
            held = w.client.hold(holder, self.exclusive_wait_s, max_s)
            sp.set(wait_s=held.acquire(), mode="exclusive", holder=holder)
        w.held[device] = held

    def _release_host(self, w: _Worker, device: str) -> None:
        held = w.held.pop(device, None)
        if held is not None and held.release():
            log.warning("host lock window on %s ended by its max hold", w.name)
            self.ops.record_event("locks", f"host:{w.name}", {"forced_release": True})

    # ---- leases ------------------------------------------------------------------

    def _lock_name(self, w: _Worker, device: str) -> str:
        return f"gpu:{w.name}:{device}"

    def _try_lease(self, w: _Worker, device: str, holder: str, max_s: int) -> bool:
        """Take the device lease once shared, cpu and exclusive work on ``w`` drained."""
        if device in w.leases or w.busy("gpu-shared", "gpu-exclusive", "cpu"):
            return False
        if not self.ops.acquire(self._lock_name(w, device), holder, max_s):
            return False
        lease = _Lease(holder, time.monotonic(), max_s)
        lease.timer = threading.Timer(max_s, self._force_release, args=(w, device, holder))
        lease.timer.daemon = True
        lease.timer.start()
        w.leases[device] = lease
        log.info("GPU lease %s:%s taken by %s (max %ss)", w.name, device, holder, max_s)
        return True

    def _release_lease(self, w: _Worker, device: str, holder: str, forced: bool) -> None:
        lease = w.leases.get(device)
        if lease is None or lease.holder != holder:
            return
        if lease.timer is not None and not forced:
            lease.timer.cancel()
        del w.leases[device]
        self.ops.release(self._lock_name(w, device), holder)
        held_s = time.monotonic() - lease.started
        if forced:
            log.warning(
                "GPU lease %s:%s of %s force-released after %.0fs", w.name, device, holder, held_s
            )
            self.ops.record_event(
                "locks",
                self._lock_name(w, device),
                {"forced_release": holder, "held_s": round(held_s, 3), "max_s": lease.max_s},
            )
        else:
            log.info("GPU lease %s:%s released by %s after %.3fs", w.name, device, holder, held_s)
        self.cond.notify_all()

    def _force_release(self, w: _Worker, device: str, holder: str) -> None:
        with self.cond:
            self._release_lease(w, device, holder, forced=True)

    # ---- scheduling --------------------------------------------------------------

    def _candidates(self, job: Job) -> list[_Worker]:
        # ``worker`` names one worker: only the settings page's worker Test uses it (D589).
        named = job.requires.get("worker")
        needs = {k: v for k, v in job.requires.items() if k not in ("pool", "worker")}
        return [
            w
            for w in self.workers
            if (named is None or w.name == named)
            and label_matches(w.reg.worker.labels, needs)
            and w.limit(job.pool) > 0
            and (w.healthy or not job.uses_gpu)
        ]

    def _place(self, p: _Pending) -> _Slot | None:
        """A slot for the job now, or None (with ``p.deferred_by`` set when a lease blocks)."""
        pool = p.job.pool
        p.deferred_by = None
        for w in self._candidates(p.job):
            if w.busy(pool) >= w.limit(pool):
                continue
            if pool == "gpu-exclusive":
                for device in w.reg.worker.devices:
                    lease = w.leases.get(device)
                    if lease is not None and lease.holder == p.holder:
                        return _Slot(w, pool, device)
                    if lease is None and self._try_lease(w, device, p.job_id, LEASE_MAX_S):
                        return _Slot(w, pool, device, lease_owner=p.job_id)
                    p.deferred_by = f"waiting for GPU lease on {w.name}:{device}"
                continue
            own = p.holder is not None and any(x.holder == p.holder for x in w.leases.values())
            if w.quiet_requested() and not own:
                holders = [x.holder for x in w.leases.values()] or ["a waiting exclusive job"]
                p.deferred_by = f"GPU lease on {w.name} held by {', '.join(holders)}"
                continue
            device = w.reg.worker.devices[0] if pool in GPU_POOLS and w.reg.worker.devices else None
            if pool in GPU_POOLS and device is None:
                continue
            blocked = hostpolicy.mem_blocks(w.config, p.job, w.mem_gb, self.mem_cap)
            if blocked is not None:
                p.deferred_by = blocked
                continue
            mem = hostpolicy.job_mem_gb(p.job, self.mem_cap) if pool == "cpu" else 0.0
            return _Slot(w, pool, device, mem_gb=mem)
        return None

    def _acquire_slot(self, p: _Pending) -> _Slot:
        logged: str | None = None
        with self.cond:
            if p.job.pool == "gpu-exclusive":
                for w in self._candidates(p.job):
                    for d in w.reg.worker.devices:
                        w.waiting_exclusive[d] = w.waiting_exclusive.get(d, 0) + 1
            try:
                while True:
                    if p.cancelled:
                        raise ExecutionError("cancelled", f"job {p.job_id} cancelled while queued")
                    slot = self._place(p)
                    if slot is not None:
                        slot.worker.running[slot.pool] = slot.worker.running.get(slot.pool, 0) + 1
                        slot.worker.mem_gb += slot.mem_gb
                        p.slot = slot
                        return slot
                    if p.deferred_by and p.deferred_by != logged:
                        log.info(
                            "job %s (%s, %s) deferred: %s",
                            p.job_id,
                            p.job.kind,
                            p.job.pool,
                            p.deferred_by,
                        )
                        logged = p.deferred_by
                    if not self._candidates(p.job) and not p.warned_unmatched:
                        log.warning(
                            "job %s waits: no registered worker matches %s pool %s",
                            p.job_id,
                            p.job.requires,
                            p.job.pool,
                        )
                        p.warned_unmatched = True
                    if time.monotonic() - p.submitted > self.queue_timeout_s:
                        reason = p.deferred_by or "no matching worker"
                        raise ExecutionError("infra", f"job {p.job_id} queued too long: {reason}")
                    self.cond.wait(POLL_S)
            finally:
                if p.job.pool == "gpu-exclusive":
                    for w in self._candidates(p.job):
                        for d in w.reg.worker.devices:
                            w.waiting_exclusive[d] -= 1

    def _free_slot(self, slot: _Slot) -> None:
        with self.cond:
            slot.worker.running[slot.pool] -= 1
            slot.worker.mem_gb -= slot.mem_gb
            if slot.lease_owner is not None and slot.device is not None:
                self._release_lease(slot.worker, slot.device, slot.lease_owner, forced=False)
            self.cond.notify_all()

    # ---- running -----------------------------------------------------------------

    def _tracing(self) -> Any:
        return nullcontext() if current_tracer() is not None else tracing(self._tracer)

    def _drive(self, p: _Pending) -> None:
        try:
            with self._tracing(), bind(job_id=p.job_id):
                queued_at = self.ctx.clock.now()
                slot = self._acquire_slot(p)
                queue_s = time.monotonic() - p.submitted
                self._queue_span(p, slot, queued_at)
                try:
                    lane = f"worker:{slot.worker.name}:{slot.pool}"
                    with span("job_run", f"{p.job.kind} {p.job_id}", lane, job_id=p.job_id) as r:
                        p.result = self._execute(slot, p, queue_s)
                        r.set(
                            ok=p.result.ok,
                            error_kind=p.result.error_kind,
                            wall_s=round(p.result.wall_s, 6),
                        )
                finally:
                    self._free_slot(slot)
        except BaseException as exc:  # noqa: BLE001 - handed to wait(), which re-raises it
            p.error = exc
        finally:
            with self.cond:
                # A cancel decides the outcome even when the job ended some other way first.
                if p.cancelled and not (
                    isinstance(p.error, ExecutionError) and p.error.kind == "cancelled"
                ):
                    p.error = ExecutionError("cancelled", f"job {p.job_id} cancelled")
                p.done.set()
                self.cond.notify_all()

    def _queue_span(self, p: _Pending, slot: _Slot, queued_at: datetime) -> None:
        """The job_queue span, written once the worker (and so its lane) is known."""
        tracer = current_tracer()
        if tracer is None:
            raise ConfigError("executor threads run with a bound tracer")
        start = SpanStart(
            tracer.ids.new(),
            current_span_id() or tracer.root_parent,
            tracer.trace_id,
            "job_queue",
            f"worker:{slot.worker.name}:{slot.pool}",
            f"{p.job.kind} {p.job_id}",
            queued_at,
            {"job_id": p.job_id},
        )
        tracer.sink.start_span(start)
        attrs = {"deferred_by": p.deferred_by} if p.deferred_by else {}
        tracer.sink.end_span(start.span_id, self.ctx.clock.now(), "ok", attrs)

    def _upload(self, client: WorkerClient, job: Job) -> None:
        shas = sorted(set(job.inputs.values()))
        missing = client.missing(shas) if shas else []
        if missing:
            client.put({sha: self.store.blob_path(sha).read_bytes() for sha in missing})

    def _download(self, client: WorkerClient, raw: dict[str, Any]) -> None:
        named = {raw["stdout_blob"]: "stdout.txt", raw["stderr_blob"]: "stderr.txt"}
        named.update({sha: name for name, sha in raw["outputs"].items()})
        for sha, data in client.get(sorted(named)).items():
            media = mimetypes.guess_type(named[sha])[0] or "application/octet-stream"
            if self.store.put_blob(data, media) != sha:
                raise ExecutionError("infra", f"blob {sha[:12]} changed in transfer")

    def _execute(self, slot: _Slot, p: _Pending, queue_s: float) -> JobResult:
        w, job = slot.worker, p.job
        if slot.lease_owner is not None:
            job = replace(job, timeout_s=min(job.timeout_s, LEASE_MAX_S))
        if p.cancelled:
            raise ExecutionError("cancelled", f"job {p.job_id} cancelled before it started")
        log.info("job %s start on %s pool %s device %s", p.job_id, w.name, slot.pool, slot.device)
        lock = self._lock_request(w, p, slot)
        try:
            self._upload(w.client, job)
            started = self.ctx.clock.now()
            raw = w.client.run(p.job_id, job, slot.device, lock)
            self._lock_span(w, p, raw.get("host_lock"), started)
            self._download(w.client, raw)
        except NotSupported:
            raise
        except ExecutionError as exc:
            log.error("job %s infra failure on %s: %s", p.job_id, w.name, exc)
            return self._infra_result(slot, p, queue_s, str(exc))
        if raw["error_kind"] == "gpu_hang":
            w.healthy = False
            self.ops.heartbeat(w.name, w.reg.worker.host, False)
            log.error(
                "worker %s marked unhealthy: GPU did not answer after job %s", w.name, p.job_id
            )
        log.info(
            "job %s end ok=%s kind=%s exit=%s wall=%.3fs",
            p.job_id,
            raw["ok"],
            raw["error_kind"],
            raw["exit_code"],
            raw["wall_s"],
        )
        return JobResult(
            ok=bool(raw["ok"]),
            exit_code=int(raw["exit_code"]),
            wall_s=float(raw["wall_s"]),
            queue_s=queue_s,
            worker=w.reg.worker_id,
            device=raw["device"],
            env_manifest=w.reg.env_id,
            stdout_blob=raw["stdout_blob"],
            stderr_blob=raw["stderr_blob"],
            outputs=dict(raw["outputs"]),
            error_kind=raw["error_kind"],
            job_id=p.job_id,
            worker_name=w.name,
        )

    def _lock_request(self, w: _Worker, p: _Pending, slot: _Slot) -> dict[str, Any] | None:
        own = p.holder is not None and any(
            lease.holder == p.holder and dev in w.held for dev, lease in w.leases.items()
        )
        mode = hostpolicy.host_mode(p.job, own, slot.lease_owner is not None)
        return hostpolicy.lock_request(
            w.config,
            mode,
            f"job:{p.job_id}",
            self.queue_timeout_s,
            self.exclusive_wait_s,
            LEASE_MAX_S,
        )

    def _lock_span(
        self, w: _Worker, p: _Pending, got: dict[str, Any] | None, started: datetime
    ) -> None:
        """A lock_wait span for the time the job waited for the host lock on its worker."""
        if not got or not float(got.get("wait_s") or 0.0):
            return
        tracer = current_tracer()
        if tracer is None:
            raise ConfigError("executor threads run with a bound tracer")
        start = SpanStart(
            tracer.ids.new(),
            current_span_id() or tracer.root_parent,
            tracer.trace_id,
            "lock_wait",
            f"host:{w.name}",
            f"host lock {got['mode']} for {p.job.kind} {p.job_id}",
            started,
            {"job_id": p.job_id, "mode": got["mode"], "wait_s": got["wait_s"]},
        )
        tracer.sink.start_span(start)
        end = started + timedelta(seconds=float(got["wait_s"]))
        tracer.sink.end_span(start.span_id, end, "ok", {})

    def _infra_result(self, slot: _Slot, p: _Pending, queue_s: float, message: str) -> JobResult:
        empty = self.store.put_blob(b"", "text/plain")
        err = self.store.put_blob(message.encode("utf-8"), "text/plain")
        return JobResult(
            ok=False,
            exit_code=-1,
            wall_s=0.0,
            queue_s=queue_s,
            worker=slot.worker.reg.worker_id,
            device=slot.device,
            env_manifest=slot.worker.reg.env_id,
            stdout_blob=empty,
            stderr_blob=err,
            error_kind="infra",
            job_id=p.job_id,
            worker_name=slot.worker.name,
        )
