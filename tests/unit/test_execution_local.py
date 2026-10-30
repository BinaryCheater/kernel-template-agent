"""Executor over the real ``local`` transport: transfer, errors, labels, pools, leases.

The worker is this machine running the same code; jobs are shell commands. GPU pools are
exercised by registering the local worker with GPU labels by hand (the jobs never touch
a GPU), which is enough to test the scheduling rules of the lease.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kta.core.config import config_from_mapping
from kta.core.context import Context
from kta.core.errors import ExecutionError
from kta.core.jobs import Job
from kta.core.registry import Registry
from kta.core.schemas import Actor, Record
from kta.core.schemas.hardware import Worker
from kta.execution.executor import JobExecutor
from kta.execution.registry import Registration, register, worker_configs
from kta.store import CodeState, SqliteOps, SqliteStore, code_state, open_ops, open_store

WAIT_S = 20.0


@pytest.fixture
def lctx(tmp_path: Path) -> Context:
    config = {
        # A health-check command that cannot answer stands in for a hung GPU.
        "workers": {
            "local": {
                "transport": "local",
                "pools": {"cpu": 2},
                "nvidia_smi": "/nonexistent/nvidia-smi",
            }
        },
        "executor": {"queue_timeout_s": 30},
    }
    return Context(
        env="test", config=config_from_mapping(config), data_dir=tmp_path / "d", registry=Registry()
    )


@pytest.fixture
def lstore(lctx: Context) -> SqliteStore:
    return open_store(lctx, CodeState(code_state().commit, False))


@pytest.fixture
def lops(lctx: Context) -> SqliteOps:
    return open_ops(lctx)


def registered(lctx: Context, lstore: SqliteStore, lops: SqliteOps) -> Registration:
    return register(lctx, lstore, lops, worker_configs(lctx)["local"], code_state().commit)


def gpu_worker(lstore: SqliteStore, reg: Registration, commit: str | None = None) -> None:
    """A later registration of the local worker that advertises one GPU device."""
    labels = {
        **reg.worker.labels,
        "gpu": "sm_test",
        "pool.gpu-shared": "2",
        "pool.gpu-exclusive": "1",
    }
    body = Worker(
        name="local",
        host=reg.worker.host,
        transport="local",
        labels=labels,
        devices=["0"],
        commit=commit or reg.worker.commit,
    )
    later = datetime.now(UTC) + timedelta(seconds=1)  # newer than the registration
    rec = Record.new(
        "Worker", body, Actor(type="system", name="test"), later, refs={"env": reg.env_id}
    )
    lstore.put(rec)


def wait_until(ex: JobExecutor, cond: Callable[[], bool]) -> None:
    deadline = threading.Event()
    timer = threading.Timer(WAIT_S, deadline.set)
    timer.start()
    try:
        with ex.cond:
            while not cond():
                if deadline.is_set():
                    raise AssertionError("condition not reached")
                ex.cond.wait(0.02)
    finally:
        timer.cancel()


def test_registration_writes_worker_and_manifest(lctx, lstore, lops) -> None:
    reg = registered(lctx, lstore, lops)
    assert lstore.get(reg.worker_id).kind == "Worker"
    manifest = lstore.get(reg.env_id)
    assert manifest.kind == "EnvManifest" and manifest.body["role"] == "worker"
    assert reg.worker.labels["pool.cpu"] == "2"
    assert reg.worker.labels["commit"] == code_state().commit
    assert lops.worker_health("local") is True


def test_inputs_and_outputs_travel_by_hash(lctx, lstore, lops) -> None:
    registered(lctx, lstore, lops)
    ex = JobExecutor(lctx, lstore, lops, code_state().commit)
    sha = lstore.put_blob(b"payload\n", "text/plain")
    job = Job(
        kind="shell",
        argv=["sh", "-c", "mkdir -p out/sub && cat in/x.txt > out/sub/y.txt"],
        inputs={"in/x.txt": sha},
    )
    result = ex.run(job)
    assert result.ok and result.error_kind == "none"
    assert lstore.blob_path(result.outputs["sub/y.txt"]).read_bytes() == b"payload\n"
    assert result.worker_name == "local" and result.env_manifest


def test_error_kinds(lctx, lstore, lops) -> None:
    registered(lctx, lstore, lops)
    ex = JobExecutor(lctx, lstore, lops, code_state().commit)
    failed = ex.run(Job(kind="shell", argv=["sh", "-c", "echo boom >&2; exit 3"]))
    assert (failed.ok, failed.error_kind, failed.exit_code) == (False, "nonzero", 3)
    assert b"boom" in lstore.blob_path(failed.stderr_blob).read_bytes()
    slow = ex.run(Job(kind="shell", argv=["sh", "-c", "sleep 30"], timeout_s=1))
    assert slow.error_kind == "timeout" and slow.wall_s < 10


def test_input_names_cannot_escape_the_job_directory(lctx, lstore, lops) -> None:
    registered(lctx, lstore, lops)
    ex = JobExecutor(lctx, lstore, lops, code_state().commit)
    sha = lstore.put_blob(b"x", "text/plain")
    result = ex.run(Job(kind="shell", argv=["true"], inputs={"../escape": sha}))
    assert result.error_kind == "infra"


def test_unmatched_requirements_wait_then_fail(lctx, lstore, lops) -> None:
    registered(lctx, lstore, lops)
    ex = JobExecutor(lctx, lstore, lops, code_state().commit)
    ex.queue_timeout_s = 0.3
    with pytest.raises(ExecutionError, match="queued too long"):
        ex.run(Job(kind="shell", argv=["true"], requires={"gpu": "sm_999"}))


def test_worker_with_another_commit_gets_no_jobs(lctx, lstore, lops) -> None:
    reg = registered(lctx, lstore, lops)
    gpu_worker(lstore, reg, commit="0" * 40)
    ex = JobExecutor(lctx, lstore, lops, code_state().commit)
    assert ex.workers == []


def test_exclusive_lease_defers_shared_and_cpu_jobs(lctx, lstore, lops, tmp_path) -> None:
    gpu_worker(lstore, registered(lctx, lstore, lops))
    ex = JobExecutor(lctx, lstore, lops, code_state().commit)
    gate = tmp_path / "release"
    hold = Job(
        kind="measure",
        argv=["sh", "-c", 'while [ ! -f "$GATE" ]; do sleep 0.01; done'],
        env={"GATE": str(gate)},
        timeout_s=60,
    )
    ex_id = ex.submit(hold)
    wait_until(ex, lambda: ex.workers[0].leases != {})
    shared = ex.submit(Job(kind="run", argv=["true"]))
    cpu = ex.submit(Job(kind="shell", argv=["true"]))
    wait_until(ex, lambda: sum("GPU lease" in (p.deferred_by or "") for p in ex.jobs.values()) == 2)
    assert not ex.jobs[shared].done.is_set() and not ex.jobs[cpu].done.is_set()
    gate.touch()
    results = [ex.wait(j) for j in (ex_id, shared, cpu)]
    assert all(r.ok for r in results)
    runs = {r.attrs["job_id"]: r for r in lops.spans(limit=100) if r.kind == "job_run"}
    end = runs[ex_id].end_ts
    assert end is not None
    assert runs[shared].start_ts >= end and runs[cpu].start_ts >= end
    events = [e.change for e in lops.events(0) if e.table_name == "locks"]
    assert any(c.get("holder") == ex_id for c in events)
    assert any(c.get("released_by") == ex_id for c in events)


def test_quiet_lease_is_force_released_after_max_s(lctx, lstore, lops) -> None:
    gpu_worker(lstore, registered(lctx, lstore, lops))
    ex = JobExecutor(lctx, lstore, lops, code_state().commit)
    with ex.quiet_lease("local", "0", "batch-1", max_s=1):
        assert ex.lease_held("local")
        wait_until(ex, lambda: not ex.workers[0].leases)
    forced = [e.change for e in lops.events(0) if "forced_release" in e.change]
    assert forced and forced[0]["forced_release"] == "batch-1"


def test_jobs_inside_a_quiet_lease_run_on_it(lctx, lstore, lops) -> None:
    gpu_worker(lstore, registered(lctx, lstore, lops))
    ex = JobExecutor(lctx, lstore, lops, code_state().commit)
    with ex.quiet_lease("local", "0", "batch-2"):
        a = ex.run(Job(kind="measure", argv=["true"]))
        b = ex.run(Job(kind="measure", argv=["true"]))
    assert a.ok and b.ok and a.device == "0"
    assert not ex.lease_held("local")


def test_gpu_timeout_without_answering_gpu_marks_worker_unhealthy(lctx, lstore, lops) -> None:
    # The configured health check cannot answer, as for a hung GPU: gpu_hang.
    gpu_worker(lstore, registered(lctx, lstore, lops))
    ex = JobExecutor(lctx, lstore, lops, code_state().commit)
    hung = ex.run(Job(kind="run", argv=["sh", "-c", "sleep 30"], timeout_s=1))
    assert hung.error_kind == "gpu_hang"
    assert lops.worker_health("local") is False
    ex.queue_timeout_s = 0.3
    with pytest.raises(ExecutionError, match="queued too long"):
        ex.run(Job(kind="run", argv=["true"]))
    assert ex.run(Job(kind="shell", argv=["true"])).ok  # cpu jobs still run
