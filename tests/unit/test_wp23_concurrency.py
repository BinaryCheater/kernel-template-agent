"""WP23: host lock, quiet-host verification, memory cap, stage queues, work units.

The host-lock tests use real processes (``python -m kta.core.hostlock hold``) and the
Executor tests the real ``local`` transport; nothing here touches a GPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, Any

import pytest

from kta.core.config import config_from_mapping
from kta.core.context import Context
from kta.core.controls import baseline
from kta.core.hostlock import HostLock, HostLockTimeout, heavy
from kta.core.jobs import Job
from kta.core.registry import Registry
from kta.execution import hostpolicy
from kta.execution.base import WorkerConfig
from kta.execution.executor import JobExecutor
from kta.execution.registry import register, worker_configs
from kta.measure import quiet
from kta.measure.runner import HostNotQuiet, Measurer
from kta.orchestrator import stagequeue, workunits
from kta.store import CodeState, SqliteOps, SqliteStore, code_state, open_ops, open_store

pytestmark = pytest.mark.contract
SHORT_S = 0.3


# ---- a lock held by another process ---------------------------------------------------------


class Holder:
    def __init__(self, directory: Path, mode: str, wait_s: float = 30.0) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "kta.core.hostlock", "hold", str(directory), mode,
             "--wait-s", str(wait_s), "--max-s", "120"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )  # fmt: skip

    def expect(self, state: str) -> dict[str, Any]:
        out: IO[bytes] | None = self.proc.stdout
        assert out is not None
        msg = json.loads(out.readline())
        assert msg["state"] == state, msg
        return msg

    def release(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        self.expect("released")
        assert self.proc.wait(timeout=30) == 0


@contextmanager
def held(directory: Path, mode: str) -> Iterator[Holder]:
    h = Holder(directory, mode)
    if mode == "exclusive":
        h.expect("waiting")
    h.expect("acquired")
    try:
        yield h
    finally:
        if h.proc.poll() is None:
            h.release()


def test_host_lock_excludes_across_two_processes(tmp_path: Path) -> None:
    lock = HostLock(tmp_path / "hl")
    with held(tmp_path / "hl", "shared"):
        with pytest.raises(HostLockTimeout):
            with lock.exclusive("measure", SHORT_S, 60):
                pass
        with lock.shared("compile", SHORT_S) as g:  # readers share
            assert g.mode == "shared"
    with held(tmp_path / "hl", "exclusive"):
        with pytest.raises(HostLockTimeout):
            with lock.shared("compile", SHORT_S):
                pass
    with lock.exclusive("measure", 5.0, 60) as g:
        assert g.mode == "exclusive" and lock.holder() is not None
    assert lock.holder() is None


def test_writer_preference_blocks_new_readers(tmp_path: Path) -> None:
    lock = HostLock(tmp_path / "hl")
    with held(tmp_path / "hl", "shared") as reader:
        writer = Holder(tmp_path / "hl", "exclusive")
        writer.expect("waiting")  # the writer holds the turnstile, the reader still runs
        with pytest.raises(HostLockTimeout):
            with lock.shared("late compile", SHORT_S):  # rw.lock is only shared-held
                pass
        reader.release()
        writer.expect("acquired")
        writer.release()
    with lock.shared("compile", SHORT_S):
        pass


def test_heavy_takes_the_lock_shared_only_on_the_compute_host(tmp_path: Path) -> None:
    with heavy({}, "solve") as g:
        assert g is None  # control plane off the compute host (dev mode)
    with held(tmp_path / "hl", "exclusive"):
        with pytest.raises(HostLockTimeout):
            with heavy({"lock_dir": str(tmp_path / "hl"), "shared_wait_s": SHORT_S}, "solve"):
                pass
    with heavy({"lock_dir": str(tmp_path / "hl")}, "solve") as g:
        assert g is not None and g.mode == "shared"


# ---- the Executor on a host-locked, unified-memory worker -----------------------------------


@pytest.fixture
def hctx(tmp_path: Path) -> Context:
    config = {
        "workers": {
            "local": {
                "transport": "local",
                "pools": {"cpu": 4},
                "nvidia_smi": "/nonexistent/nvidia-smi",
                "host_lock": str(tmp_path / "hl"),
                "unified_memory": True,
            }
        },
        "executor": {"queue_timeout_s": 30},
    }
    return Context(
        env="test", config=config_from_mapping(config), data_dir=tmp_path / "d", registry=Registry()
    )


@pytest.fixture
def hstore(hctx: Context) -> SqliteStore:
    return open_store(hctx, CodeState(code_state().commit, False))


@pytest.fixture
def hops(hctx: Context) -> SqliteOps:
    return open_ops(hctx)


def test_cpu_job_waits_for_another_control_planes_measurement(
    hctx: Context, hstore: SqliteStore, hops: SqliteOps, tmp_path: Path
) -> None:
    register(hctx, hstore, hops, worker_configs(hctx)["local"], code_state().commit)
    ex = JobExecutor(hctx, hstore, hops, code_state().commit)
    with held(tmp_path / "hl", "exclusive"):  # a measurement of another control plane
        job_id = ex.submit(Job(kind="shell", argv=["true"]))
        assert not ex.jobs[job_id].done.wait(1.0)
    assert ex.wait(job_id).ok
    waits = [s for s in hops.spans(limit=100) if s.kind == "lock_wait"]
    assert waits and waits[0].lane == "host:local" and waits[0].attrs["mode"] == "shared"


def test_unified_memory_cap_defers_cpu_jobs() -> None:
    cfg = WorkerConfig(name="w", transport="local", unified_memory=True)
    cap = {"cpu_jobs_gb": 8.0, "job_default_gb": 4.0}
    job = Job(kind="compile", argv=["true"])
    assert hostpolicy.mem_blocks(cfg, job, 0.0, cap) is None  # one job always fits
    assert hostpolicy.mem_blocks(cfg, job, 4.0, cap) is None
    assert "unified-memory cap" in (hostpolicy.mem_blocks(cfg, job, 8.0, cap) or "")
    big = Job(kind="compile", argv=["true"], mem_limit_mb=16 * 1024)
    assert hostpolicy.job_mem_gb(big, cap) == 16.0
    plain = WorkerConfig(name="v", transport="local")
    assert hostpolicy.mem_blocks(plain, big, 64.0, cap) is None  # not a unified-memory host
    assert baseline().unified_memory_cap["cpu_jobs_gb"] > 0


def test_host_mode_per_job() -> None:
    m = hostpolicy.host_mode
    assert m(Job(kind="shell", argv=["x"], requires={"pool": "agent"}), False, False) == "none"
    assert m(Job(kind="compile", argv=["x"]), False, False) == "shared"
    assert m(Job(kind="run", argv=["x"]), False, False) == "shared"
    assert m(Job(kind="measure", argv=["x"]), False, True) == "exclusive"
    assert m(Job(kind="measure", argv=["x"]), True, False) == "held"


# ---- quiet host: interfered measurements are requeued and never counted ---------------------


def reading(mono: float, busy: int, own: float, util: int = 0, procs: int = 1) -> dict[str, Any]:
    gpu = {"sm_mhz": 2400, "sm_max_mhz": 2418, "temp_c": 50, "throttle_reasons": 0,
           "gpu_util_pct": util, "compute_procs": procs}  # fmt: skip
    return {"mono_s": mono, "cpu_busy_jiffies": busy, "cpu_total_jiffies": 10_000, "hz": 100,
            "ncpu": 20, "own_cpu_s": own, "load1": 0.5, "mem_used_bytes": 8 * 2**30,
            "mem_total_bytes": 119 * 2**30, "gpu": gpu}  # fmt: skip


def timing(noisy: bool) -> dict[str, Any]:
    # 400 jiffies at HZ 100 over 1 s while the job itself used 0.9 s: ~3 other cores.
    end_busy = 1000 + (400 if noisy else 90)
    return {
        "times_ms": [[0.1, 0.1, 0.1]],
        "host_pre": reading(0.0, 990, 0.0),
        "host_start": reading(1.0, 1000, 0.1),
        "host_end": reading(2.0, end_busy, 1.0),
    }


def test_assess_marks_other_cpu_and_gpu_activity() -> None:
    th = baseline().quiet_thresholds
    assert not quiet.assess(timing(False), th).interfered
    v = quiet.assess(timing(True), th)
    assert v.interfered and v.verified and "other_cpu_cores" in v.reasons[0]
    busy_gpu = timing(False)
    busy_gpu["host_pre"] = reading(0.0, 990, 0.0, util=60, procs=2)
    v = quiet.assess(busy_gpu, th)
    assert {r.split()[0] for r in v.reasons} == {"gpu_util_pct_pre", "other_gpu_procs"}
    assert quiet.assess({"times_ms": []}, th).verified is False


def test_parse_proc_counters() -> None:
    busy, total = quiet.parse_stat("cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 1 2 3 4\n")
    assert (busy, total) == (150, 1000)
    mem = quiet.parse_meminfo("MemTotal: 1000 kB\nMemFree: 10 kB\nMemAvailable: 600 kB\n")
    assert mem["mem_used_bytes"] == 400 * 1024


SCRIPT = """
import json, pathlib, sys
state = pathlib.Path(sys.argv[1]); n = int(state.read_text()) if state.exists() else 0
state.write_text(str(n + 1))
plan = json.loads(sys.argv[2])
pathlib.Path("out").mkdir(exist_ok=True)
pathlib.Path("out/timing.json").write_text(json.dumps(plan[min(n, len(plan) - 1)]))
"""


def measure_job(tmp_path: Path, plan: list[dict[str, Any]]) -> Job:
    return Job(
        kind="shell",
        argv=["{python}", "-c", SCRIPT, str(tmp_path / "attempts"), json.dumps(plan)],
    )


def test_interfered_measurement_is_requeued_and_not_counted(
    hctx: Context, hstore: SqliteStore, hops: SqliteOps, tmp_path: Path
) -> None:
    register(hctx, hstore, hops, worker_configs(hctx)["local"], code_state().commit)
    ex = JobExecutor(hctx, hstore, hops, code_state().commit)
    m = Measurer(hstore, ex, hctx.clock, hctx.rng, 0, baseline().quiet_thresholds)
    result, data, verdict, interfered = m._run_quiet(  # pyright: ignore[reportPrivateUsage]
        measure_job(tmp_path, [timing(True), timing(False)])
    )
    assert result.ok and not verdict.interfered and (tmp_path / "attempts").read_text() == "2"
    assert len(interfered) == 1
    rep = hstore.get(interfered[0])
    assert (
        rep.body["subject"] == "measurement_interfered" and "other_cpu_cores" in rep.body["summary"]
    )
    assert hstore.query("Measurement", None, limit=10) == []  # the noisy times were not counted
    assert data["host_end"]["cpu_busy_jiffies"] == 1090


def test_a_host_that_never_quiets_raises_and_writes_no_measurement(
    hctx: Context, hstore: SqliteStore, hops: SqliteOps, tmp_path: Path
) -> None:
    register(hctx, hstore, hops, worker_configs(hctx)["local"], code_state().commit)
    ex = JobExecutor(hctx, hstore, hops, code_state().commit)
    th = {**baseline().quiet_thresholds, "max_requeues": 1}
    m = Measurer(hstore, ex, hctx.clock, hctx.rng, 0, th)
    with pytest.raises(HostNotQuiet, match="2 attempts") as err:
        m._run_quiet(measure_job(tmp_path, [timing(True)]))  # pyright: ignore[reportPrivateUsage]
    assert err.value.kind == "infra"
    reports = hstore.query("Report", {"subject": "measurement_interfered"}, limit=10)
    assert len(reports) == 2 and hstore.query("Measurement", None, limit=10) == []


# ---- stage queues: WIP limits and backpressure ----------------------------------------------

NOW = datetime(2026, 10, 31, 12, tzinfo=UTC)


def row(i: str, state: str, kind: str = "solve", age_s: float = 0, **data: Any) -> dict[str, Any]:
    since = (NOW - timedelta(seconds=age_s)).isoformat()
    return {"id": i, "kind": kind, "subject": f"case-{i}", "state": state,
            "data": {"state_since": since, **data}}  # fmt: skip


CAP = {"analyse": 4, "generate": 4, "build": 1, "check": 1, "ready_to_measure": 8,
       "measure": 1, "judge": 1}  # fmt: skip
WIN = {"window_s": 30.0, "max_items": 2, "exclusive_wait_s": 900.0}


def test_wip_backpressure_holds_upstream_when_downstream_is_full() -> None:
    rows = [row("a", "building"), row("b", "solving"), row("c", "retrieving")]
    p = stagequeue.plan(rows, CAP, WIN, NOW)
    # downstream first: "a" leaves build (check has room), which frees build for one loop
    assert p.advance == ["a", "b"]
    assert p.held == {"c": "backpressure: build full (1/1)"}
    # a loop this process cannot move (another process's) keeps check full: build waits
    rows = [row("x", "checking", process="experiment"), row("d", "building")]
    p = stagequeue.plan(rows, CAP, WIN, NOW)
    assert p.held["d"] == "backpressure: check full (1/1)" and "d" not in p.advance


def test_measurement_window_collects_until_full_or_due() -> None:
    one = stagequeue.plan([row("a", "measuring", age_s=5)], CAP, WIN, NOW)
    assert one.window == [] and "collecting" in one.held["a"]
    due = stagequeue.plan([row("a", "measuring", age_s=31)], CAP, WIN, NOW)
    assert due.window == ["a"]
    three = [row("a", "measuring"), row("b", "measuring"), row("c", "measuring")]
    full = stagequeue.plan(three, {**CAP, "judge": 4}, WIN, NOW)
    assert len(full.window) == 2 and "next measurement window" in next(iter(full.held.values()))
    busy = stagequeue.plan([row("a", "measuring", age_s=60)], CAP, WIN, NOW, open_windows=1)
    assert busy.window == [] and busy.held["a"].startswith("backpressure: measure full")
    judge_full = stagequeue.plan(
        [row("a", "measuring", age_s=60), row("j", "judging", process="experiment")], CAP, WIN, NOW
    )
    assert judge_full.window == [] and "judge full" in judge_full.held["a"]


def test_dequeue_prefers_work_that_unblocks_waiting_cases() -> None:
    cap = {**CAP, "build": 2}
    rows = [
        row("old", "solving", age_s=600),
        row("key", "solving", age_s=10, waiting_tasks=["t9"]),
        row("w", "waiting", waiting_on="t9"),  # a case waiting on a task loop "key" created
    ]
    p = stagequeue.plan(rows, cap, WIN, NOW)
    assert p.advance[:2] == ["key", "old"] or p.advance[0] == "key"
    assert stagequeue.unblocks(rows) == {"key": 1}


def test_unknown_states_are_outside_the_pipeline_and_expand_loops_fill_stages() -> None:
    rows = [row("x", "frozen", kind="expand"), row("s", "solving")]
    p = stagequeue.plan(rows, CAP, WIN, NOW)
    assert p.depth["build"] == 1 and p.held["s"].startswith("backpressure: build full")
    assert stagequeue.stage_of("solve", "no-such-state") is None


# ---- work unit = case x role ----------------------------------------------------------------


def test_work_unit_roles() -> None:
    u = workunits.unit_for
    assert u("L1", "T1", "case_role", "k1") == u("L1", "T6", "case_role", "k2") == "L1:kernel"
    assert u("L1", "T4", "case_role", "k") == u("L1", "T5", "case_role", "k") == "L1:model"
    assert u("L1", "T2", "case_role", "k") == u("L1", "T3", "case_role", "k") == "L1:knowledge"
    assert u("L1", "T9", "case_role", "k") == "L1:other"
    assert u("L1", "T4", "case", "k") == "L1"
    assert u("L1", "T1", "step", "a") != u("L1", "T1", "step", "b")
    assert u("human:7", "T4", "case_role", "k") == "human:7"
    assert workunits.loop_of("L1:model") == "L1" and workunits.loop_of("human:7") == "human:7"
    assert baseline().work_unit_rule == "case_role"
    kinds = {t for ts in workunits.roles().values() for t in ts}
    assert kinds == {"T1", "T2", "T3", "T4", "T5", "T6"}
