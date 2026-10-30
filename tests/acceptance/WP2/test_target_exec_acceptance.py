"""WP2 acceptance (spec/modules/target-exec.md "验收"), run on the real workers.

The control plane is this process with a temporary test store; spark and slai are
reached over the ``ssh`` transport configured in config/kta.toml and must have this
commit deployed (``scripts/worker-setup spark`` / ``slai``). Every GPU job goes through
the Executor to spark.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from kta.backends import CudaBackend, TileLangBackend
from kta.backends.smoke import Smoke
from kta.core.config import default_config_path, load_config
from kta.core.context import Context
from kta.core.locks import file_sha256
from kta.core.registry import Registry
from kta.core.schemas.hardware import Target
from kta.execution.executor import JobExecutor
from kta.execution.registry import Registration, register, worker_configs
from kta.store import SqliteOps, SqliteStore, code_state, open_ops, open_store
from kta.targets.gb10 import Gb10Target
from kta.targets.gb10.probe import load_features

pytestmark = [pytest.mark.gpu, pytest.mark.acceptance]
ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Plane:
    ctx: Context
    store: SqliteStore
    ops: SqliteOps
    spark: Registration
    ex: JobExecutor
    target: Target
    plugin: Gb10Target


@pytest.fixture(scope="module")
def plane(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Plane]:
    ctx = Context(
        env="test",
        config=load_config(default_config_path()),
        data_dir=tmp_path_factory.mktemp("wp2"),
        registry=Registry(),
    )
    store, ops = open_store(ctx), open_ops(ctx)
    commit = code_state().commit
    spark = register(ctx, store, ops, worker_configs(ctx)["spark"], commit)
    ex = JobExecutor(ctx, store, ops, commit)
    plugin = Gb10Target(store, ctx.clock)
    target = plugin.probe(ex)
    yield Plane(ctx, store, ops, spark, ex, target, plugin)


def test_spark_registers_as_gpu_worker_over_ssh(plane: Plane) -> None:
    labels = plane.spark.worker.labels
    assert plane.spark.worker.transport == "ssh"
    assert labels["gpu"] == "sm_121" and labels["cuda"].startswith("13.")
    assert {"pool.gpu-exclusive", "pool.gpu-shared", "pool.cpu", "backend.cuda"} <= set(labels)
    assert plane.spark.worker.devices == ["0"]
    manifest = plane.store.get(plane.spark.env_id).body
    assert manifest["role"] == "worker" and manifest["commit"] == code_state().commit
    assert manifest["uv_lock_sha256"] == file_sha256(ROOT / "uv.lock")


def test_probe_writes_target_facts_and_feature_support(plane: Plane) -> None:
    t = plane.target
    assert (t.name, t.sm, t.gpu) == ("gb10", "sm_121", "NVIDIA GB10")
    assert t.driver and t.cuda.startswith("13.") and "cuda" in t.backends
    limits = plane.plugin.limits()
    assert limits["sm_count"] > 0 and limits["smem_per_cta"] >= 48 * 1024
    assert limits["warp_size"] == 32
    features = {f.name: f for f in plane.plugin.features()}
    assert set(features) == {f.name for f in load_features()}
    support = {name: f.support["gb10"] for name, f in features.items()}
    assert support["ld_st_global"] == "yes" and support["mma_f16_f32acc"] == "yes"
    assert set(support.values()) <= {"yes", "no", "unknown"}


def test_vector_add_on_both_backends_is_correct(plane: Plane) -> None:
    smoke = Smoke(plane.store, plane.ops, plane.ctx.clock, plane.ex, plane.target)
    cuda = CudaBackend(plane.store, plane.ctx.clock, plane.plugin)
    tilelang = TileLangBackend(plane.store, plane.ctx.clock, plane.plugin)
    for backend in (cuda, tilelang):
        result = smoke.vector_add(backend, seed=7)
        assert result.build_ok and result.passed, (backend.name, result.detail)
        build = plane.store.get(result.build_id)
        assert build.refs["worker"] == plane.spark.worker_id
        assert build.refs["env"] == plane.spark.env_id and "module" in build.blobs
    again = smoke.vector_add(cuda, seed=8)
    assert again.build_id == smoke.vector_add(cuda, seed=7).build_id  # input-hash cache


def test_exclusive_job_defers_concurrent_shared_jobs(plane: Plane) -> None:
    smoke = Smoke(plane.store, plane.ops, plane.ctx.clock, plane.ex, plane.target)
    cuda = CudaBackend(plane.store, plane.ctx.clock, plane.plugin)
    report = plane.store.get(smoke.lease_deferral(cuda, lambda: plane.ex.lease_held("spark")))
    assert report.body["data"]["deferred_until_exclusive_end"] is True
    assert all(report.body["data"]["ok"].values())


def test_gpu_is_usable_after_a_timed_out_kernel(plane: Plane) -> None:
    smoke = Smoke(plane.store, plane.ops, plane.ctx.clock, plane.ex, plane.target)
    cuda = CudaBackend(plane.store, plane.ctx.clock, plane.plugin)
    report = plane.store.get(smoke.timeout_recovery(cuda, seed=9, timeout_s=5))
    assert report.body["data"]["spin"]["error_kind"] == "timeout"
    assert report.body["data"]["after"]["passed"] is True


def test_slai_cpu_worker_compiles_with_the_same_lock(plane: Plane) -> None:
    slai = register(
        plane.ctx, plane.store, plane.ops, worker_configs(plane.ctx)["slai"], code_state().commit
    )
    assert "gpu" not in slai.worker.labels and slai.worker.labels["arch"] == "x86_64"
    ex = JobExecutor(plane.ctx, plane.store, plane.ops, code_state().commit)
    smoke = Smoke(plane.store, plane.ops, plane.ctx.clock, ex, plane.target)
    build = plane.store.get(smoke.cpu_compile({"arch": "x86_64"}))
    assert build.body["ok"] is True and build.refs["worker"] == slai.worker_id
    spark_lock = plane.store.get(plane.spark.env_id).body["uv_lock_sha256"]
    assert plane.store.get(slai.env_id).body["uv_lock_sha256"] == spark_lock
