"""WP3 acceptance (spec/modules/measure-check.md "验收"), run on the real GPU worker.

The control plane is this process with a temporary test store; spark is reached over
the ``ssh`` transport and must have this commit deployed (``scripts/worker-setup
spark``). Every GPU job goes through the Executor.

Thresholds: the only timing assertion is the one the specification states (a ~5%
slower kernel is significant, an identical one is not); the noise floor itself is
reported, not asserted.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from kta.backends import CudaBackend
from kta.checks import selftest as check_selftest
from kta.checks.base import Checker
from kta.core.config import default_config_path, load_config
from kta.core.context import Context
from kta.core.errors import NotSupported
from kta.core.ids import record_id
from kta.core.registry import Registry
from kta.core.schemas.hardware import Target
from kta.execution.executor import JobExecutor
from kta.execution.registry import register, worker_configs
from kta.measure import noise, protocol, selftest
from kta.measure.profile import run_profile
from kta.measure.runner import Measurer
from kta.store import SqliteStore, code_state, open_ops, open_store
from kta.targets.gb10 import Gb10Target

pytestmark = [pytest.mark.gpu, pytest.mark.acceptance]


@dataclass
class Plane:
    store: SqliteStore
    m: Measurer
    c: Checker
    cuda: CudaBackend
    plugin: Gb10Target
    target: Target
    target_id: str


@pytest.fixture(scope="module")
def plane(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Plane]:
    data = tmp_path_factory.mktemp("wp3")
    ctx = Context(
        env="test",
        config=load_config(default_config_path()),
        data_dir=data,
        registry=Registry(),
    )
    store, ops = open_store(ctx), open_ops(ctx)
    commit = code_state().commit
    register(ctx, store, ops, worker_configs(ctx)["spark"], commit)
    ex = JobExecutor(ctx, store, ops, commit)
    plugin = Gb10Target(store, ctx.clock)
    target = plugin.probe(ex)
    target_id = plugin.latest_target_id()
    m = Measurer(store, ex, ctx.clock, ctx.rng, ctx.seed)
    c = Checker(store, ex, ctx.clock, ctx.rng, data / "checker", ctx.seed)
    yield Plane(store, m, c, CudaBackend(store, ctx.clock, plugin), plugin, target, target_id)


def test_checker_passes_the_correct_kernel_and_names_each_fault(plane: Plane) -> None:
    flags = plane.plugin.compile_flags("cuda")
    rid, results = check_selftest.run_all(plane.c, plane.cuda, plane.target, plane.target_id, flags)
    by = {r.variant: r for r in results}
    assert by["correct"].check.passed and not by["correct"].check.checker_error
    assert "sanitizer.memcheck" in by["oob_write"].failed_checks
    assert {"anticheat.nan_coverage", "numeric"} <= by["missing_row"].failed_checks
    assert {"anticheat.libraries", "anticheat.kernels"} <= by["cublas"].failed_checks
    for r in results:
        assert not r.check.checker_error, r.variant
        rec = plane.store.get(r.check.id)
        assert rec.kind == "Check" and rec.actor.type == "checker"
        assert rec.refs["worker"] and rec.refs["env"] and len(rec.refs["checks"]) >= 6
    assert plane.store.get(rid).body["data"]["cublas"]["as_expected"] is True


def test_noise_probe_and_revised_protocol(plane: Plane) -> None:
    v1 = protocol.ensure_v1(plane.store, plane.m.clock)
    w = selftest.workload(plane.store, plane.m.clock, "vadd", plane.target_id)
    module = selftest.build(
        plane.cuda, selftest.vadd_kernel(w.shape["n"]), plane.target, plane.m.ex
    )
    rid, data = noise.noise_probe(plane.m, module, w.case, w.shape, v1, 30, "vadd")
    assert data["groups"] == 30 and len(plane.store.get(rid).refs["measurements"]) == 30
    first = plane.store.get(str(plane.store.get(rid).refs["measurements"][0]))
    assert first.body["protocol"] == v1 and first.body["n"] >= protocol.V1.min_iters
    worker = plane.store.get(str(first.refs["worker"]))
    assert worker.body["labels"]["gpu"] == plane.target.sm
    assert first.body["clocks"]["sm_mhz_start"] and first.body["temps"]["temp_c_start"]
    v2 = noise.revise_protocol(plane.m, v1, [rid])
    params = protocol.load_params(plane.store, v2)
    assert params.noise_floor_rel == noise.floor_from([data["cv"]])
    assert params == protocol.revise(protocol.V1, params.noise_floor_rel)
    assert plane.store.get(v2).body["version"] == 2


def test_compare_separates_a_five_percent_kernel_and_not_an_identical_one(plane: Plane) -> None:
    pid = protocol.find_protocol(plane.store, protocol.NAME, 2) or protocol.ensure_v1(
        plane.store, plane.m.clock
    )
    w = selftest.workload(plane.store, plane.m.clock, "fma", plane.target_id)
    base = selftest.build(plane.cuda, selftest.fma_kernel(1000), plane.target, plane.m.ex)
    slower = selftest.build(plane.cuda, selftest.fma_kernel(1050), plane.target, plane.m.ex)
    differ = plane.m.compare(base, slower, w.case, w.shape, pid)
    assert differ.significant and 0.02 < differ.rel_diff < 0.08, differ
    same = plane.m.compare(base, base, w.case, w.shape, pid)
    assert not same.significant, same
    report = plane.store.get(differ.report_id)
    assert report.refs["measurements"] == [differ.a.id, differ.b.id]
    assert record_id(plane.store.get(differ.a.id)) == differ.a.id


def test_profile_writes_a_profile_or_reports_not_supported(plane: Plane) -> None:
    w = selftest.workload(plane.store, plane.m.clock, "gemm", plane.target_id)
    small = {"M": 256, "N": 256, "K": 256}
    module = selftest.build(plane.cuda, selftest.gemm_kernel(small), plane.target, plane.m.ex)
    try:
        pid, body = run_profile(plane.m, module, w.case, small)
    except NotSupported as exc:
        assert "ERR_NVGPUCTRPERM" in str(exc)
        return
    assert body.metrics and plane.store.get(pid).kind == "Profile"
