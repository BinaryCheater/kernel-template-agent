"""WP7 additions below the orchestrator: peaks, baseline, TileLang launch checks, inputs."""

from __future__ import annotations

import random

import numpy as np

from kta.backends.base import GeneratedKernel, LaunchSpec, Param
from kta.backends.tilelang import launch_mismatch
from kta.checks.inputs import TABLE
from kta.core.context import Context
from kta.measure.baseline import TORCH_MATMUL, ensure_baseline
from kta.measure.peak_kernels import kernels
from kta.measure.peaks import MODEL_ASSUMPTIONS, next_version, workload
from kta.store.base import Store
from kta_wp7.stats import ranks, spearman


def test_spearman_with_ties() -> None:
    assert ranks([3.0, 1.0, 3.0]) == [2.5, 1.0, 2.5]
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([1, 2], [1, 2]) is None


def test_peak_kernels_count_work_per_unit() -> None:
    ks = {k.unit: k for k in kernels(48)}
    assert set(ks) == {"dram", "l2", "smem", "tensor", "fma", "launch"}
    fma = ks["fma"].work(ks["fma"].shape)
    assert fma["fma"] == 32 * fma["issue"]  # FMAs per thread vs warp instructions
    tensor = ks["tensor"].work(ks["tensor"].shape)["tensor"]
    assert tensor == 2 * 16 * 8 * 16 * 8 * (48 * 4 * 256 // 32) * 2048
    for k in ks.values():
        names = {p.name for p in k.params if not p.is_pointer}
        assert names <= set(k.shape), k.unit  # scalars come from the shape
        assert k.entry in k.source


def test_peak_workload_records(ctx: Context, store: Store) -> None:
    k = kernels(48)[0]
    a = workload(store, ctx.clock, k, "target-id")
    b = workload(store, ctx.clock, k, "target-id")
    assert a == b
    assert store.get(a[0]).refs["contract"]
    assert next_version(store, "gb10") == 1
    assert set(MODEL_ASSUMPTIONS) == {"smem_queue_depth", "smem_banks", "smem_bank_bytes"}


def test_baseline_record_is_fixed_once(ctx: Context, store: Store) -> None:
    pid = workload(store, ctx.clock, kernels(48)[0], "target-id")[1].problem
    a = ensure_baseline(store, ctx.clock, pid, TORCH_MATMUL, "torch", "sha256:x")
    b = ensure_baseline(store, ctx.clock, pid, TORCH_MATMUL, "torch", "sha256:x")
    c = ensure_baseline(store, ctx.clock, pid, TORCH_MATMUL, "torch", "sha256:y")
    assert a == b != c
    assert store.get(a).body["subject"] == "baseline"


def _gk(grid: list[int], smem: int) -> GeneratedKernel:
    params = [Param("A", "f16*"), Param("B", "f16*"), Param("C", "f16*")]
    return GeneratedKernel("tilelang", {}, "gemm", LaunchSpec(grid, [128], smem, params))


def test_tilelang_launch_mismatch_grid_and_dynamic_smem() -> None:
    gen = {"block": 128, "params": ["A", "B", "C"], "grid": [32, 32, 1], "dynamic_smem": 65536}
    assert launch_mismatch(_gk([32, 32], 65536), gen) is None
    assert launch_mismatch(_gk([32, 32], 98304), gen) is None  # more is harmless
    assert "dynamic shared memory" in str(launch_mismatch(_gk([32, 32], 1024), gen))
    assert "grid" in str(launch_mismatch(_gk([32, 16], 65536), gen))
    assert launch_mismatch(_gk([32, 16], 0), {**gen, "grid": None, "dynamic_smem": None}) is None


def test_large_dynamic_range_inputs() -> None:
    g = np.random.default_rng(random.Random(0).getrandbits(63))
    x = TABLE["large_dynamic_range"]((64, 64), "f16", g)
    mags = np.abs(x)
    assert mags.min() >= 2.0**-6 and mags.max() <= 2.0**3
    assert (x < 0).any() and (x > 0).any()
