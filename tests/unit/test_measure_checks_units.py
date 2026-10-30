"""Unit tests of WP3's pure logic: statistics, protocols, inputs, parsers, verdict rules.

GPU behaviour (bench jobs, sanitizer, nsys, references) is covered by the acceptance
tests in tests/acceptance/WP3, which run on the real worker.
"""

from __future__ import annotations

import json
import random
import sqlite3
import struct
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from kta.backends.base import BuiltModule, LaunchSpec, Param
from kta.checks import elf, hidden
from kta.checks.anticheat import library_failures, traced_kernels
from kta.checks.numeric import compare_output, dists_for
from kta.checks.sass import opcode_counts, sass_failures
from kta.core.context import Context
from kta.core.errors import ConfigError, NotSupported
from kta.core.jobs import JobRunner
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Contract, TensorSpec
from kta.core.schemas.knowledge import SassExpect
from kta.measure import inputs, jobs, noise, protocol, stats
from kta.measure.profile import parse_ncu_csv
from kta.measure.selftest import PROBLEMS
from kta.store import SqliteStore
from storehelp import T0

# ---- statistics ---------------------------------------------------------------------


def test_bootstrap_median_brackets_the_median() -> None:
    values = [10.0 + 0.01 * i for i in range(201)]
    med, lo, hi = stats.bootstrap_median(values, 1000, 0.95, random.Random(1))
    assert med == pytest.approx(11.0)
    assert lo <= med <= hi and hi - lo < 0.5


def test_bootstrap_is_reproducible_from_the_seed() -> None:
    values = list(np.random.default_rng(0).normal(100, 1, 300))
    a = stats.bootstrap_median(values, 500, 0.95, random.Random(7))
    assert a == stats.bootstrap_median(values, 500, 0.95, random.Random(7))


def test_rel_diff_ci_contains_true_difference() -> None:
    g = np.random.default_rng(3)
    a = list(g.normal(100.0, 0.2, 400))
    b = list(g.normal(105.0, 0.2, 400))
    rel, lo, hi = stats.bootstrap_rel_diff(a, b, 1000, 0.95, random.Random(2))
    assert lo < 0.05 < hi or abs(rel - 0.05) < 0.002


def test_significance_needs_disjoint_cis_and_a_difference_above_the_floor() -> None:
    v = stats.significance((99, 100), (104, 105), 0.05, 0.01, 3.0)
    assert v.significant and v.cis_disjoint and v.above_floor
    small = stats.significance((99, 100), (101, 102), 0.02, 0.01, 3.0)
    assert small.cis_disjoint and not small.above_floor and not small.significant
    overlap = stats.significance((99, 106), (104, 105), 0.05, 0.01, 3.0)
    assert not overlap.significant


def test_noise_summary_cv() -> None:
    s = stats.noise_summary([100.0, 101.0, 99.0, 100.0])
    assert s["groups"] == 4 and s["median_of_medians_us"] == 100.0
    assert s["cv"] == pytest.approx(np.std([100, 101, 99, 100], ddof=1) / 100)
    assert s["dev_max"] == pytest.approx(0.01)


# ---- protocols ------------------------------------------------------------------------


def test_v1_is_written_once_and_loads_back(store: SqliteStore, ctx: Context) -> None:
    pid = protocol.ensure_v1(store, ctx.clock)
    assert protocol.ensure_v1(store, ctx.clock) == pid
    assert protocol.load_params(store, pid) == protocol.V1
    body = store.get(pid).body
    assert (body["name"], body["version"], body["type"]) == ("measure", 1, "measurement")


def test_a_version_cannot_change_its_params(store: SqliteStore, ctx: Context) -> None:
    protocol.ensure_v1(store, ctx.clock)
    with pytest.raises(ConfigError, match="new version"):
        protocol.ensure_protocol(store, ctx.clock, 1, protocol.revise(protocol.V1, 0.5), "x")


def test_unknown_protocol_params_are_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown"):
        protocol.MeasureParams.from_dict({"min_iters": 5, "lock_clocks_harder": True})


def test_noise_floor_rounds_up_to_a_tenth_of_a_percent() -> None:
    assert noise.floor_from([0.0012, 0.0035]) == 0.004
    assert noise.floor_from([0.002]) == 0.002


def test_load_params_rejects_other_kinds(store: SqliteStore, ctx: Context) -> None:
    rid = store.put(
        Record.new(
            "Protocol",
            {"name": "x", "version": 1, "type": "experiment", "description": "d", "params": {}},
            Actor(type="tool", name="t"),
            T0,
        )
    )
    with pytest.raises(ConfigError, match="not a measurement"):
        protocol.load_params(store, rid)


# ---- inputs ---------------------------------------------------------------------------


def test_tensor_dims_are_expressions_over_the_shape() -> None:
    t = TensorSpec(name="A", shape=["M", "K // 2"], dtype="f16", layout="row_major")
    assert inputs.dims(t, {"M": 4, "K": 8}) == (4, 4)
    assert inputs.nbytes(t, {"M": 4, "K": 8}) == 32


def test_generators_are_seeded_and_typed() -> None:
    t = TensorSpec(name="A", shape=["n"], dtype="f16", layout="contiguous")
    a = inputs.generate(t, {"n": 64}, "normal", random.Random(5))
    b = inputs.generate(t, {"n": 64}, "normal", random.Random(5))
    assert a.dtype == np.float16 and np.array_equal(a, b)
    assert not inputs.generate(t, {"n": 64}, "zeros", random.Random(5)).any()
    with pytest.raises(NotSupported):
        inputs.generate(t, {"n": 4}, "no_such_dist", random.Random(0))


def test_bf16_has_no_host_dtype_yet() -> None:
    with pytest.raises(NotSupported, match="bf16"):
        inputs.np_dtype("bf16")


# ---- bench jobs -----------------------------------------------------------------------


def _build(store: SqliteStore, launch: LaunchSpec) -> BuiltModule:
    key = store.put_blob(json.dumps({"launch": launch.to_dict()}).encode(), "application/json")
    env = store.put(
        Record.new(
            "EnvManifest",
            {
                "role": "worker",
                "host": "h",
                "os": "o",
                "arch": "a",
                "python": "p",
                "uv": "u",
                "uv_lock_sha256": "x",
                "commit": "c",
                "dirty": False,
            },
            Actor(type="tool", name="t"),
            T0,
        )
    )
    worker = store.put(
        Record.new(
            "Worker",
            {
                "name": "w",
                "host": "h",
                "transport": "ssh",
                "labels": {},
                "commit": "c",
                "devices": [],
            },
            Actor(type="tool", name="t"),
            T0,
            refs={"env": env},
        )
    )
    module = store.put_blob(b"\x7fELF", "application/octet-stream")
    bid = store.put(
        Record.new(
            "Build",
            {"ok": True, "log_summary": ""},
            Actor(type="tool", name="t"),
            T0,
            refs={"env": env, "worker": worker},
            blobs={"build_key": key, "module": module},
        )
    )
    return BuiltModule(module, "cubin", {}, None, "gemm", bid)


def test_bench_job_takes_launch_from_the_build_and_scalars_from_the_shape(
    store: SqliteStore,
) -> None:
    launch = LaunchSpec(
        [2, 2],
        [16, 16],
        0,
        [
            Param("A", "f16*"),
            Param("B", "f16*"),
            Param("C", "f16*"),
            Param("M", "i32"),
            Param("N", "i32"),
            Param("K", "i32"),
        ],
    )
    module = _build(store, launch)
    shape = {"M": 32, "N": 32, "K": 16}
    sets = [jobs.random_inputs(PROBLEMS["gemm"], shape, "normal", random.Random(0))]
    ex = cast(JobRunner, store)  # only put_blob is used while building the job
    bj = jobs.bench_job(
        ex,
        store,
        [module],
        PROBLEMS["gemm"],
        shape,
        sets,
        "run",
        "check",
        {"gpu": "sm_121"},
        {"repeat": 2},
        wrap=["tool"],
    )
    assert bj.job.argv[0] == "tool" and bj.job.argv[-1] == "spec.json"
    assert bj.spec["scalars"] == shape and bj.spec["outputs"] == ["C"]
    assert bj.spec["buffers"]["C"]["bytes"] == 32 * 32 * 2
    assert bj.spec["modules"][0]["launch"]["grid"] == [2, 2]
    assert set(bj.job.inputs) == {"m0.cubin", "in/s0_A.bin", "in/s0_B.bin", "spec.json"}
    with pytest.raises(ConfigError, match="not a shape variable"):
        jobs.bench_job(
            ex, store, [module], PROBLEMS["gemm"], {"M": 32, "N": 32}, [], "run", "check", {}
        )


# ---- numeric verdicts -----------------------------------------------------------------


TOL = {"rtol": 1e-2, "atol": 1e-2, "rel_l2": 1e-3}


def test_compare_output_accepts_rounding_and_rejects_unwritten_elements() -> None:
    ref = np.linspace(-4, 4, 100)
    ok = compare_output(ref.astype(np.float16), ref, TOL)
    assert ok["mismatches"] == 0 and ok["nan_count"] == 0 and ok["rel_l2"] < 1e-3
    got = ref.astype(np.float16)
    got[-10:] = np.nan
    bad = compare_output(got, ref, TOL)
    assert bad["mismatches"] == 10 and bad["nan_count"] == 10 and bad["rel_l2"] is None


def test_compare_output_matches_nan_and_inf_of_the_reference() -> None:
    ref = np.array([np.nan, np.inf, 1.0])
    out = compare_output(np.array([np.nan, np.inf, 1.0], dtype=np.float32), ref, TOL)
    assert out["mismatches"] == 0


def test_dists_merge_the_contract_and_numeric_risks() -> None:
    c = Contract(input_dists=["normal", "zeros"], timing_boundary="x")
    assert dists_for(c, ["extreme", "normal"]) == ["normal", "zeros", "extreme"]
    with pytest.raises(ConfigError):
        dists_for(Contract(timing_boundary="x"), None)


# ---- anti-cheat parsers ---------------------------------------------------------------


def _elf(needed: list[str], undefined: list[str]) -> bytes:
    strtab = b"\0"
    offs: dict[str, int] = {}
    for s in [*needed, *undefined]:
        offs[s] = len(strtab)
        strtab += s.encode() + b"\0"
    dynamic = b"".join(struct.pack("<qQ", 1, offs[n]) for n in needed) + struct.pack("<qQ", 0, 0)
    dynsym = b"\0" * 24 + b"".join(
        struct.pack("<IBBHQQ", offs[u], 0x12, 0, 0, 0, 0) for u in undefined
    )
    body = strtab + dynamic + dynsym
    base = 64
    shoff = base + len(body)
    sections = [
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 3, 0, 0, base, len(strtab), 0, 0, 1, 0),
        (0, 6, 0, 0, base + len(strtab), len(dynamic), 1, 0, 8, 16),
        (0, 11, 0, 0, base + len(strtab) + len(dynamic), len(dynsym), 1, 0, 8, 24),
    ]
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<Q", header, 0x28, shoff)
    struct.pack_into("<HH", header, 0x3A, 64, len(sections))
    shdrs = b"".join(struct.pack("<IIQQQQIIQQ", *s) for s in sections)
    return bytes(header) + body + shdrs


def test_elf_dynamic_section_lists_needed_libraries_and_imports() -> None:
    data = _elf(["libcublas.so.13", "libc.so.6"], ["cublasGemmEx", "memcpy"])
    info = elf.dynamic_info(data)
    assert info.needed == ["libcublas.so.13", "libc.so.6"]
    assert info.undefined == ["cublasGemmEx", "memcpy"]
    (failure,) = library_failures(data)
    assert failure["needed"] == ["libcublas.so.13"] and failure["symbols"] == ["cublasGemmEx"]
    assert library_failures(_elf(["libc.so.6"], ["memcpy"])) == []


def test_traced_kernels_reads_an_nsys_export(tmp_path: Path) -> None:
    db = tmp_path / "t.sqlite"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE StringIds (id INTEGER, value TEXT);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (shortName INTEGER);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (bytes INTEGER);"
        "INSERT INTO StringIds VALUES (1, 'gemm'), (2, 'nvjet_gemm');"
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1), (2), (1);"
    )
    con.commit()
    con.close()
    assert sorted(traced_kernels(db.read_bytes()) or []) == ["gemm", "gemm", "nvjet_gemm"]
    empty = tmp_path / "e.sqlite"
    sqlite3.connect(empty).close()
    assert traced_kernels(empty.read_bytes()) is None


# ---- SASS -----------------------------------------------------------------------------

SASS = """
        /*0000*/                   LDC R1, c[0x0][0x28] ;
        /*0010*/              @!P0 BRA 0x80 ;
        /*0020*/                   HMMA.16816.F32 R4, R8, R12, R4 ;
        /*0030*/                   LDGSTS.E.BYPASS.128 [R2], desc[UR4][R6.64] ;
"""


def test_sass_opcodes_and_expectations() -> None:
    counts = opcode_counts(SASS)
    assert counts["HMMA.16816.F32"] == 1 and counts["BRA"] == 1
    expect = [
        SassExpect(when="stages >= 2", must_contain=["LDGSTS"], must_not_contain=["LDS"]),
        SassExpect(when="use_mma", must_contain=["HMMA", "IMMA"]),
    ]
    failures = sass_failures(counts, expect, {"stages": 3, "use_mma": True})
    assert [f["missing"] for f in failures] == [["IMMA"]]
    assert failures[0]["failure"] == "intent_not_realized"
    assert sass_failures(counts, expect, {"stages": 1, "use_mma": False}) == []


# ---- hidden shapes, ncu ----------------------------------------------------------------


def test_hidden_seed_is_private_and_shapes_are_stable(tmp_path: Path) -> None:
    seed = hidden.private_seed(tmp_path / "checker")
    assert hidden.private_seed(tmp_path / "checker") == seed
    assert (tmp_path / "checker" / "hidden_seed").stat().st_mode & 0o077 == 0
    base = {"M": 128, "N": 128, "K": 128, "dtype": "f16"}
    shapes = hidden.hidden_shapes(seed, base)
    assert [c for _, c in shapes] == list(hidden.CATEGORIES)
    assert shapes == hidden.hidden_shapes(seed, base)
    assert all(s["dtype"] == "f16" for s, _ in shapes)
    assert shapes[0][0]["M"] % 16 != 0 and 1 in shapes[1][0].values()
    assert hidden.hidden_shapes("other-seed", base) != shapes


def test_ncu_csv_rows_of_the_kernel_become_metrics() -> None:
    text = "\n".join(
        [
            "==PROF== Connected to process",
            '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"',
            '"0","gemm","gpu__time_duration.sum","ns","1,234"',
            '"0","gemm","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","56.7"',
            '"1","other","gpu__time_duration.sum","ns","9"',
        ]
    )
    metrics: dict[str, Any] = parse_ncu_csv(text, "gemm")
    assert metrics == {
        "gpu__time_duration.sum": 1234.0,
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": 56.7,
    }
