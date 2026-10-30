"""WP7 GEMM family: validation, generation, formulation costs, legality fault, reuse plan."""

from __future__ import annotations

import random
from typing import Any

from knowhelp import seed
from kta.core.context import Context
from kta.core.schemas.evaluation import Case
from kta.core.schemas.hardware import HwModel
from kta.formulation import assemble
from kta.knowledge import errors_of, load_family, validate_family
from kta.models import UnitModel
from kta.solve import solve
from kta.store.base import Store
from kta_wp7 import records as r
from kta_wp7.faults import WRONG_GLUE, wrong_legality
from kta_wp7.pipeline import Generator, sass_for, templates_of
from kta_wp7.reuse import EXPECTED_INVALID, run_reuse

# Device limits as WP2's probe reports them for GB10, plus the model assumptions.
LIMITS: dict[str, int | float] = {
    "sm_count": 48,
    "max_ctas_per_sm": 24,
    "max_threads_per_sm": 1536,
    "smem_per_sm": 102400,
    "smem_per_cta": 101376,
    "smem_reserved_per_cta": 1024,
    "regs_per_sm": 65536,
    "smem_queue_depth": 1,
    "smem_banks": 32,
    "smem_bank_bytes": 4,
}
# Illustrative peaks for unit tests (not GB10 measurements).
HW = HwModel(
    target="test-gpu",
    version=1,
    peaks={
        "tensor": 1.0e14,
        "issue": 4.6e11,
        "l2": 2.0e12,
        "dram": 2.5e11,
        "smem": 1.15e11,
        "fma": 1.47e13,
        "launch": 2.0e9,
    },
    limits=LIMITS,
)
SHAPES = [{"M": 4096, "N": 4096, "K": 4096}, {"M": 1024, "N": 1024, "K": 16384}]


def _case() -> Case:
    return Case(
        problem="gemm_fp16",
        shape_dist=[{"shape": s, "weight": 1.0} for s in SHAPES],
        objective="weighted_latency",
        target="test-gpu",
    )


def _solve(ctx: Context, store: Store, wrong: bool = False, bucket: int = 0) -> Any:
    seed(ctx, store)
    fp = load_family(r.FAMILY_DIR)
    if wrong:
        fp = wrong_legality(fp)
    f = assemble(fp, fp.chain, templates_of(store, fp), _case(), LIMITS)
    b = {"name": f"s{bucket}", "shape": SHAPES[bucket], "weight": 1.0}
    return solve(f, _case(), b, UnitModel(target="test-gpu"), HW, rng=random.Random(0))


def test_family_validates_with_full_markers(ctx: Context, store: Store) -> None:
    seed(ctx, store)
    fp = load_family(r.FAMILY_DIR)
    assert errors_of(validate_family(fp, store, LIMITS)) == []
    markers = {m for item in fp.chain.items for m in item.code_markers}
    gk = Generator(fp)({"BM": 128, "BN": 128, "BK": 64, "S": 3, "threads": 128, "split": 1}, {})
    for m in markers:
        assert f"kta:step {m} begin" in gk.files["kernel.py"]
        assert f"kta:step {m} end" in gk.files["kernel.py"]
    assert set(gk.markers) == markers


def test_generator_launch_spec() -> None:
    gen = Generator(load_family(r.FAMILY_DIR))
    a = {"BM": 128, "BN": 256, "BK": 32, "S": 4, "threads": 256, "split": 1}
    gk = gen(a, {"M": 1000, "N": 3000, "K": 512})
    assert gk.launch.grid == [12, 8]
    assert gk.launch.block == [256]
    assert gk.launch.smem_bytes == 4 * (128 + 256) * 32 * 2
    assert [p.name for p in gk.launch.params] == ["A", "B", "C"]
    split = gen({**a, "split": 2}, {"M": 1000, "N": 3000, "K": 512})
    assert split.launch.grid == [12, 8, 2]
    assert "bz * K_PER" in split.files["kernel.py"]


def test_formulation_costs_contraction_and_traffic(ctx: Context, store: Store) -> None:
    res = _solve(ctx, store)
    assert res.exhaustive
    best = res.candidates[0].prediction
    assert {"tensor", "l2", "dram", "issue", "smem"} <= set(best.per_unit)
    assert best.serial > 0  # launch slots are serial
    totals = [c.prediction.total for c in res.candidates]
    # Distinct predictions among the shortlist: the WP5 issue (top 8 tie) is gone.
    assert len({round(t, 6) for t in totals[:16]}) > 1
    assert all(c.assignment["split"] == 1 for c in res.candidates)
    assert res.pruned_by.get("glue:g_no_combine", 0) > 0


def test_wrong_legality_admits_racy_split(ctx: Context, store: Store) -> None:
    res = _solve(ctx, store, wrong=True, bucket=1)
    assert any(c.assignment["split"] == 2 for c in res.candidates)
    assert "glue:g_no_combine" not in res.pruned_by
    fp = wrong_legality(load_family(r.FAMILY_DIR))
    glue = {g.id: g.expr for g in fp.family.glue_constraints}
    assert glue["g_no_combine"] == WRONG_GLUE


def test_sass_expectation_from_constant_binding(ctx: Context, store: Store) -> None:
    seed(ctx, store)
    fp = load_family(r.FAMILY_DIR)
    a = {"BM": 128, "BN": 128, "BK": 64, "S": 2, "P": 1, "threads": 128, "split": 1, "KT": 64}
    sass, _ = sass_for(fp, templates_of(store, fp), a)
    assert [s.must_contain for s in sass] == [["HMMA.16816.F32"]]


def test_reuse_plan_marks_only_the_instruction_step(ctx: Context, store: Store) -> None:
    seed(ctx, store)
    ids = {
        label: r.write_case(store, ctx.clock, path, "test-target", None)
        for label, path in (("bf16", r.CASE_BF16), ("fp16", r.CASE_FP16))
    }
    _, chain_id, _ = r.write_family(store, ctx.clock, ids["fp16"].problem)
    rid = run_reuse(store, ctx.clock, chain_id, ids, LIMITS)
    data = store.get(rid).body["data"]
    assert data["bf16"]["invalid"] == EXPECTED_INVALID["bf16"] == ["mma"]
    assert data["fp16"]["invalid"] == []
    assert data["bf16"]["as_expected"] and data["fp16"]["as_expected"]
    verdicts = {s["key"]: s["verdict"] for s in data["bf16"]["steps"]}
    assert verdicts == {k: ("invalid" if k == "mma" else "keep") for k in verdicts}


def test_case_records_are_reused(ctx: Context, store: Store) -> None:
    a = r.write_case(store, ctx.clock, r.CASE_FP16, "t", "sha256:img")
    b = r.write_case(store, ctx.clock, r.CASE_FP16, "t", "sha256:img")
    assert a.case == b.case and a.baseline == b.baseline
    assert store.get(a.case).refs["baseline"] == a.baseline
    assert a.body.problem == a.problem and len(a.body.shape_dist) == 3
