"""kta.formulation and kta.solve beyond the acceptance tests."""

from __future__ import annotations

import itertools
import math
import random

import pytest

from kta.core.context import Context
from kta.core.errors import NotSupported, SchemaError
from kta.core.schemas import Actor
from kta.core.schemas.evaluation import Build
from kta.formulation import assemble, cost_breakdown, per_unit_work, violated
from kta.models import UnitModel
from kta.models.unit_model import Interaction
from kta.solve import enumerate_space, record_solve, rerank, shape_buckets, solve, space_size
from kta.store.base import Store
from solvehelp import TEST_ARCH, TEST_HW, TEST_HW2, TEST_MM, case, gemm, linear_family

SHAPE = {"M": 512, "N": 512, "K": 512}
MODEL = UnitModel(target=TEST_HW.target)


def test_unbound_pattern_parameter_is_reported(ctx: Context, store: Store) -> None:
    fp, templates, _ = gemm(ctx, store)
    item = next(i for i in fp.chain.items if i.key == "ab_stages")
    del item.bindings["p.iterations"]
    with pytest.raises(SchemaError, match=r"ab_stages.*p\.iterations is used but not bound"):
        assemble(fp, fp.chain, templates, case(), TEST_HW.limits)
    item.bindings["p.iterations"] = "KT +"
    with pytest.raises(SchemaError, match="not a valid expression"):
        assemble(fp, fp.chain, templates, case(), TEST_HW.limits)


def test_missing_hw_names_are_listed(ctx: Context, store: Store) -> None:
    fp, templates, _ = gemm(ctx, store)
    f = assemble(fp, fp.chain, templates, case(), {})
    assert "smem_per_cta" in f.missing_hw and "smem_queue_depth" in f.missing_hw


def test_work_sums_per_unit_and_diagnostics_name_sources(ctx: Context, store: Store) -> None:
    _, _, f = gemm(ctx, store)
    point = {"BM": 128, "BN": 128, "BK": 32, "S": 3, "P": 0, "KT": 16, "mech": "instruction"}
    point |= {"gran": 16, "swz": "identity", "threads": 128}
    over, serial = per_unit_work(f, point, SHAPE, TEST_MM)
    tiles, tile_bytes = 16, (128 + 128) * 32 * 2  # (512 / 128)^2 tiles; one A/B k-tile pair
    # cp.async moves every k-tile pair from L2 (the operand traffic is declared once, D460)
    assert over["l2"] == tile_bytes * 16 * tiles
    accesses = 2 * 16 * tiles * tile_bytes // 512
    barriers = 16 * 2 * tiles  # leading + trailing barrier per k-tile, one instruction per warp
    assert over["issue"] == over["l2"] / 512 + barriers * 4 + accesses
    assert over["smem"] == accesses * 4 * 2  # identity map: 4 wavefronts, twice
    assert serial == {}  # P = 0 serialises transfer and contract in L1 (a pair bound)
    with pytest.raises(NotSupported, match="hardware model"):
        per_unit_work(f, point, SHAPE, TEST_HW.limits)
    rows = cost_breakdown(f, point, SHAPE, TEST_MM)
    assert {r["source"] for r in rows} == {"ab_async", "ab_stages", "ab_swizzle"}
    bad = {**point, "P": 3, "KT": 15}
    labels = {c.label for c in violated(f, bad, SHAPE, TEST_HW.limits)}
    assert labels == {"ab_stages:c_lifetime", "k_split:c_cover", "ab_swizzle:c_min_wavefronts"}


def test_effects_without_architecture_class_are_refused(ctx: Context, store: Store) -> None:
    _, _, f = gemm(ctx, store)
    with pytest.raises(NotSupported, match="architecture class"):
        solve(f, case(SHAPE), SHAPE, MODEL, TEST_HW)


def test_gemm_solve_is_exhaustive_and_accounts_for_every_point(ctx: Context, store: Store) -> None:
    _, _, f = gemm(ctx, store)
    r = solve(
        f,
        case(SHAPE),
        {"name": "s0", "shape": SHAPE},
        MODEL,
        TEST_HW2,
        arch=TEST_ARCH,
        shortlist=16,
    )
    assert r.method == "enumerate" and r.exhaustive and r.model_optimal
    assert r.feasible + r.pruned == r.space_size == space_size(f)
    # identity bank mapping is excluded by the performance assumption of step 5
    assert r.pruned_by["ab_swizzle:c_min_wavefronts"] == r.space_size // 2
    totals = [c.prediction.total for c in r.candidates]
    assert totals == sorted(totals) and len(totals) == 16
    for c in r.candidates:
        a = c.assignment
        assert a["KT"] == math.ceil(512 / a["BK"]) and a["swz"] == "xor_swizzle"
        assert violated(f, a, SHAPE, TEST_HW.limits) == []
        assert c.bucket == "s0" and c.prediction.model == MODEL.version
        assert c.prediction.total > 0 and c.prediction.unpriced == {}
    assert r.scope["model"] == MODEL.version and r.scope["family"] == "gemm_seed5_test@1"


def test_interval_pruning_agrees_with_brute_force() -> None:
    fp, templates = linear_family(hi=100)
    f = assemble(fp, fp.chain, templates, case(), {})
    shape = {"cap": 150, "base": 200000}
    seen: list[dict[str, int]] = []
    stats = enumerate_space(f, shape, {}, seen.append)
    brute = [
        {"x": x, "y": y}
        for x, y in itertools.product(range(1, 101), range(1, 101))
        if not violated(f, {"x": x, "y": y}, shape, {})
    ]
    assert sorted(map(str, seen)) == sorted(map(str, brute))
    assert stats.feasible + stats.pruned == 100 * 100


def test_budgeted_sampling_is_not_exhaustive(ctx: Context, store: Store) -> None:
    _, _, f = gemm(ctx, store)
    r = solve(
        f, case(SHAPE), SHAPE, MODEL, TEST_HW2, arch=TEST_ARCH, rng=random.Random(1), max_nodes=300
    )
    assert not r.exhaustive and not r.model_optimal
    assert any("node budget" in n for n in r.notes)


def test_large_linear_space_goes_to_mip() -> None:
    fp, templates = linear_family(hi=1000)
    f = assemble(fp, fp.chain, templates, case(), {})
    shape = {"cap": 1500, "base": 2000000}
    r = solve(f, case(), shape, MODEL, TEST_HW, shortlist=4)
    assert r.method == "mip" and r.mip_status == "Optimal" and r.model_optimal
    assert len(r.candidates) == 4 and len({c.key for c in r.candidates}) == 4
    with_pair = UnitModel(
        target=TEST_HW.target, interactions=(Interaction(("dram", "fma"), 0.1, "t"),)
    )
    r2 = solve(f, case(), shape, with_pair, TEST_HW, shortlist=2)
    assert r2.method == "mip" and not r2.model_optimal  # the MIP objective lacks the pair term


def test_menus_too_large_for_mip_fall_back_to_enumeration() -> None:
    fp, templates = linear_family(hi=5000)
    f = assemble(fp, fp.chain, templates, case(), {})
    r = solve(f, case(), {"cap": 150, "base": 10**7}, MODEL, TEST_HW, shortlist=2)
    assert r.method == "enumerate" and r.exhaustive
    assert any("MIP not used" in n for n in r.notes)


def test_rerank_uses_compile_feedback_and_drops_failed_builds(ctx: Context, store: Store) -> None:
    _, _, f = gemm(ctx, store)
    r = solve(f, case(SHAPE), SHAPE, MODEL, TEST_HW2, arch=TEST_ARCH, shortlist=4)
    a, b, c, d = r.candidates
    builds = {
        a.key: Build(ok=True, regs=255, spills=0, smem=60000, log_summary="ok"),
        b.key: Build(ok=True, regs=32, spills=0, smem=8192, log_summary="ok"),
        c.key: Build(ok=False, log_summary="ptxas error"),
    }
    out = rerank(r.candidates, builds, MODEL, TEST_MM, k=8)
    assert [x.key for x in out] == [b.key, a.key]  # a is starved of occupancy, c failed, d unbuilt
    assert out[1].prediction.total > a.prediction.total
    assert d.key not in {x.key for x in out}


def test_record_solve_writes_candidates_predictions_and_report(ctx: Context, store: Store) -> None:
    _, _, f = gemm(ctx, store)
    r = solve(f, case(SHAPE), SHAPE, MODEL, TEST_HW2, arch=TEST_ARCH, shortlist=3)
    recs = record_solve(ctx, store, r, Actor(type="tool", name="kta solve"))
    assert len(recs.candidates) == len(recs.predictions) == 3
    pred = store.get(recs.predictions[0])
    assert pred.refs["candidate"] == recs.candidates[0]
    assert pred.body["total"] == pytest.approx(r.candidates[0].prediction.total)
    report = store.get(recs.report)
    assert report.body["subject"] == "solve"
    assert report.body["data"]["model_optimal"] is True
    assert report.body["data"]["scope"]["formulation"] == f.fingerprint


def test_shape_buckets_one_per_shape() -> None:
    buckets = shape_buckets(case({"M": 1}, {"M": 2}))
    assert [b["name"] for b in buckets] == ["s0", "s1"]
    assert buckets[1]["shape"] == {"M": 2}
