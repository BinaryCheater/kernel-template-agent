"""WP16: L2 transaction-level simulator, what-if, L3 adapter and the regression scoreboard.

Code-stage acceptance (spec/workplan.md WP16): the GEMM seed family simulated at L2 with a
timeline; L1 equals the steady-state limit of L2 on a loop-dominated fixture; a what-if
shows a non-zero shadow price on a walled fixture; the scoreboard renders on fixture data
marked FIXTURE.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_wp15_effects_hwmodel import POINT, gemm_case, gemm_formulation

from apihelp import TEST_ACTOR, request, seeded_api
from kta.api.app import create_app
from kta.core.context import Context
from kta.core.controls import baseline
from kta.core.errors import NotSupported
from kta.formulation import evaluate_effects
from kta.models import MachineModel, price, simulate, what_if
from kta.models.l1 import EvalEffects, EvalTx
from kta.models.l2 import CATEGORIES, STALL_NCU
from kta.models.l3 import ExternalSassSimulator
from kta.models.regression import Observation, admit, results, spearman, summarise
from kta.models.whatif import record_l2, record_what_if
from kta.store.base import Store
from kta_devseed.knowledge import Seeder
from kta_devseed.models import gb10, gemm_effects, seed_models

REL = 1e-9


def _checked(r, l1_total: float) -> None:  # type: ignore[no-untyped-def]
    """Invariants every feasible L2 result keeps."""
    assert r.feasible and r.total_us >= l1_total * (1 - REL)  # L1 <= L2 (obligation O-a)
    for res, busy in r.busy_us.items():
        idle = r.idle_us.get(res, {})
        assert set(idle) <= set(CATEGORIES)
        assert busy + sum(idle.values()) == pytest.approx(r.total_us, rel=1e-6), res


# ---- L2 on the GEMM seed family -----------------------------------------------------


def test_gemm_seed_family_simulated_at_l2_with_a_timeline(ctx: Context, store: Store) -> None:
    mm = gb10()
    f = gemm_formulation(ctx, store, mm)
    _, case = gemm_case()
    shapes = [d["shape"] for d in case.shape_dist]
    assert len(shapes) >= 3
    for shape in shapes:
        point = {**POINT, "KT": max(1, shape["K"] // POINT["BK"])}
        ev = evaluate_effects(f, point, shape, mm.hw)
        l1 = price(ev, mm)
        r = simulate(ev, mm)
        _checked(r, l1.total_us)
        # L2 does the same work as L1: per-resource busy time equals L1's.
        for res, busy in l1.composed.busy_us.items():
            assert r.busy_us[res] == pytest.approx(busy, rel=1e-6)
        assert r.waves and r.intervals["full" if "full" in r.intervals else "tail"]
        assert r.waves[0].start_us == pytest.approx(r.prefix_us)
        assert r.waves[-1].end_us == pytest.approx(r.total_us - r.interaction_us)
        assert r.ctas == sum(w.ctas for w in r.waves)
        assert r.metrics["gpu__time_duration.sum"] == pytest.approx(r.total_us * 1e3)
        assert "lts__t_bytes.sum.pct_of_peak_sustained_elapsed" in r.metrics
        assert set(r.stalls) <= set(STALL_NCU.values())
        assert r.unmodelled == {"launch": r.ctas}  # reported, not free (D465)
        assert r.model.startswith("l2:1:gb10@1:")
        assert simulate(ev, mm).summary() == r.summary()  # deterministic


def test_the_l2_result_is_stored_as_a_report(ctx: Context, store: Store) -> None:
    mm = gb10()
    r = simulate(gemm_effects({"M": 1024, "N": 1024, "K": 1024}, _c(3)), mm)
    rid = record_l2(store, r, TEST_ACTOR, ctx.clock.now(), {})
    data = store.get(rid).body["data"]
    assert data["total_us"] == r.total_us and data["intervals"]["full"]


def _c(s: int) -> dict[str, int]:
    return {"BM": 128, "BN": 128, "BK": 32, "S": s}


# ---- L1 is the steady-state limit of L2 -------------------------------------------------


def test_l1_equals_the_steady_state_limit_of_l2_on_a_loop_dominated_fixture() -> None:
    mm = gb10()
    gaps = []
    for k in (1024, 8192, 65536):
        # 8 x 6 = 48 CTAs: one wave on 48 SMs, so the kernel is the main loop; four
        # stages (three loads in flight) hide the load and MMA latency (resource-bound).
        ev = gemm_effects({"M": 1024, "N": 768, "K": k}, _c(4))
        l1, l2 = price(ev, mm), simulate(ev, mm)
        _checked(l2, l1.total_us)
        assert len(l2.waves) == 1 and l2.trips == k // 32
        gaps.append((l2.total_us - l1.total_us) / l1.total_us)
    assert gaps[0] > gaps[1] > gaps[2] >= 0
    assert gaps[2] < 0.001  # within 0.1 % at 2048 trips: the prologue/epilogue amortised


def test_with_shallow_pipelines_l2_exceeds_l1_by_the_recurrence() -> None:
    # Known L1 simplification (D464): its recurrence bound counts latencies only, while the
    # load -> sync -> compute -> slot cycle also holds the producers' busy time. L1 stays
    # below L2 (an optimistic MIP), and the gap does not shrink with more trips.
    mm = gb10()
    ev = gemm_effects({"M": 1024, "N": 768, "K": 65536}, _c(3))
    l1, l2 = price(ev, mm), simulate(ev, mm)
    _checked(l2, l1.total_us)
    assert l2.total_us > 1.1 * l1.total_us


def test_without_a_loop_statement_l2_runs_one_trip_and_says_so() -> None:
    mm = gb10()
    ev = gemm_effects({"M": 1024, "N": 768, "K": 1024}, _c(3))
    r = simulate(EvalEffects(ev.txs, ev.storage, ()), mm)
    assert r.trips == 1 and any("one trip per CTA" in n for n in r.notes)


# ---- walls and what-if --------------------------------------------------------------


def _walled() -> EvalEffects:
    # Four stages of 128 x 128 x 32 FP16 tiles: 64 kB of shared memory per CTA, so the
    # smem_per_sm wall allows one CTA per SM and 1024 CTAs run in 22 waves.
    return gemm_effects({"M": 4096, "N": 4096, "K": 4096}, _c(4))


def test_what_if_shows_a_non_zero_shadow_price_on_a_walled_fixture(
    ctx: Context, store: Store
) -> None:
    mm = gb10()
    ev = _walled()
    base = simulate(ev, mm)
    assert base.residency == 1 and base.ctas > 48
    wall = next(w for w in base.walls if w.residency == 1)
    assert sum(base.idle_us[base.bottleneck].values()) > 0
    assert base.idle_us[base.bottleneck]["wall"] > 0  # attributed to the binding wall
    assert any(f"wall {wall.name} binds" in n for n in base.notes)
    w = what_if(ev, mm, walls={wall.name: 2 * wall.usage})
    assert w.moved.residency == 2
    assert w.shadow_price_us is not None and w.shadow_price_us > 0
    assert w.delta_total_us is not None and w.delta_total_us < 0
    rid = record_what_if(store, w, TEST_ACTOR, ctx.clock.now(), {})
    assert store.get(rid).body["data"]["shadow_price_us"] == w.shadow_price_us


def test_what_if_on_a_peak_and_on_an_unknown_name() -> None:
    mm = gb10()
    ev = gemm_effects({"M": 4096, "N": 4096, "K": 4096}, _c(3))
    rate = mm.resources["l2"].rate or 0.0
    w = what_if(ev, mm, peaks={"l2": 2 * rate})
    assert w.delta_total_us is not None and w.delta_total_us < 0
    assert w.moved.model != w.base.model  # the moved model is another version
    with pytest.raises(KeyError):
        what_if(ev, mm, peaks={"no_such_unit": 1.0})
    with pytest.raises(KeyError):
        what_if(ev, mm, walls={"no_such_wall": 1.0})


def test_a_no_launch_wall_is_infeasible_not_a_cost() -> None:
    mm = gb10()
    ev = gemm_effects({"M": 4096, "N": 4096, "K": 4096}, _c(3))
    tight = what_if(ev, mm, walls={"smem_per_cta": 8192})
    assert tight.base.feasible and not tight.moved.feasible
    assert tight.shadow_price_us is None and "no launch" in tight.moved.notes[0]


def test_unpriced_operation_classes_are_carried_into_l2() -> None:
    mm = gb10()
    ev = gemm_effects({"M": 1024, "N": 768, "K": 1024}, _c(3))
    extra = EvalTx("x", "odd", "other", "no_such_class", 5.0)
    r = simulate(EvalEffects((*ev.txs, extra), ev.storage, ev.overlaps), mm)
    assert r.unpriced == {"no_such_class": 5.0}


# ---- L3 adapter ---------------------------------------------------------------------


def test_l3_adapter_defines_an_executor_job_and_is_not_supported_until_run() -> None:
    sim = ExternalSassSimulator("/opt/bundle", "/opt/upstream")
    job = sim.job("ab" * 32, {"grid_x": 32, "block_x": 128})
    assert job.kind == "shell" and job.pool == "cpu" and job.inputs == {"kernel.cubin": "ab" * 32}
    assert "/opt/bundle/tools/simulate.py" in job.argv
    with pytest.raises(NotSupported, match="pending real run"):
        sim.simulate("ab" * 32, {"grid_x": 32})


# ---- regression and scoreboard ----------------------------------------------------------


def test_spearman_ties_and_degenerate_inputs() -> None:
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(0.8660254, rel=1e-6)
    assert spearman([1], [1]) is None and spearman([1, 1], [1, 2]) is None


def test_scoring_errors_per_resource_and_admission() -> None:
    def o(k: str, p: float, m: float, hold: bool = False) -> Observation:
        return Observation(k, {}, p, m, {"l2": p / 2}, {"l2": m / 2}, hold)

    res = results([o("a", 1, 1.1), o("a", 2, 2.2), o("b", 1, 1.0, True), o("b", 3, 1.5, True)])
    assert [r.kernel for r in res] == ["a", "b"]
    a, b = res
    assert a.rank_correlation == pytest.approx(1.0) and not a.holdout and b.holdout
    assert a.error_by_resource["total"] == pytest.approx(0.1 / 1.1)
    assert a.error_by_resource["l2"] == pytest.approx(0.1 / 1.1)
    s = summarise(res)
    assert s["holdout_rank_correlation_mean"] == pytest.approx(1.0)
    knob = baseline().version_admission
    assert knob == {"rank_min": 0.7, "rel_error_max": 0.25}
    assert s["holdout_error_median:total"] == pytest.approx(0.5)  # median of 0 and 1
    assert admit(s, knob) is False
    assert admit(summarise(res[:1]), knob) is None  # no hold-out case scored


def test_scoreboard_renders_on_fixture_data_marked_fixture(tmp_path: Path) -> None:
    api = seeded_api(tmp_path)
    ids = seed_models(Seeder(api.store, api.ctx.clock))  # type: ignore[arg-type]
    app = create_app(api)
    status, sb = request(app, "GET", "/model/scoreboard?target=gb10")
    assert status == 200 and sb["fixture"] is True
    assert [v["id"] for v in sb["versions"]] == ids
    assert {v["origin"] for v in sb["versions"]} == {"fixture"}
    first = sb["versions"][0]
    assert first["change"] == "initial" and sb["versions"][1]["previous"] == ids[0]
    assert {r["kernel"] for r in first["results"]} == {
        "gemm_fp16 4096^3",
        "gemm_fp16 128x128x16384",
    }
    assert "total" in sb["error_keys"] and "l2" in sb["error_keys"]
    assert all(r["rank_correlation"] is not None for r in first["results"])
    status, tl = request(app, "GET", f"/model/versions/{ids[0]}/timeline?kernel=gemm_fp16 4096^3")
    assert status == 200 and tl["origin"] == "fixture" and len(tl["configurations"]) == 4
    c0 = tl["configurations"][0]
    assert c0["waves"] and c0["intervals"]["full"] and c0["measured_us"] > 0
    status, _ = request(app, "GET", f"/model/versions/{ids[0]}/timeline?kernel=nope")
    assert status == 404


def test_l2_is_fast_enough_for_pre_compile_analysis() -> None:
    # 02 §4 speed budget "ms"; checked on work done, not on wall time (no clock in tests):
    # the schedule books one entry per phase and trip, merged into few intervals.
    mm = gb10()
    r = simulate(gemm_effects({"M": 4096, "N": 4096, "K": 4096}, _c(3)), mm)
    assert sum(len(v) for v in r.intervals.values()) < 4 * r.trips * len(r.busy_us)
    assert isinstance(mm, MachineModel)
