"""kta.models: bound, per-unit prediction, occupancy feedback, fitting and versions."""

from __future__ import annotations

import pytest

from kta.core.context import Context
from kta.core.errors import NotSupported
from kta.core.schemas import Actor, Record, RefValue
from kta.core.schemas.evaluation import Measurement, Prediction, Problem
from kta.models import NoResidentCta, UnitModel, bound, calibrate, load_model, min_work, save_model
from kta.models.unit_model import Interaction, occupancy
from kta.store.base import Store
from solvehelp import TEST_HW, case
from storehelp import T0

ACTOR = Actor(type="tool", name="test")


def test_bound_is_the_largest_unit_time() -> None:
    b = bound(case(), {"M": 1}, TEST_HW, {"dram": 2.5e5, "tensor": 2.0e8, "fma": 0.0})
    assert b.per_unit_us == pytest.approx({"dram": 1.0, "tensor": 1.0})
    assert b.bound_us == pytest.approx(1.0)
    with pytest.raises(NotSupported, match="no peak"):
        bound(case(), {}, TEST_HW, {"l2": 1.0})


def test_min_work_from_problem_params() -> None:
    problem = Problem(
        name="gemm",
        reference_impl="torch.matmul",
        params={"min_work": {"tensor": "2 * case.M * case.N * case.K", "dram": "2 * case.M"}},
    )
    assert min_work(problem, {"M": 2, "N": 3, "K": 4}) == {"tensor": 48.0, "dram": 4.0}
    with pytest.raises(NotSupported):
        min_work(Problem(name="x", reference_impl="y"), {})


def test_predict_is_max_plus_serial_plus_interactions() -> None:
    m = UnitModel(
        target="t",
        eff={"dram": 0.5},
        serial_coef=2.0,
        serial_const_us=1.0,
        interactions=(Interaction(("dram", "fma"), 0.5, "test"),),
    )
    p = m.predict({"dram": 2.5e5, "fma": 1.0e7}, {"issue": 4.0e5}, TEST_HW)
    # peak times: dram 1 us, fma 1 us, serial issue 1 us; dram at eff 0.5 takes 2 us
    assert p.per_unit == pytest.approx({"dram": 1.0, "fma": 1.0})
    assert p.serial == pytest.approx(1.0)
    assert p.total == pytest.approx(2.0 + 2.0 * 1.0 + 1.0 + 0.5 * 1.0)
    assert p.model == m.version


def test_version_changes_with_parameters_and_round_trips() -> None:
    a = UnitModel(target="t")
    b = UnitModel(target="t", eff={"dram": 0.8})
    assert a.version != b.version
    assert UnitModel.from_dict(b.to_dict()).version == b.version


def test_occupancy_feedback_slows_low_occupancy_candidates() -> None:
    m = UnitModel(target="t")
    full = {"regs": 32, "smem": 8192, "threads": 256}
    starved = {"regs": 255, "smem": 8192, "threads": 256}  # one CTA of 256 threads per SM
    assert occupancy(full, TEST_HW) == pytest.approx(1536 / 1536)
    assert occupancy(starved, TEST_HW) == pytest.approx(256 / 1536)
    p = m.predict({"dram": 2.5e5}, {}, TEST_HW)
    slow = m.repredict(p, TEST_HW, starved)
    assert slow.total == pytest.approx(p.total / ((256 / 1536) / 0.25))
    assert m.repredict(p, TEST_HW, full).total == pytest.approx(p.total)
    with pytest.raises(NoResidentCta):
        m.repredict(p, TEST_HW, {"smem": 200_000, "threads": 128})


def _synthetic(truth: UnitModel, n: int) -> list[tuple[Prediction, Measurement]]:
    """Points whose 'measurement' is the truth model's prediction (fit test data)."""
    out: list[tuple[Prediction, Measurement]] = []
    for i in range(n):
        work = {"dram": 1e6 * (1 + i % 7), "fma": 4e7 * (1 + (i * 3) % 5)}
        p = UnitModel(target="t").predict(work, {"issue": 4e4 * (1 + i % 3)}, TEST_HW)
        y = truth.total_us(p.per_unit, p.serial)
        out.append((p, Measurement(median_us=y, ci_low_us=y, ci_high_us=y, n=5, protocol="x")))
    return out


def test_fit_recovers_efficiency_and_reports_holdout() -> None:
    truth = UnitModel(target="t", eff={"dram": 0.6, "fma": 0.8}, serial_coef=1.5)
    start = UnitModel(target="t")
    fitted = start.fit(_synthetic(truth, 30))
    assert fitted.parent == start.version and fitted.version != start.version
    assert fitted.eff["dram"] == pytest.approx(0.6, rel=0.05)
    assert fitted.serial_coef == pytest.approx(1.5, rel=0.1)
    holdout = fitted.fit_info["holdout"]
    assert holdout["n"] == 6 and holdout["median_abs_rel_err"] < 0.02
    assert not fitted.interactions  # no interaction in the data, none added
    assert start.eff == {}  # the old version is unchanged


def test_fit_adds_an_interaction_only_when_the_data_support_it() -> None:
    truth = UnitModel(
        target="t",
        eff={"dram": 0.7, "fma": 0.9},
        interactions=(Interaction(("dram", "fma"), 0.6, "truth"),),
    )
    fitted = UnitModel(target="t").fit(_synthetic(truth, 40))
    assert [i.units for i in fitted.interactions] == [("dram", "fma")]
    assert "F=" in fitted.interactions[0].basis
    assert fitted.fit_info["holdout"]["median_abs_rel_err"] < 0.05


def _job_refs(ctx: Context, store: Store) -> dict[str, RefValue]:
    env = store.put(
        Record.new(
            "EnvManifest",
            {
                "role": "worker",
                "host": "h",
                "os": "linux",
                "arch": "aarch64",
                "python": "3.12",
                "uv": "0",
                "uv_lock_sha256": "0" * 64,
                "commit": "0" * 40,
                "dirty": False,
            },
            ACTOR,
            T0,
        )
    )
    worker = store.put(
        Record.new(
            "Worker",
            {"name": "w", "host": "h", "transport": "local", "commit": "0" * 40},
            ACTOR,
            T0,
        )
    )
    return {"env": env, "worker": worker}


def test_calibrate_writes_a_new_version_with_its_data(ctx: Context, store: Store) -> None:
    truth = UnitModel(target="t", eff={"dram": 0.5})
    refs = _job_refs(ctx, store)
    pairs: list[tuple[str, str]] = []
    for p, m in _synthetic(truth, 10):
        pid = store.put(Record.new("Prediction", p, ACTOR, T0, origin="fixture"))
        mid = store.put(Record.new("Measurement", m, ACTOR, T0, refs=refs, origin="fixture"))
        pairs.append((pid, mid))
    first = save_model(ctx, store, UnitModel(target="t"), ACTOR)
    new, new_id = calibrate(ctx, store, first, pairs, ACTOR)
    rec = store.get(new_id)
    assert rec.refs["previous"] == first and rec.refs["predictions"] == [p for p, _ in pairs]
    assert rec.body["data"]["fitted_on"] == [list(p) for p in pairs]
    assert load_model(store, new_id).version == new.version
    assert load_model(store, first).version == UnitModel(target="t").version
