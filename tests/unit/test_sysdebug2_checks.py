"""SYSDEBUG-2: tiered, library-relative checks, shape scope, step work (D694–D700)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from kta.api.cases import _step_work  # pyright: ignore[reportPrivateUsage]
from kta.checks.full import in_scope
from kta.checks.numeric import library_verdict
from kta.core.context import Context
from kta.core.errors import SchemaError
from kta.core.ids import canonical_json
from kta.core.schemas.evaluation import Contract
from kta.orchestrator import runner as runner_mod
from kta.orchestrator.pipeline import Services
from kta.orchestrator.runner import Runner
from kta.store.base import SpanRow
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps
from orchesthelp import world

TOL = {"rtol": 0.01, "atol": 0.01, "relative_to": "library", "loose": 4.0, "strict": 1.5}


def _stats(rel_l2: float, max_abs: float, nonfinite: int = 0) -> dict[str, float | int | None]:
    return {
        "rel_l2": rel_l2,
        "max_abs_err": max_abs,
        "mismatches": 480,
        "nonfinite_where_ref_finite": nonfinite,
    }


def test_fp16_rounding_like_the_library_passes_where_fixed_tolerance_failed() -> None:
    # the rehearsal's numbers: 480 element-wise misses, rel L2 2.1e-4, max error 8.06; the
    # library itself makes about the same error on the same inputs
    cand, lib = _stats(2.1e-4, 8.06), _stats(1.9e-4, 7.5)
    assert library_verdict(cand, lib, TOL, "loose")[0]
    ok, detail = library_verdict(cand, lib, TOL, "strict")
    assert ok and detail["factor"] == 1.5 and detail["library_max_abs_err"] == 7.5


def test_a_real_bug_fails_and_strict_is_tighter_than_loose() -> None:
    lib = _stats(1.9e-4, 7.5)
    assert not library_verdict(_stats(0.3, 900.0), lib, TOL, "loose")[0]  # wrong results
    two_x = _stats(3.8e-4, 15.0)  # twice the library's error
    assert library_verdict(two_x, lib, TOL, "loose")[0]
    assert not library_verdict(two_x, lib, TOL, "strict")[0]
    assert not library_verdict(_stats(1e-5, 0.1, nonfinite=3), lib, TOL, "loose")[0]


def test_an_exact_library_does_not_make_every_rounding_a_failure() -> None:
    exact = _stats(0.0, 0.0)
    assert library_verdict(_stats(5e-7, 5e-4), exact, TOL, "strict")[0]  # within the floors
    assert not library_verdict(_stats(1e-3, 0.5), exact, TOL, "strict")[0]


def test_shape_scope_static_range_and_undeclared() -> None:
    own = [{"M": 4096, "N": 4096, "K": 4096}, {"M": 8192, "N": 1024, "K": 4096}]
    unit = {"M": 1, "N": 4096, "K": 4096}
    assert in_scope(own[1], {"kind": "static"}, own)
    assert not in_scope(unit, {"kind": "static"}, own)
    rng = {"kind": "range", "ranges": {"M": [1, 8192], "N": [1024, 8192]}}
    assert in_scope(unit, rng, own) and not in_scope({"M": 4, "N": 512, "K": 64}, rng, own)
    assert in_scope(unit, {}, own)  # undeclared: everything is checked, as before


def test_an_undeclared_scope_keeps_old_contracts_byte_identical() -> None:
    old = {"tolerances": {}, "input_dists": ["normal"], "edge_shapes": [], "timing_boundary": "t"}
    assert canonical_json(Contract.model_validate(old).model_dump(mode="json")) == canonical_json(
        old
    )
    scoped = Contract.model_validate({**old, "shape_scope": {"kind": "static"}})
    assert scoped.model_dump(mode="json")["shape_scope"] == {"kind": "static"}


def test_step_work_counts_jobs_and_their_largest_overlap() -> None:
    t0 = datetime(2026, 11, 2, tzinfo=UTC)

    def job(name: str, start: int, end: int, status: str = "ok") -> SpanRow:
        return SpanRow(
            name, None, "loop", "job_run", "worker", name, t0 + timedelta(seconds=start),
            t0 + timedelta(seconds=end), status, {},
        )  # fmt: skip

    serial = [job(f"compile {i}", 5 * i, 5 * i + 5) for i in range(4)]
    side = [job("check a", 0, 10), job("check b", 2, 9, "fail"), job("check c", 3, 4)]
    work = {w["kind"]: w for w in _step_work(serial + side, t0, t0 + timedelta(seconds=60))}
    assert work["compile"]["count"] == 4 and work["compile"]["max_parallel"] == 1
    assert work["compile"]["busy_s"] == 20.0
    assert work["check"]["max_parallel"] == 3 and work["check"]["failed"] == 1


def test_a_failing_transition_blocks_the_loop_and_logs_it(
    ctx: Context, store: SqliteStore, ops: SqliteOps, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D700/D701: an error while deciding or applying the transition (e.g. creating the
    routed task) blocks the loop and lands in its event log, instead of escaping and
    repeating the state's work on every step."""
    svc = Services(ctx, store, ops)
    w = world(svc)
    r = Runner(svc, "test")
    loop_id = r.start("solve", w["case"], "c")

    def boom(*_: Any, **__: Any) -> Any:
        raise SchemaError("T4 measurements need distinct candidate identities")

    monkeypatch.setattr(runner_mod, "step", boom)
    assert r.advance(loop_id) is None
    blocked = ops.get_loop(loop_id)["data"]["blocked"]
    assert blocked["error"] == "SchemaError" and "distinct candidate" in blocked["message"]
    logged = [
        e.change["blocked"] for e in ops.events_for("loops", loop_id) if e.change.get("blocked")
    ]
    assert logged and logged[-1]["error"] == "SchemaError"
