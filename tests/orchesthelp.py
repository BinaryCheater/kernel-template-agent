"""Builders for orchestrator tests: facts for every branch of the pure step functions, and
a small real world (Problem, Contract, Case, HwModel) in a test store. Plain helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from kta.core.context import Context
from kta.core.schemas import Actor, Record
from kta.orchestrator.pipeline import Services
from kta.orchestrator.transition import Loop
from kta.store import controls as controls_store
from kta.store.base import Store
from solvehelp import TEST_HW

ID = "a" * 64  # a syntactically valid record ID for pure-step facts
HUMAN = Actor(type="human", name="test")


def ctx_with(ctx: Context, changes: dict[str, Any]) -> Context:
    """``ctx`` with an unstored controls proposal (knob id -> value), for pure step tests."""
    return replace(ctx, controls=ctx.controls.with_changes(changes))


def set_controls(store: Store, ops: Any, changes: dict[str, Any], reason: str = "test") -> str:
    """Write a controls version through the real path; returns its record ID."""
    now = store.clock.now()  # type: ignore[attr-defined]
    expected = controls_store.ensure(store, ops, now).version_id
    snap = controls_store.propose(store, ops, changes, HUMAN, reason, now, expected)
    assert snap.version_id is not None
    return snap.version_id


def loop(kind: str, state: str, facts: dict[str, Any] | None = None, **data: Any) -> Loop:
    return Loop("L1", kind, ID, state, {"round": 1, **data}, facts or {})


def bucket(**over: Any) -> dict[str, Any]:
    """A judged bucket where prediction ranks match, error is small, gap is explained."""
    b: dict[str, Any] = {
        "name": "s0",
        "failed_checks": [],
        "pairs": [(10.0, 11.0), (20.0, 21.0), (30.0, 31.0)],
        "evidence": [ID, ID],
        "best_us": 11.0,
        "predicted_us": 10.0,
        "bound_us": 5.0,
        "reference_us": 10.5,
        "units": [{"unit": "tensor", "bound_us": 5.0, "predicted_us": 10.0, "measured_us": None}],
        "assignment": {"bm": 64},
        "measurement": ID,
        "check": ID,
        "bound": ID,
        "ci_us": 0.1,
        "solver": {"method": "enumerate", "model_optimal": True, "exhaustive": True},
    }
    b.update(over)
    return b


def space_loss(**over: Any) -> dict[str, Any]:
    """A bucket whose model is accurate and solver certified, but whose representation
    loses about half the time to SOL (gap class l_rep)."""
    pairs = [(10.8, 11.0), (20.0, 21.0), (30.0, 31.0)]
    return bucket(**{"pairs": pairs, "predicted_us": 10.8, "reference_us": 8.0, **over})


def judging(*buckets: dict[str, Any], references: list[str] | None = None, **data: Any) -> Loop:
    facts = {
        "buckets": list(buckets),
        "references": references or [],
        "family": ID,
        "model": ID,
        "significance_rel": 0.012,
    }
    return loop("solve", "judging", facts, **data)


def world(svc: Services, target: str = "test-gpu", hw: bool = True) -> dict[str, str]:
    """Problem, Contract and Case records (and a HwModel for ``target`` when ``hw``)."""
    s: Store = svc.store
    now = svc.ctx.clock.now()
    problem = s.put(
        Record.new(
            "Problem",
            {
                "name": "gemm_fp16_test",
                "reference_impl": "torch.matmul(A, B)",
                "params": {
                    "min_work": {
                        "tensor": "2 * case.M * case.N * case.K",
                        "dram": "2 * (case.M * case.K + case.K * case.N)",
                    }
                },
            },
            HUMAN,
            now,
        )
    )
    contract = s.put(Record.new("Contract", {"timing_boundary": "kernel"}, HUMAN, now))
    case = s.put(
        Record.new(
            "Case",
            {
                "problem": problem,
                "shape_dist": [{"shape": {"M": 256, "N": 256, "K": 256}, "weight": 1.0}],
                "objective": "weighted_latency",
                "target": target,
            },
            HUMAN,
            now,
            refs={"contract": contract},
        )
    )
    out = {"problem": problem, "contract": contract, "case": case}
    if hw:
        out["hw"] = s.put(
            Record.new("HwModel", TEST_HW.model_copy(update={"target": target}), HUMAN, now)
        )
    return out
