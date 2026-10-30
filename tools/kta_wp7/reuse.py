"""reuse_plan of the GEMM chain on the BF16 case (and on the FP16 case as the control).

The Problem's numeric params (operand bits, exponent bits) are passed explicitly:
``knowledge.scope.problem_params`` looks the Problem up by name, while the Case refers to
it by record ID (see deliveries/WP7.md, needed changes). Γ is evaluated for one
assignment of the family (the expression bindings of p.* need one, decision D286): the
best measured candidate of the latest top-k run, else ``REPRESENTATIVE``. Expected: only
the ``mma`` step (compute_instruction_selection@2, Γ: the instruction's operand type) is
invalid for BF16, and every step is kept for FP16.
"""

from __future__ import annotations

from typing import Any

from kta.core.context import Clock
from kta.core.schemas import Record
from kta.core.schemas.evaluation import Problem
from kta.core.schemas.process import Report
from kta.knowledge import reuse_plan
from kta.store.base import Store
from kta_wp7.records import ACTOR, CaseIds

EXPECTED_INVALID = {"bf16": ["mma"], "fp16": []}
REPRESENTATIVE = {
    "BM": 128,
    "BN": 128,
    "BK": 64,
    "S": 3,
    "P": 2,
    "threads": 128,
    "split": 1,
    "KT": 64,
}


def best_measured(store: Store) -> dict[str, Any] | None:
    """Assignment of the fastest measured candidate of the first bucket of the latest
    top-k Report, or None before any top-k run."""
    found = store.query("Report", {"subject": "wp7_topk"}, order="-created_at", limit=1)
    if not found:
        return None
    rows = [x for x in found[0].body["data"]["buckets"][0]["top"] if "median_us" in x]
    return dict(min(rows, key=lambda x: x["median_us"])["assignment"]) if rows else None


def numeric_params(problem: Problem) -> dict[str, Any]:
    return {k: v for k, v in problem.params.items() if isinstance(v, (int, float, bool))}


def run_reuse(
    store: Store,
    clock: Clock,
    chain_id: str,
    cases: dict[str, CaseIds],
    hw_limits: dict[str, int | float],
    assignment: dict[str, Any] | None = None,
) -> str:
    used = dict(assignment or REPRESENTATIVE)
    data: dict[str, Any] = {"chain": chain_id, "assignment": used}
    for label, ids in cases.items():
        params = numeric_params(ids.problem_body)
        plan = reuse_plan(store, chain_id, ids.body, hw_limits, params, used)
        steps = [
            {
                "key": s.key,
                "template": s.template,
                "verdict": s.verdict,
                "reason": s.reason,
                "alternatives": list(s.alternatives),
            }
            for s in plan.steps
        ]
        invalid = [s["key"] for s in steps if s["verdict"] == "invalid"]
        data[label] = {
            "case": ids.case,
            "steps": steps,
            "invalid": invalid,
            "expected_invalid": EXPECTED_INVALID[label],
            "as_expected": invalid == EXPECTED_INVALID[label]
            and all(s["verdict"] == "keep" for s in steps if s["key"] not in invalid),
        }
    summary = "; ".join(
        f"{k}: invalid {v['invalid']} (expected {v['expected_invalid']})"
        for k, v in data.items()
        if k in cases
    )
    body = Report(title="wp7 reuse plan", subject="wp7_reuse", summary=summary, data=data)
    refs: dict[str, str | list[str]] = {
        "chain": chain_id,
        "cases": [ids.case for ids in cases.values()],
    }
    return store.put(Record.new("Report", body, ACTOR, clock.now(), refs=refs))
