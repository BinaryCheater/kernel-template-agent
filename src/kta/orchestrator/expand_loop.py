"""Expand loop: pure transition function (docs/design/10 §2.2, machines.yaml ``expand``).

Facts each state's work reports (``kta.orchestrator.expand_work``):

- frozen: ``source``, ``problem``, ``revision``, ``analysis_only`` (reason, or None)
- measured: ``reference_point`` (Measurement ID; None when analysis-only)
- analyzing, gap: ``tasks`` {task ID: lifecycle state}; when the awaited T2 is accepted,
  ``reconstruction``, ``submission`` (the T2 Submission), ``gaps`` [{"id", "missing"}] and
  ``to_extract`` (Template refs the chain names that the pool lacks)
- reconstructed: ``analysis_only`` and ``coverage`` (Report ID) for an analysis-only
  reference; otherwise ``error_rel`` (reproduction error), ``family``, ``measurement``
- reproduced, ablated: ``report`` (Report ID of the reproduction / ablation)

Tasks follow the chain, not the Gaps (D382): one T3 per Template the chain names that the
pool lacks. Gaps are recorded for every unmatched transaction and stay visible in the
coverage report; those no chain item names a Template for stay open (D384).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kta.core.context import Context
from kta.orchestrator.transition import Draft, Loop, Settings, TaskRequest, Transition, hold

LINEAR = {"frozen": "measured", "reproduced": "ablated", "ablated": "covered"}
# A cancelled extraction task is your decision to leave its Template unextracted (D384);
# while analyzing, or for the T1 after a missed reproduction, it stops the loop.
STUCK = frozenset({"escalated", "suspended"})
SETTLED = frozenset({"accepted", "cancelled"})


def _t2(loop: Loop, previous: str | None = None) -> TaskRequest:
    d = {**loop.data, **loop.facts}
    inputs = [loop.subject] + [str(d[k]) for k in ("source", "problem") if d.get(k)]
    if previous:
        return TaskRequest(
            "T2", "Re-match the chain of this reference against the pool", [*inputs, previous]
        )
    return TaskRequest("T2", "Decompose this reference into a composition chain", inputs)


def _awaited(loop: Loop) -> list[str]:
    states: Mapping[str, str] = loop.facts.get("tasks", {})
    return [states.get(str(t), "") for t in loop.data.get("waiting_tasks", [])]


def step(loop: Loop, ctx: Context) -> Transition:
    s = Settings.of(ctx.controls)
    f = loop.facts
    if loop.state in LINEAR:
        return Transition(loop.state, LINEAR[loop.state], "normal", data=dict(f))
    if loop.state == "measured":
        data = {"reference_point": f["reference_point"]}
        return Transition("measured", "analyzing", "creates_task", data=data, tasks=(_t2(loop),))
    if loop.state in ("analyzing", "gap"):
        got = _awaited(loop)
        strict = loop.state == "analyzing" or loop.data.get("resume_at") == "reconstructed"
        stuck = [g for g in got if g in STUCK or (strict and g == "cancelled")]
        if stuck:
            return hold(loop, f"awaited task is {stuck[0]}", escalated=True)
        if not got or any(g not in SETTLED for g in got):
            return hold(loop, "waiting for tasks")
        return _analyzed(loop) if loop.state == "analyzing" else _filled(loop, got, s)
    if loop.state == "reconstructed":
        if f.get("analysis_only"):
            data = {"coverage": f["coverage"]}
            return Transition("reconstructed", "covered", "normal", data=data, note="analysis only")
        return _reproduction(loop, s)
    return hold(loop, f"{loop.state} has no outgoing transition")


def _analyzed(loop: Loop) -> Transition:
    f = loop.facts
    recon = str(f["reconstruction"])
    data: dict[str, Any] = {"reconstruction": recon, "submission": f.get("submission")}
    to_extract = [str(t) for t in f.get("to_extract", [])]
    if not to_extract:
        return Transition("analyzing", "reconstructed", "normal", data=data)
    inputs = [recon, loop.subject] + [
        str(x) for x in (loop.data.get("source"), f.get("submission")) if x
    ]
    tasks = tuple(
        TaskRequest(
            "T3", f"Extract Template {ref} named by the chain of this Reconstruction", inputs
        )
        for ref in to_extract
    )
    data["resume_at"] = "analyzing"
    return Transition("analyzing", "gap", "creates_task", data=data, tasks=tasks)


def _filled(loop: Loop, got: list[str], s: Settings) -> Transition:
    if loop.data.get("resume_at") == "reconstructed":
        return Transition("gap", "reconstructed", "resume", data={"resume_at": None})
    if "accepted" not in got:
        return Transition(
            "gap",
            "reconstructed",
            "normal",
            data={"resume_at": None},
            note="no Template extracted; the Gaps stay open",
        )
    rounds = int(loop.data.get("round", 1))
    if rounds >= s.max_rounds:
        return hold(loop, f"round limit {s.max_rounds} reached", escalated=True)
    previous = loop.data.get("reconstruction")
    return Transition(
        "gap",
        "analyzing",
        "creates_task",
        data={"resume_at": None, "round": rounds + 1},
        tasks=(_t2(loop, str(previous) if previous else None),),
    )


def _reproduction(loop: Loop, s: Settings) -> Transition:
    f = loop.facts
    err = float(f["error_rel"])
    if err <= s.epsilon_rel:
        return Transition("reconstructed", "reproduced", "normal", data={"error_rel": err})
    gap = Draft(
        "Gap",
        {
            "missing": "family",
            "description": f"reproduction error {err:.3f} above epsilon {s.epsilon_rel}",
            "attribution": "space",
        },
        {"reference": loop.subject, "family": str(f["family"])},
        data_key="reproduction_gap",
    )
    task = TaskRequest(
        "T1", "Extend the family to reproduce the reference", [str(f["family"]), loop.subject]
    )
    return Transition(
        "reconstructed",
        "gap",
        "creates_task",
        data={"resume_at": "reconstructed", "error_rel": err},
        tasks=(task,),
        records=(gap,),
    )
