"""Expand loop: pure transition function (docs/design/10 §2.2, machines.yaml ``expand``).

Facts each state's work reports (``kta.orchestrator.expand_work``):

- frozen: ``source``, ``revision``
- measured: ``reference_point`` (Measurement ID)
- analyzing, gap: ``tasks`` {task ID: lifecycle state}; when the awaited T2 is accepted,
  ``reconstruction`` and ``gaps`` [{"id", "missing"}] from its outputs
- reconstructed: ``error_rel`` (reproduction error), ``family``, ``measurement``
- reproduced, ablated: ``report`` (Report ID of the reproduction / ablation)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kta.core.context import Context
from kta.orchestrator.solve_loop import STUCK_TASK_STATES
from kta.orchestrator.transition import Draft, Loop, Settings, TaskRequest, Transition, hold

LINEAR = {"frozen": "measured", "reproduced": "ablated", "ablated": "covered"}


def _t2(loop: Loop) -> TaskRequest:
    return TaskRequest("T2", "Decompose this reference into a composition chain", [loop.subject])


def _awaited(loop: Loop) -> tuple[list[str], list[str]]:
    """(states of the awaited tasks, those that are stuck)."""
    states: Mapping[str, str] = loop.facts.get("tasks", {})
    ids = [str(t) for t in loop.data.get("waiting_tasks", [])]
    got = [states.get(t, "") for t in ids]
    return got, [s for s in got if s in STUCK_TASK_STATES]


def step(loop: Loop, ctx: Context) -> Transition:
    s = Settings.from_config(ctx.config)
    f = loop.facts
    if loop.state in LINEAR:
        return Transition(loop.state, LINEAR[loop.state], "normal", data=dict(f))
    if loop.state == "measured":
        data = {"reference_point": f["reference_point"]}
        return Transition("measured", "analyzing", "creates_task", data=data, tasks=(_t2(loop),))
    if loop.state in ("analyzing", "gap"):
        got, stuck = _awaited(loop)
        if stuck:
            return hold(loop, f"awaited task is {stuck[0]}", escalated=True)
        if not got or any(g != "accepted" for g in got):
            return hold(loop, "waiting for tasks")
        return _analyzed(loop) if loop.state == "analyzing" else _filled(loop)
    if loop.state == "reconstructed":
        return _reproduction(loop, s)
    return hold(loop, f"{loop.state} has no outgoing transition")


def _analyzed(loop: Loop) -> Transition:
    gaps: list[Mapping[str, Any]] = list(loop.facts.get("gaps", []))
    data = {"reconstruction": loop.facts["reconstruction"]}
    if not gaps:
        return Transition("analyzing", "reconstructed", "normal", data=data)
    tasks = tuple(
        TaskRequest(
            "T1" if g.get("missing") == "family" else "T3",
            "Extend the family for this Gap"
            if g.get("missing") == "family"
            else "Extract a Template for this Gap",
            [str(g["id"]), loop.subject],
        )
        for g in gaps
    )
    data["resume_at"] = "analyzing"
    return Transition("analyzing", "gap", "creates_task", data=data, tasks=tasks)


def _filled(loop: Loop) -> Transition:
    if loop.data.get("resume_at") == "reconstructed":
        return Transition("gap", "reconstructed", "resume", data={"resume_at": None})
    return Transition(
        "gap", "analyzing", "creates_task", data={"resume_at": None}, tasks=(_t2(loop),)
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
