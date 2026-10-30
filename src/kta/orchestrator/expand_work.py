"""The work of each expand-loop state (docs/design/10 §2.2).

Freezing is checked here. Building the reference in its container, measuring it under our
protocol, reproducing it from a family and ablating chain steps need the learn process
and the reference build path (WP11); they raise NotSupported, which blocks the instance
in that state with an inbox item (D264). The task-waiting states read task results.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kta.core.errors import NotSupported, SchemaError
from kta.core.schemas.knowledge import Reference
from kta.orchestrator.outputs import RESULT_TITLE
from kta.orchestrator.pipeline import Services
from kta.orchestrator.transition import Loop

Facts = dict[str, Any]


def task_states(svc: Services, loop: Loop) -> dict[str, str]:
    ids = [str(t) for t in loop.data.get("waiting_tasks", [])]
    return {t: svc.ops.task(t).state for t in ids}


def task_result(svc: Services, task_id: str) -> dict[str, Any]:
    """The outputs an accepted task wrote (its task_result Report)."""
    card = svc.ops.task(task_id).card_id
    found = svc.store.query(
        "Report", {"title": RESULT_TITLE, "subject": card}, order="-created_at", limit=1
    )
    if not found:
        raise SchemaError(f"accepted task {task_id} has no {RESULT_TITLE} Report")
    return dict(found[0].body["data"]["outputs"])


def work_frozen(svc: Services, loop: Loop) -> Facts:
    rec = svc.store.get(loop.subject)
    if rec.kind != "Reference":
        raise SchemaError(f"{loop.subject} is a {rec.kind}, not a Reference")
    Reference.model_validate(rec.body)
    source = rec.refs.get("source")
    if not isinstance(source, str):
        raise SchemaError("the Reference names no Source")
    revision = str(svc.store.get(source).body.get("revision", ""))
    if not revision:
        raise SchemaError(f"Source {source[:12]} has no revision: the reference is not frozen")
    raise NotSupported(
        f"building Reference {rec.body['name']} at {revision[:12]} in its container (WP11)"
    )


def work_measured(svc: Services, loop: Loop) -> Facts:
    raise NotSupported("measuring a reference build under our protocol (WP11)")


def work_waiting(svc: Services, loop: Loop) -> Facts:
    states = task_states(svc, loop)
    facts: Facts = {"tasks": states}
    if loop.state == "analyzing" and states and all(s == "accepted" for s in states.values()):
        out = task_result(svc, next(iter(states)))
        facts.update(reconstruction=out["reconstruction"], gaps=out.get("gaps", []))
    return facts


def work_reconstructed(svc: Services, loop: Loop) -> Facts:
    raise NotSupported("reproducing a reference from a family (WP11)")


def work_ablate(svc: Services, loop: Loop) -> Facts:
    raise NotSupported("ablating chain steps of a reproduced reference (WP11)")


WORK: dict[str, Callable[[Services, Loop], Facts]] = {
    "frozen": work_frozen,
    "measured": work_measured,
    "analyzing": work_waiting,
    "gap": work_waiting,
    "reconstructed": work_reconstructed,
    "reproduced": work_ablate,
    "ablated": work_ablate,
}
