"""The work of each expand-loop state (docs/design/10 §2.2).

Freezing is checked here. Building the reference in its container, measuring it under our
protocol, reproducing it from a family and ablating chain steps need a GPU worker and the
reference build path (WP11); they raise NotSupported, which blocks the instance in that
state with an inbox item (D264). An analysis-only reference (24 §2 step 1: empty
``runnable_on``, or ``analysis_only`` set on the loop by you, D385) skips them: it is
decomposed and its coverage written without a reference point or a reproduction. The
task-waiting states read task results.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kta.core.errors import ConfigError, NotSupported, SchemaError
from kta.core.ids import record_id
from kta.core.schemas import Actor, Record
from kta.core.schemas.knowledge import Reference
from kta.core.schemas.process import PoolSnapshot
from kta.knowledge import coverage_report
from kta.knowledge.coverage import coverage_page
from kta.knowledge.pool import ALL as MAX_LIMIT
from kta.knowledge.pool import chain_template_refs, find_template
from kta.orchestrator.outputs import RESULT_TITLE
from kta.orchestrator.pipeline import Services
from kta.orchestrator.transition import Loop

Facts = dict[str, Any]


def task_states(svc: Services, loop: Loop) -> dict[str, str]:
    ids = [str(t) for t in loop.data.get("waiting_tasks", [])]
    return {t: svc.ops.task(t).state for t in ids}


def _result(svc: Services, task_id: str) -> Record:
    card = svc.ops.task(task_id).card_id
    found = svc.store.query(
        "Report", {"title": RESULT_TITLE, "subject": card}, order="-created_at", limit=1
    )
    if not found:
        raise SchemaError(f"accepted task {task_id} has no {RESULT_TITLE} Report")
    return found[0]


def task_result(svc: Services, task_id: str) -> dict[str, Any]:
    """The outputs an accepted task wrote (its task_result Report)."""
    return dict(_result(svc, task_id).body["data"]["outputs"])


def analysis_only(loop: Loop, ref: Reference) -> str | None:
    """Why the reference is analysis-only, or None when it is to be built and measured."""
    if loop.data.get("analysis_only"):
        return str(loop.data["analysis_only"])
    if not ref.runnable_on:
        return "runnable_on is empty (analysis only, 24 §2 step 1)"
    return None


def work_frozen(svc: Services, loop: Loop) -> Facts:
    rec = svc.store.get(loop.subject)
    if rec.kind != "Reference":
        raise SchemaError(f"{loop.subject} is a {rec.kind}, not a Reference")
    ref = Reference.model_validate(rec.body)
    source = rec.refs.get("source")
    if not isinstance(source, str):
        raise SchemaError("the Reference names no Source")
    revision = str(svc.store.get(source).body.get("revision", ""))
    if not revision:
        raise SchemaError(f"Source {source[:12]} has no revision: the reference is not frozen")
    reason = analysis_only(loop, ref)
    if reason is None:
        raise NotSupported(
            f"building Reference {ref.name} at {revision[:12]} in its container: intake jobs "
            f"are defined (kta reference plan {loop.subject[:12]}), not run (D569)"
        )
    problem = rec.refs.get("problem")
    return {
        "source": source,
        "problem": problem if isinstance(problem, str) else None,
        "revision": revision,
        "analysis_only": reason,
    }


def work_measured(svc: Services, loop: Loop) -> Facts:
    if loop.data.get("analysis_only"):
        return {"reference_point": None}
    raise NotSupported("measuring a reference build under our protocol (WP11)")


def work_waiting(svc: Services, loop: Loop) -> Facts:
    states = task_states(svc, loop)
    facts: Facts = {"tasks": states}
    if loop.state == "analyzing" and states and all(s == "accepted" for s in states.values()):
        report = _result(svc, next(iter(states)))
        out = report.body["data"]["outputs"]
        chain = svc.store.get(str(out["chain"])).body
        missing = [r for r in chain_template_refs(chain) if find_template(svc.store, r) is None]
        facts.update(
            reconstruction=out["reconstruction"],
            gaps=out.get("gaps", []),
            submission=report.refs.get("submission"),
            to_extract=sorted(set(missing)),
        )
    return facts


# Record kinds a pool snapshot holds (docs/design/20 §5): the knowledge, not the evidence.
POOL_KINDS = (
    "Pattern",
    "Template",
    "Realisation",
    "Structure",
    "Finding",
    "Coupling",
    "Regime",
    "Negative",
    "Capability",
    "CapabilityDecl",
    "HwFeature",
    "HwFact",
    "Family",
    "Chain",
    "Source",
    "Problem",
    "Reference",
    "Reconstruction",
    "Gap",
)
CURRENT_POOL = "current_pool"  # kta.store.snapshot


def pool_snapshot(svc: Services, note: str, actor: Actor) -> str:
    """A PoolSnapshot of every knowledge record, made current (D387).

    ``kta.store.snapshot.snapshot`` does the same on a SqliteStore; the Services store is
    the Store protocol, so the record and the compare-and-set are written here.
    """
    members = sorted(
        {record_id(r) for k in POOL_KINDS for r in svc.store.query(k, limit=MAX_LIMIT)}
    )
    parent = svc.ops.get_pointer(CURRENT_POOL)
    refs: dict[str, Any] = {"members": members}
    if parent:
        refs["previous"] = parent
    body = PoolSnapshot(members=members, note=note)
    sid = svc.store.put(Record.new("PoolSnapshot", body, actor, svc.ctx.clock.now(), refs=refs))
    if not svc.ops.set_pointer(CURRENT_POOL, parent, sid):
        raise ConfigError(f"current_pool moved while snapshot {sid[:12]} was written")
    return sid


def _trend(svc: Services) -> list[dict[str, Any]]:
    reports = svc.store.query("Report", {"title": "coverage"}, order="created_at", limit=MAX_LIMIT)
    return [dict(t) for r in reports for t in r.body["data"].get("trend", [])[-1:]]


def write_coverage(svc: Services, loop: Loop) -> str:
    """Snapshot the pool and write its ``coverage`` Report (state ``covered``, 10 §2.2).

    The Report has the coverage page's fields (spec/modules/api-web.md) plus ``report``,
    the full CoverageReport, and the rng seed of the discovery shuffles.
    """
    actor = Actor(type="system", name="kta.orchestrator.expand_loop")
    name = str(loop.data.get("name", loop.subject[:12]))
    sid = pool_snapshot(svc, f"after the expand loop of {name}", actor)
    stream = "coverage"
    rep = coverage_report(svc.store, sid, svc.ctx.rng(stream), controls=svc.ctx.controls)
    states = {str(r["subject"]): str(r["state"]) for r in svc.ops.list_loops("expand")}
    states[loop.subject] = "covered"
    page = coverage_page(rep, states, _trend(svc))
    refs: dict[str, Any] = {"snapshot": sid, "reference": loop.subject}
    if loop.data.get("reconstruction"):
        refs["reconstruction"] = str(loop.data["reconstruction"])
    body = {
        "title": "coverage",
        "subject": sid,
        "summary": (
            f"{rep.reconstructions} references, {rep.templates_observed} Templates observed, "
            f"Chao1 {rep.chao1:.1f}, {len(rep.templates_missing)} Templates not extracted"
        ),
        "data": {**page, "report": rep.to_dict(), "rng": {"seed": svc.ctx.seed, "stream": stream}},
    }
    return svc.store.put(Record.new("Report", body, actor, svc.ctx.clock.now(), refs=refs))


def work_reconstructed(svc: Services, loop: Loop) -> Facts:
    if loop.data.get("analysis_only"):
        return {"analysis_only": True, "coverage": write_coverage(svc, loop)}
    raise NotSupported(
        "reproducing the performance core from a family: planned (kta reference plan), not run"
    )


def work_ablate(svc: Services, loop: Loop) -> Facts:
    raise NotSupported(
        "ablating our family (kta.knowledge.ablation design): planned, not run (D566)"
    )


WORK: dict[str, Callable[[Services, Loop], Facts]] = {
    "frozen": work_frozen,
    "measured": work_measured,
    "analyzing": work_waiting,
    "gap": work_waiting,
    "reconstructed": work_reconstructed,
    "reproduced": work_ablate,
    "ablated": work_ablate,
}
