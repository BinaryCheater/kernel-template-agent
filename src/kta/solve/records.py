"""Writing a solve result as records: Candidate and Prediction per shortlisted point, and
one Report (subject ``solve``) with the method, counts and the optimality claim's scope
(decision D251)."""

from __future__ import annotations

from dataclasses import dataclass

from kta.core.context import Context
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import Report
from kta.solve.result import SolveResult
from kta.store.base import Store

SOLVE_SUBJECT = "solve"


@dataclass(frozen=True)
class SolveRecords:
    candidates: list[str]
    predictions: list[str]
    report: str


def record_solve(
    ctx: Context,
    store: Store,
    result: SolveResult,
    actor: Actor,
    case_id: str | None = None,
    family_id: str | None = None,
    hw_model_id: str | None = None,
    model_id: str | None = None,
) -> SolveRecords:
    """Candidates refer to the case and family; Predictions to their candidate, the
    HwModel and the model version (a ``unit_model`` Report)."""
    now = ctx.clock.now()
    cand_refs = {k: v for k, v in (("case", case_id), ("family", family_id)) if v}
    pred_refs = {k: v for k, v in (("hw_model", hw_model_id), ("model", model_id)) if v}
    # Store.put fills actor.commit/dirty, so a Candidate's ID is known only once it is written.
    cand_ids = [
        store.put(Record.new("Candidate", c.body(), actor, now, refs=dict(cand_refs)))
        for c in result.candidates
    ]
    pred_ids = store.put_many(
        Record.new("Prediction", c.prediction, actor, now, refs={"candidate": cid, **pred_refs})
        for c, cid in zip(result.candidates, cand_ids, strict=True)
    )
    claim = "model-optimal" if result.model_optimal else "no optimality claim"
    report = Report(
        title=f"solve {result.scope.get('family')} bucket {result.scope.get('bucket')}",
        subject=SOLVE_SUBJECT,
        summary=(
            f"{result.method}: {result.feasible} feasible of {result.space_size}, "
            f"exhaustive={result.exhaustive}, {claim}"
        ),
        data={**result.summary(), "candidate_ids": cand_ids, "prediction_ids": pred_ids},
    )
    refs: dict[str, str | list[str]] = {**cand_refs, **pred_refs}
    if cand_ids:
        refs["candidates"] = cand_ids
        refs["predictions"] = pred_ids
    rid = store.put(Record.new("Report", report, actor, now, refs=refs))
    return SolveRecords(cand_ids, pred_ids, rid)
