"""Calibration: fit a model on stored (Prediction, Measurement) pairs and keep every version.

A model version is a Report record (subject ``unit_model``, decision D245): its data hold
the parameters, the fit summary with the hold-out error, and the IDs of the records it
was fitted on; ``refs.previous`` points at the version it was fitted from.
"""

from __future__ import annotations

from kta.core.context import Context
from kta.core.errors import SchemaError
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Measurement, Prediction
from kta.core.schemas.process import Report
from kta.models.unit_model import UnitModel
from kta.store.base import Store

MODEL_SUBJECT = "unit_model"


def save_model(
    ctx: Context,
    store: Store,
    model: UnitModel,
    actor: Actor,
    previous: str | None = None,
    fitted_on: list[tuple[str, str]] | None = None,
) -> str:
    """Write one model version as a Report; returns its record ID."""
    pairs = fitted_on or []
    data = {**model.to_dict(), "fitted_on": [list(p) for p in pairs]}
    holdout = model.fit_info.get("holdout", {})
    summary = (
        f"UnitModel {model.version} for {model.target}; fitted on {len(pairs)} pairs; "
        f"hold-out median |rel err| {holdout.get('median_abs_rel_err')}"
    )
    refs: dict[str, str | list[str]] = {}
    if previous:
        refs["previous"] = previous
    if pairs:
        refs["predictions"] = [p for p, _ in pairs]
        refs["measurements"] = [m for _, m in pairs]
    body = Report(
        title=f"UnitModel {model.version}", subject=MODEL_SUBJECT, summary=summary, data=data
    )
    return store.put(Record.new("Report", body, actor, ctx.clock.now(), refs=refs))


def load_model(store: Store, record_id: str) -> UnitModel:
    rec = store.get(record_id)
    if rec.kind != "Report" or rec.body.get("subject") != MODEL_SUBJECT:
        raise SchemaError(f"{record_id} is not a {MODEL_SUBJECT} Report")
    return UnitModel.from_dict(rec.body["data"])


def calibrate(
    ctx: Context,
    store: Store,
    model_id: str,
    pairs: list[tuple[str, str]],
    actor: Actor,
) -> tuple[UnitModel, str]:
    """Fit the stored model ``model_id`` on (Prediction ID, Measurement ID) pairs.

    The new version is written next to the old one, which stays unchanged.
    """
    model = load_model(store, model_id)
    data: list[tuple[Prediction, Measurement]] = []
    for pred_id, meas_id in pairs:
        p, m = store.get(pred_id), store.get(meas_id)
        if p.kind != "Prediction" or m.kind != "Measurement":
            raise SchemaError(f"({pred_id}, {meas_id}) is not a (Prediction, Measurement) pair")
        data.append((Prediction.model_validate(p.body), Measurement.model_validate(m.body)))
    new = model.fit(data)
    return new, save_model(ctx, store, new, actor, previous=model_id, fitted_on=pairs)
