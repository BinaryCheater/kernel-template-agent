"""Canaries (D-T17, 12 §3 canary rate; WP18 D528): planted faults the router must catch.

Each batch the summarize process takes the newest GapDecomposition of the window, plants
the next fault kind of routing.yaml ``canaries`` into one of its buckets' recorded inputs,
re-runs the decomposition and the classification, and writes a Report (title ``canary``)
with the verdict:

- ``caught``: the planted input is classed as the kind expects;
- ``missed``: it is not; with ``escalation.to_user.on_canary_missed`` the inbox lists it
  until a later canary of the same kind is caught;
- ``uninformative``: the base bucket already had the expected class (nothing is learned);
- ``no_base``: the window has no decomposition to plant into.

The canaries test the decomposition and routing on real recorded inputs; they do not plant
wrong kernels into the correctness checkers (that needs GPU runs, WP18 pending real runs).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from kta.core.ids import record_id
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import Report
from kta.orchestrator import gap
from kta.orchestrator.pipeline import Services
from kta.orchestrator.solve_loop import bucket_metrics, model_off
from kta.orchestrator.transition import Settings

ACTOR = Actor(type="checker", name="kta.orchestrator.canary")
TITLE = "canary"


def plant(inputs: Mapping[str, Any], how: Mapping[str, Any]) -> dict[str, Any]:
    """The bucket inputs with one fault planted (routing.yaml ``plant``)."""
    b: dict[str, Any] = {**inputs, "name": "canary", "failed_checks": []}
    pairs = [tuple(map(float, p)) for p in inputs.get("pairs", [])]
    if "scale_predictions" in how:
        k = float(how["scale_predictions"])
        pairs = [(p * k, m) for p, m in pairs]
        b["predicted_us"] = float(inputs["predicted_us"]) * k
        if inputs.get("top_pred_us") is not None:
            b["top_pred_us"] = float(inputs["top_pred_us"]) * k
    if "scale_all" in how:  # the space got worse; the model still agrees with it
        k = float(how["scale_all"])
        pairs = [(p * k, m * k) for p, m in pairs]
        for key in ("best_us", "predicted_us", "top_pred_us", "ci_us"):
            if inputs.get(key) is not None:
                b[key] = float(inputs[key]) * k
    if how.get("no_reference"):
        b["reference_us"] = None
    if how.get("uncertify"):
        b["solver"] = {"method": "enumerate", "model_optimal": False, "exhaustive": False}
    if how.get("fail_check"):
        b["failed_checks"] = [["canary", "canary"]]
    b["pairs"] = pairs
    return b


def classify_inputs(b: Mapping[str, Any], s: Settings) -> str:
    """The class ``judge`` would give one bucket (same functions, same order)."""
    table = gap.routing()
    dec = gap.decompose(b, s.improvement_band)
    off = model_off(bucket_metrics(b), s) if b.get("pairs") else False
    return gap.classify(dec, table.order, bool(b.get("failed_checks")), off)


def _kind(svc: Services, kinds: Sequence[Mapping[str, Any]], rotate: bool) -> Mapping[str, Any]:
    if not rotate:
        return kinds[0]
    done = len(svc.store.query("Report", {"title": TITLE}, limit=100000))
    return kinds[done % len(kinds)]


def run(svc: Services, batch: str, decs: Sequence[Record], s: Settings, now: datetime) -> list[str]:
    """``per_batch`` canaries for one batch; their Report IDs."""
    kinds = gap.routing().canaries
    out: list[str] = []
    for _ in range(int(s.canary["per_batch"]) if kinds else 0):
        kind = _kind(svc, kinds, bool(s.canary["rotate_kinds"]))
        base = next((d for d in reversed(decs) if _usable(d)), None)
        data: dict[str, Any] = {"batch": batch, "kind": kind["kind"], "expect": kind["expect"]}
        refs: dict[str, Any] = {}
        if base is None:
            data["verdict"] = "no_base"
        else:
            bucket = next(b for b in base.body["buckets"] if b.get("inputs"))
            refs["decomposition"] = record_id(base)
            got = classify_inputs(plant(bucket["inputs"], kind["plant"]), s)
            data.update(base_class=bucket["class"], got=got, bucket=bucket["name"])
            if bucket["class"] == kind["expect"]:
                data["verdict"] = "uninformative"
            else:
                data["verdict"] = "caught" if got == kind["expect"] else "missed"
        data["escalate"] = data["verdict"] == "missed" and bool(s.to_user["on_canary_missed"])
        body = Report(
            title=TITLE,
            subject=batch,
            summary=f"canary {kind['kind']}: {data['verdict']}",
            data={**data, "routing": gap.routing().id},
        )
        out.append(svc.store.put(Record.new("Report", body, ACTOR, now, refs)))
    return out


def _usable(rec: Record) -> bool:
    return any(b.get("inputs", {}).get("best_us") for b in rec.body.get("buckets", []))


def missed(svc: Services) -> list[Record]:
    """Escalated missed canaries whose kind has not been caught since."""
    reps = svc.store.query("Report", {"title": TITLE}, order="created_at", limit=100000)
    last: dict[str, Record] = {}
    for r in reps:
        if r.body["data"]["verdict"] in ("caught", "missed"):
            last[str(r.body["data"]["kind"])] = r
    return [r for r in last.values() if r.body["data"].get("escalate")]
