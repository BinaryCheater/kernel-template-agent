"""Review support for supervised trials (SYSDEBUG-2, D690, D691).

GET /settings/in-force: every knob a run reads, with the value in force, who set it and
whether anything calibrated it, grouped by the step that reads it (knob_stages.yaml). With
``loop``, the controls versions that loop ran under. Hardware-model values follow with
their provenance status.

Annotations: the user's verdict on any step, record or knob (agree / question / change),
stored as a HumanAction ``annotate {...}`` (no new record kind); ``resolve`` closes one.
GET /annotations lists them, newest first, with their state, as the review list.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict

from kta.api import operate
from kta.api.common import Api, iso
from kta.core.controls import definitions
from kta.core.ids import is_record_id, record_id
from kta.store import controls as controls_store

VERDICTS = ("agree", "question", "change")
WHO = {
    "default": "assistant default, not calibrated",
    "spec": "design documents (not confirmed by you)",
}


@cache
def stage_table() -> dict[str, list[str]]:
    raw = yaml.safe_load((Path(__file__).with_name("knob_stages.yaml")).read_text("utf-8"))
    return {str(k): [str(x) for x in (v or [])] for k, v in raw.items()}


def _stages_of() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for stage, knobs in stage_table().items():
        for k in knobs:
            out.setdefault(k, []).append(stage)
    return out


def _last_changes(api: Api) -> dict[str, dict[str, Any]]:
    """Per knob, the newest stored version that changed it: version, author, reason, time."""
    out: dict[str, dict[str, Any]] = {}
    for rid, rec in controls_store.history(api.store, 1000):  # newest first
        for c in controls_store.changes_of(rec):
            out.setdefault(
                c.knob,
                {
                    "version": rec.body["version"],
                    "record": rid,
                    "by": rec.actor.name,
                    "by_type": rec.actor.type,
                    "reason": rec.body["reason"],
                    "at": iso(rec.created_at),
                    "old": c.old,
                },
            )
    return out


def in_force(api: Api, loop: str | None = None) -> dict[str, Any]:
    snap = controls_store.current(api.store, api.ops)
    defs = definitions()
    stages = _stages_of()
    changed = _last_changes(api)
    notes = annotation_index(api)
    knobs = []
    for k in defs.knobs:
        last = changed.get(k.id)
        human = last is not None and last["by_type"] == "human"
        who = f"you (version {last['version']})" if human and last else WHO.get(k.status, k.status)
        knobs.append(
            {
                "id": k.id,
                "row": k.row,
                "controls": k.controls,
                "value": snap.value(k.id),
                "initial": k.value,
                "unit": k.unit,
                "status": k.status,
                "who": who,
                "set_by_you": human,
                "calibrated": bool(last and last["by_type"] == "process"),
                "set_by": list(k.set_by),
                "set_from": k.set_from,
                "source": k.source,
                "prevents": k.prevents,
                "consumer": k.consumer,
                "editable": k.editable,
                "stages": stages.get(k.id, ["other"]),
                "last_change": last,
                "annotations": notes.get(f"knob:{k.id}", []),
            }
        )
    seen: list[str] = []
    if loop:
        data = cast(dict[str, Any], api.ops.get_loop(loop)["data"])
        seen = [str(v) for v in cast(list[Any], data.get("controls_seen") or [])]
    count = Counter(
        "you" if x["set_by_you"] else ("calibrated" if x["calibrated"] else x["status"])
        for x in knobs
    )
    return {
        "version": {"id": snap.version_id, "version": snap.version},
        "loop": loop,
        "versions_seen": [_version_brief(api, v) for v in seen],
        "stages": [*stage_table(), "other"],
        "counts": dict(count),
        "knobs": knobs,
        "hardware": hardware(api),
    }


def _version_brief(api: Api, rid: str) -> dict[str, Any]:
    rec = api.store.get(rid)
    return {
        "id": rid,
        "version": rec.body["version"],
        "at": iso(rec.created_at),
        "by": rec.actor.name,
        "reason": rec.body["reason"],
    }


def hardware(api: Api) -> dict[str, Any]:
    """The newest HwModel's values with their provenance status (L1 reads these)."""
    recs = api.store.query("HwModel", order="-created_at", limit=1)
    if not recs:
        return {"model": None, "values": [], "counts": {}}
    rec = recs[0]
    body = rec.body
    prov = cast(dict[str, Any], body.get("provenance") or {})
    values = []
    for name, value in sorted(cast(dict[str, Any], body.get("limits") or {}).items()):
        p = cast(dict[str, Any], prov.get(f"limits.{name}") or {})
        values.append(
            {
                "name": f"hw.{name}",
                "value": value,
                "status": p.get("status", "unknown"),
                "source": p.get("source", ""),
            }
        )
    return {
        "model": {
            "id": record_id(rec),
            "target": body.get("target"),
            "version": body.get("version"),
            "arch": body.get("arch"),
        },
        "values": values,
        "counts": dict(Counter(v["status"] for v in values)),
    }


# ---- annotations ---------------------------------------------------------------


class AnnotateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str  # a record ID, "knob:<id>", "loop:<id>:<state>" or "page:<path>"
    verdict: Literal["agree", "question", "change"]
    note: str = ""
    loop: str | None = None
    label: str = ""  # what the user was looking at, in words (shown in the review list)


class ResolveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    annotation: str
    outcome: Literal["done", "kept", "later"]
    note: str = ""


def annotate(api: Api, b: AnnotateBody) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    if is_record_id(b.subject):
        api.store.get(b.subject)  # 404 for an unknown record
        refs["subject"] = b.subject
    params: dict[str, Any] = {"subject": b.subject, "verdict": b.verdict, "label": b.label}
    if b.loop:
        params["loop"] = b.loop
    action = operate.write_action(api, "annotate", b.note, refs or None, params)
    return {"annotation": action}


def resolve(api: Api, b: ResolveBody) -> dict[str, Any]:
    rec = api.store.get(b.annotation)
    if rec.kind != "HumanAction" or not str(rec.body["action"]).startswith("annotate "):
        raise ValueError(f"{b.annotation} is not an annotation")
    action = operate.write_action(
        api,
        "resolve",
        b.note,
        {"annotation": b.annotation},
        {"annotation": b.annotation, "outcome": b.outcome},
    )
    return {"resolution": action}


def _params(action: str, verb: str) -> dict[str, Any] | None:
    if not action.startswith(verb + " "):
        return None
    try:
        return cast(dict[str, Any], json.loads(action[len(verb) + 1 :]))
    except json.JSONDecodeError:
        return None


def list_annotations(api: Api, open_only: bool = False) -> list[dict[str, Any]]:
    recs = api.store.query("HumanAction", order="-created_at", limit=5000)
    resolved: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for rec in recs:
        p = _params(str(rec.body["action"]), "resolve")
        if p and "annotation" in p:
            resolved.setdefault(
                str(p["annotation"]),
                {
                    "outcome": p.get("outcome"),
                    "note": rec.body["reason"],
                    "at": iso(rec.created_at),
                },
            )
    for rec in recs:
        p = _params(str(rec.body["action"]), "annotate")
        if p is None:
            continue
        rid = record_id(rec)
        res = resolved.get(rid)
        if open_only and res:
            continue
        out.append(
            {
                "id": rid,
                "at": iso(rec.created_at),
                "subject": p.get("subject"),
                "verdict": p.get("verdict"),
                "label": p.get("label", ""),
                "loop": p.get("loop"),
                "note": rec.body["reason"],
                "resolution": res,
            }
        )
    return out


def annotation_index(api: Api) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for a in list_annotations(api):
        out.setdefault(str(a["subject"]), []).append(a)
    return out


def router(api: Api) -> APIRouter:
    r = APIRouter()

    @r.get("/settings/in-force")
    def get_in_force(loop: str | None = None) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return in_force(api, loop)

    @r.get("/annotations")
    def get_annotations(open: bool = Query(False)) -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return list_annotations(api, open)

    @r.post("/actions/annotate")
    def post_annotate(b: AnnotateBody) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return annotate(api, b)

    @r.post("/actions/resolve")
    def post_resolve(b: ResolveBody) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return resolve(api, b)

    return r
