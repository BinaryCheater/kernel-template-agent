"""GET /mechanisms (WP22, D591): what the round-4 mechanisms have recorded so far.

One read view for the web page "Mechanisms": FormulationPlans and their diff against the
deterministic default (WP19), L2 what-if runs (WP16), canaries (WP18), selection arms per
gap bucket (WP18), learning-loop stages per Reference (WP20, a dry-run plan computed on
read, D569) and a pointer to X1–X12 (WP21, shown on Settings). Every row carries the
record's origin so the page can label FIXTURE data. Nothing here writes.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from fastapi import APIRouter

from kta.api.common import Api, iso
from kta.core.controls import definitions
from kta.core.errors import KtaError
from kta.core.ids import record_id
from kta.core.schemas import Record
from kta.orchestrator import access, learning_plan

LIMIT = 50
STAGES = ("intake", "decompose", "admit", "reproduce", "ablate", "cover")


def _row(rec: Record) -> dict[str, Any]:
    return {"id": record_id(rec), "at": iso(rec.created_at), "origin": rec.origin}


def plans(api: Api) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in api.store.query("FormulationPlan", order="-created_at", limit=LIMIT):
        rid = record_id(rec)
        view = access.plan_view(api.store, rid)
        diff = view.get("diff") or []
        out.append(
            {
                **_row(rec),
                "family": rec.body.get("family"),
                "author": rec.body.get("author"),
                "chain_source": rec.body.get("chain_source"),
                "case": rec.refs.get("case"),
                "default": view.get("default"),
                "diff": diff,
                "uncited": [f"{c['section']}:{c['key']}" for c in diff if not c["cited"]],
            }
        )
    return out


def what_ifs(api: Api) -> list[dict[str, Any]]:
    keys = ("bottleneck", "shadow_price_us", "delta_total_us", "base_total_us", "moved_total_us")
    return [
        {
            **_row(rec),
            "model": rec.body["subject"],
            "summary": rec.body["summary"],
            "peaks": rec.body["data"].get("peaks", {}),
            "walls": rec.body["data"].get("walls", {}),
            **{k: rec.body["data"].get(k) for k in keys},
        }
        for rec in api.store.query("Report", {"title": "what_if"}, "-created_at", LIMIT)
    ]


def canaries(api: Api) -> dict[str, Any]:
    rows = [
        {
            **_row(rec),
            "batch": rec.body["subject"],
            **{
                k: rec.body["data"].get(k)
                for k in ("kind", "expect", "got", "verdict", "bucket", "escalate")
            },
        }
        for rec in api.store.query("Report", {"title": "canary"}, "-created_at", 500)
    ]
    by_kind: dict[str, Counter[str]] = {}
    for r in rows:
        by_kind.setdefault(str(r["kind"]), Counter())[str(r["verdict"])] += 1
    return {"rows": rows[:LIMIT], "by_kind": {k: dict(c) for k, c in by_kind.items()}}


def selection(api: Api) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in api.store.query("GapDecomposition", order="-created_at", limit=LIMIT):
        for b in rec.body.get("buckets", []):
            sel = b.get("selection")
            if sel:
                out.append(
                    {
                        **_row(rec),
                        "case": rec.refs.get("case"),
                        "round": rec.body.get("round"),
                        "bucket": b.get("name"),
                        **sel,
                    }
                )
    return out


def learning(api: Api, target: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in api.store.query("Reference", order="-created_at", limit=LIMIT):
        rid = record_id(rec)
        row: dict[str, Any] = {**_row(rec), "name": rec.body.get("name"), "target": target}
        try:
            p = learning_plan.plan(api.store, rid, target)
        except KtaError as exc:  # e.g. a Source without a frozen revision: shown, not hidden
            row["error"] = str(exc)
            out.append(row)
            continue
        row.update(
            source_class=p.source_class,
            priority=p.priority,
            tiers=p.tiers,
            stages=[
                {
                    "name": s.name,
                    "status": s.status,
                    "why": s.why,
                    "steps": len(s.jobs) + len(s.runs),
                }
                for s in p.stages
            ],
        )
        out.append(row)
    return out


def mechanisms_view(api: Api, target: str) -> dict[str, Any]:
    exps = definitions().experiments
    return {
        "plans": plans(api),
        "what_if": what_ifs(api),
        "canaries": canaries(api),
        "selection": selection(api),
        "learning": learning(api, target),
        "stages": list(STAGES),
        "experiments": {
            "count": len(exps),
            "by_status": dict(Counter(e.status for e in exps)),
            "page": "/settings",
        },
    }


def router(api: Api) -> APIRouter:
    r = APIRouter()

    @r.get("/mechanisms")
    def get_mechanisms(target: str = "gb10") -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return mechanisms_view(api, target)

    return r
