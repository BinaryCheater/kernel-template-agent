"""Pool views: /families, /families/{id}/chain, /snapshots, /pool/diff, /coverage,
/model/residuals.

A family belongs to a snapshot when its record is a member of that PoolSnapshot. The
coverage page shows a stored Report with ``title == "coverage"`` whose ``subject`` is the
snapshot ID; the API does not compute coverage itself (that is WP10b/WP11).
"""

from __future__ import annotations

from typing import Any

from kta.api.common import Api, body, envelope, incoming, latest, ref_one
from kta.core.errors import NotSupported
from kta.knowledge import pooldiff

CURRENT_POOL = "current_pool"


def _domain(d: Any) -> str:
    if isinstance(d, dict):
        return f"{d['lo']}..{d['hi']}" + (" pow2" if d.get("pow2") else "")
    return "{" + ",".join(str(x) for x in d) + "}"  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]


def _ids(api: Api, kind: str) -> list[str]:
    with api.store.db.read() as con:
        rows = con.execute(
            "SELECT id FROM records WHERE kind = ? ORDER BY created_at", (kind,)
        ).fetchall()
    return [r[0] for r in rows]


def snapshots(api: Api) -> list[dict[str, Any]]:
    current = api.ops.get_pointer(CURRENT_POOL)
    out = []
    for sid in _ids(api, "PoolSnapshot"):
        rec = api.store.get_json(sid)
        note = rec["body"]["note"]
        out.append(
            {
                "id": sid,
                "label": f"{note}{' · current' if sid == current else ''}",
                "created_at": rec["created_at"],
                "members": len(rec["body"]["members"]),
                "current": sid == current,
            }
        )
    return out


def _members(api: Api, snapshot: str | None) -> set[str] | None:
    return set(body(api, snapshot)["members"]) if snapshot else None


def _latest_versions(api: Api, members: set[str] | None) -> list[str]:
    best: dict[str, tuple[int, str]] = {}
    for fid in _ids(api, "Family"):
        if members is not None and fid not in members:
            continue
        b = body(api, fid)
        if b["name"] not in best or b["version"] > best[b["name"]][0]:
            best[b["name"]] = (b["version"], fid)
    return [fid for _, fid in sorted(best.values(), key=lambda x: body(api, x[1])["name"])]


def _summary(api: Api, fid: str) -> dict[str, Any]:
    b = body(api, fid)
    chain = ref_one(api, fid, "chain")
    problem = ref_one(api, fid, "problem")
    items = body(api, chain)["items"] if chain else []
    steps = []
    for it in items:
        tid = _template(api, it["template"])
        t = body(api, tid) if tid else {}
        steps.append(
            {
                "key": it["key"],
                "template": it["template"],
                "level_from": t.get("level_from"),
                "level_to": t.get("level_to"),
                "assurance": (api.store.status(tid)[1] if tid else None) or "declared",
            }
        )
    versions = [x for x in _ids(api, "Family") if body(api, x)["name"] == b["name"]]
    return {
        "id": fid,
        "name": b["name"],
        "version": b["version"],
        "backend": b["backend"],
        "problem": body(api, problem)["name"] if problem else "",
        "steps": len(items),
        "solutions": len(incoming(api, fid, "family", "Solution")),
        "step_detail": steps,
        "versions": len(versions),
        "updated_at": api.store.get_json(fid)["created_at"],
    }


def families(api: Api, snapshot: str | None) -> list[dict[str, Any]]:
    return [_summary(api, f) for f in _latest_versions(api, _members(api, snapshot))]


def _template(api: Api, ref: str) -> str | None:
    name, _, version = ref.partition("@")
    with api.store.db.read() as con:
        row = con.execute(
            "SELECT id FROM records WHERE kind = 'Template' "
            "AND json_extract(json, '$.body.name') = ? "
            "AND json_extract(json, '$.body.version') = ? LIMIT 1",
            (name, int(version)),
        ).fetchone()
    return str(row[0]) if row else None


def chain(api: Api, fid: str) -> dict[str, Any]:
    fam = body(api, fid)
    cid = ref_one(api, fid, "chain")
    items = body(api, cid)["items"] if cid else []
    nodes = []
    for it in items:
        tid = _template(api, it["template"])
        t = body(api, tid) if tid else {}
        env = envelope(api, tid) if tid else {}
        nodes.append(
            {
                "key": it["key"],
                "template": it["template"],
                "template_record": tid,
                "applies_to": it["applies_to"],
                "after": it["after"],
                "bindings": it["bindings"],
                "level_from": t.get("level_from"),
                "level_to": t.get("level_to"),
                "change": t.get("change"),
                "assurance": env.get("assurance") or "declared",
                "status": env.get("status"),
                "summary": t.get("summary", "Template record not found"),
                "requires": t.get("requires", []),
                "probed_on": t.get("probed_on", []),
                "variables": [
                    {"name": v["name"], "domain": _domain(v["domain"]), "binding": v["binding"]}
                    for v in t.get("variables", [])
                ],
            }
        )
    versions = []
    for other in _ids(api, "Family"):
        b = body(api, other)
        if b["name"] == fam["name"]:
            versions.append(
                {
                    "version": b["version"],
                    "record": other,
                    "created_at": api.store.get_json(other)["created_at"],
                    "note": "",
                }
            )
    sols = incoming(api, fid, "family", "Solution")
    return {
        **_summary(api, fid),
        "workload": body(api, cid)["workload"] if cid else "",
        "chain_record": cid,
        "nodes": nodes,
        "variables": [
            {"name": v["name"], "domain": _domain(v["domain"]), "meaning": v["meaning"]}
            for v in fam["variables"]
        ],
        "glue": [
            {"id": g["id"], "expr": g["expr"], "meaning": g["meaning"]}
            for g in fam["glue_constraints"]
        ],
        "versions": sorted(versions, key=lambda v: v["version"]),
        "solution_ids": [
            {
                "id": s,
                "label": f"{body(api, s).get('bucket') or 'all'} · "
                f"{body(api, s)['pct_of_bound'] * 100:.0f}% of bound",
            }
            for s in sols
        ],
    }


def diff(api: Api, a: str, b: str) -> dict[str, Any]:
    """GET /pool/diff: what changed between snapshots ``a`` and ``b`` (E14, D449)."""
    return pooldiff.diff(api.store, a, b).to_dict()


def coverage(api: Api, snapshot: str) -> dict[str, Any]:
    reports = [
        r
        for r in _ids(api, "Report")
        if body(api, r)["title"] == "coverage" and body(api, r)["subject"] == snapshot
    ]
    rid = latest(api, reports)
    if rid is None:
        raise NotSupported(
            f"no coverage report for snapshot {snapshot[:12]} yet (computed by WP10b/WP11)"
        )
    return {"snapshot": snapshot, "record": rid, **body(api, rid)["data"]}


def residuals(api: Api, model: str | None) -> list[dict[str, Any]]:
    """Predicted vs measured latency for every measured candidate (/model/residuals)."""
    out = []
    for pid in _ids(api, "Prediction"):
        p = body(api, pid)
        if model and p["model"] != model:
            continue
        cand = ref_one(api, pid, "candidate")
        build = latest(api, incoming(api, cand, "candidate", "Build")) if cand else None
        meas = latest(api, incoming(api, build, "build", "Measurement")) if build else None
        if meas is None:
            continue
        m = float(body(api, meas)["median_us"])
        out.append(
            {
                "prediction": pid,
                "measurement": meas,
                "model": p["model"],
                "predicted_us": p["total"],
                "measured_us": m,
                "rel_error": (p["total"] - m) / m,
            }
        )
    return out
