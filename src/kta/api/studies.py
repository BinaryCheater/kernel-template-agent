"""GET /pool, /experiments and /improvements (design/30 pages Pool, Experiments, Improvement).

These pages read records only. The conventions below are the API's (decision D345, needs
user review); WP10b, WP11 and WP12 write the records this way:

- Pool: Pattern, Template, Structure, Gap, Reference and Reconstruction records; with a
  snapshot, only records that are members of that PoolSnapshot (Gaps are listed always).
- Experiments: a Protocol with ``type == "experiment"``; ``params.arms`` (list of names) and
  ``params.runs_per_arm`` (int) when known; ExperimentRun ``refs.protocol``, its arm is
  ``condition``; result Reports have ``title == "experiment"`` and ``refs.protocol``, with
  ``data.rows`` (list of objects) for the result table. Approval is a HumanAction
  ``approve_protocol`` with ``refs.protocol`` (operate.approve).
- Improvement: Reports with ``title == "improvement"``, ``data.batch``, ``data.attribution``
  (category -> count) and ``data.items`` (each ``key``, ``title``, ``kind``, ``description``);
  attribution is also counted directly from Gap records (``Gap.attribution``).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from kta.api import operate, proofview
from kta.api.common import Api, body, latest
from kta.core.ids import record_id
from kta.orchestrator.proc_common import get_state
from kta.store.assurance import effective
from kta.store.sqlite import RecordNotFound

POOL_KINDS = ("Pattern", "Template", "Structure", "Reference", "Reconstruction")


def _rows(api: Api, kind: str, limit: int = 2000) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for rec in api.store.query(kind, order="created_at", limit=limit):
        data = rec.to_json_dict()
        # The ID is of the stored form; the body shown is the current shape (D440).
        out.append((record_id(data), {**data, "body": rec.body}))
    return out


def _env(api: Api, rid: str, data: dict[str, Any]) -> dict[str, Any]:
    status, assurance = api.store.status(rid)
    if data.get("kind") in ("Template", "Family", "Chain"):
        assurance = effective(api.store, rid)
    return {
        "id": rid,
        "created_at": data["created_at"],
        "origin": data["origin"],
        "actor": data["actor"]["name"],
        "status": status,
        "assurance": assurance,
    }


def _template(b: dict[str, Any]) -> dict[str, Any]:
    keys = ("name", "version", "level_from", "level_to", "change", "refines", "produces")
    return {
        **{k: b.get(k) for k in keys},
        "summary": b.get("summary", ""),
        "requires": b.get("requires", []),
        "probed_on": b.get("probed_on", []),
        "scope": (b.get("scope") or {}).get("predicate", "True"),
        "variables": len(b.get("variables", [])),
        "constraints": len(b.get("constraints", [])),
        "obligations": len(b.get("obligations", [])),
    }


def _confirmations(api: Api) -> dict[str, str]:
    """Record ID -> HumanAction ID of its confirm_pattern / confirm_structure."""
    out: dict[str, str] = {}
    for aid, data in _rows(api, "HumanAction"):
        verb = operate.parse_action(data["body"]["action"])["verb"]
        if verb in ("confirm_pattern", "confirm_structure"):
            for v in data["refs"].values():
                if isinstance(v, str):
                    out[v] = aid
    return out


def pool(api: Api, snapshot: str | None) -> dict[str, Any]:
    members = set(body(api, snapshot)["members"]) if snapshot else None
    confirmed = _confirmations(api)

    def keep(rid: str) -> bool:
        return members is None or rid in members

    out: dict[str, Any] = {"snapshot": snapshot}
    out["patterns"] = [
        {
            **_env(api, rid, d),
            **{k: d["body"].get(k) for k in ("name", "level", "category", "description")},
            "examples": len(d["body"].get("examples", [])),
            "confirmed_by": confirmed.get(rid),
        }
        for rid, d in _rows(api, "Pattern")
        if keep(rid)
    ]
    used = _usage(api)
    names = _names(api, "Source")
    out["templates"] = [
        {
            **_env(api, rid, d),
            **_template(d["body"]),
            "sources": [names.get(s, s) for s in _list(d["refs"].get("sources"))],
            "used_by": used.get(rid, []),
            "proof": proofview.template_proof(api, rid),
        }
        for rid, d in _rows(api, "Template")
        if keep(rid)
    ]
    out["structures"] = [
        {**_env(api, rid, d), **d["body"], "confirmed_by": confirmed.get(rid)}
        for rid, d in _rows(api, "Structure")
        if keep(rid)
    ]
    out["gaps"] = [
        {**_env(api, rid, d), **d["body"], "refs": d["refs"]} for rid, d in _rows(api, "Gap")
    ]
    out["references"] = _references(api, keep)
    out["counts"] = {k: len(v) for k, v in out.items() if isinstance(v, list)}
    out["other_patterns"] = sum(1 for p in out["patterns"] if p["category"] == "other")
    out["assurance"] = proofview.distribution(api, [t["id"] for t in out["templates"]])
    return out


def _list(v: Any) -> list[str]:
    return [v] if isinstance(v, str) else [x for x in v or [] if isinstance(x, str)]


def _names(api: Api, kind: str) -> dict[str, str]:
    return {rid: str(d["body"].get("name", rid)) for rid, d in _rows(api, kind)}


def _usage(api: Api) -> dict[str, list[dict[str, str]]]:
    """Template ID -> the chains that name it, by owner: a Reference whose reconstruction
    matched it (analysis) or a Family built with it (a kernel). One entry per owner name."""
    owner: dict[str, dict[str, str]] = {}
    refs = _names(api, "Reference")
    for _, d in _rows(api, "Reconstruction"):
        ref, ch = d["refs"].get("reference"), d["refs"].get("chain")
        if isinstance(ref, str) and isinstance(ch, str):
            owner[ch] = {"kind": "reference", "id": ref, "name": refs.get(ref, ref)}
    for rid, d in _rows(api, "Family"):
        ch = d["refs"].get("chain")
        if isinstance(ch, str):
            owner[ch] = {"kind": "family", "id": rid, "name": str(d["body"]["name"])}
    out: dict[str, list[dict[str, str]]] = {}
    for cid, d in _rows(api, "Chain"):
        o = owner.get(cid)
        if o is None:
            continue
        for tid in _list(d["refs"].get("templates")):
            rows = out.setdefault(tid, [])
            if not any(r["kind"] == o["kind"] and r["name"] == o["name"] for r in rows):
                rows.append(o)
    return out


def template_page(api: Api, tid: str) -> dict[str, Any]:
    """GET /templates/{id}: one Template for its own page (WP25): the body, where it came
    from, which references and families use it, its versions, relations and checks."""
    rec = api.store.get(tid)
    if rec.kind != "Template":
        raise RecordNotFound(f"{tid} is a {rec.kind}, not a Template")
    data = rec.to_json_dict()
    b = rec.body
    rows = _rows(api, "Template")
    versions = [
        {"id": rid, "version": d["body"]["version"], "created_at": d["created_at"]}
        for rid, d in rows
        if d["body"]["name"] == b["name"]
    ]
    newest: dict[str, tuple[int, str]] = {}
    for rid, d in rows:
        n, v = d["body"]["name"], d["body"]["version"]
        if n not in newest or v > newest[n][0]:
            newest[n] = (v, rid)
    sources = []
    for sid in _list(data["refs"].get("sources")):
        sb = body(api, sid)
        keys = ("name", "locator", "revision", "type")
        sources.append({"id": sid, **{k: sb.get(k) for k in keys}})
    relations = [
        {**r, "id": newest.get(r["target"].partition("@")[0], (0, None))[1]}
        for r in b.get("relations", [])
    ]
    return {
        **_env(api, tid, data),
        "body": b,
        "sources": sources,
        "used_by": _usage(api).get(tid, []),
        "versions": sorted(versions, key=lambda x: x["version"]),
        "relations": relations,
        "proof": proofview.template_proof(api, tid),
    }


def _references(api: Api, keep: Any) -> list[dict[str, Any]]:
    recon: dict[str, list[str]] = {}
    for rid, d in _rows(api, "Reconstruction"):
        ref = d["refs"].get("reference")
        if isinstance(ref, str):
            recon.setdefault(ref, []).append(rid)
    rows: list[dict[str, Any]] = []
    for rid, d in _rows(api, "Reference"):
        if not keep(rid):
            continue
        last = latest(api, recon.get(rid, []))
        rb = body(api, last) if last else None
        rows.append(
            {
                **_env(api, rid, d),
                "name": d["body"]["name"],
                "runnable_on": d["body"].get("runnable_on", []),
                "reconstruction": last,
                "matched": rb["matched"] if rb else None,
                "total": rb["total"] if rb else None,
                "unmatched": len(rb["unmatched_transactions"]) if rb else None,
            }
        )
    return rows


def _actions_on(api: Api, verb: str, ref: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for aid, d in _rows(api, "HumanAction"):
        if operate.parse_action(d["body"]["action"])["verb"] != verb:
            continue
        target = d["refs"].get(ref)
        if isinstance(target, str):
            out.setdefault(target, []).append(
                {"id": aid, "at": d["created_at"], "by": d["actor"]["name"], **d["body"]}
            )
    return out


def experiments(api: Api) -> dict[str, Any]:
    approvals = _actions_on(api, "approve_protocol", "protocol")
    runs = _rows(api, "ExperimentRun")
    reports = [(rid, d) for rid, d in _rows(api, "Report") if d["body"]["title"] == "experiment"]
    protocols: list[dict[str, Any]] = []
    for pid, d in _rows(api, "Protocol"):
        b = d["body"]
        if b.get("type") != "experiment":
            continue
        mine = [(rid, r) for rid, r in runs if r["refs"].get("protocol") == pid]
        params = b.get("params", {})
        arms = list(params.get("arms", [])) or sorted({r["body"]["condition"] for _, r in mine})
        count = Counter(r["body"]["condition"] for _, r in mine)
        cases = params.get("cases")
        per_arm = params.get("runs_per_arm")
        if isinstance(per_arm, int) and isinstance(cases, list):
            per_arm *= max(1, len(cases))  # WP11 D403: one run per arm x case x repetition
        protocols.append(
            {
                **_env(api, pid, d),
                **{k: b.get(k) for k in ("name", "version", "description")},
                "params": params,
                "approved": approvals.get(pid, []),
                "progress": _experiment_state(api, pid),
                "arms": [{"name": a, "runs": count.get(a, 0), "planned": per_arm} for a in arms],
                "runs": [{**_env(api, rid, r), **r["body"], "refs": r["refs"]} for rid, r in mine],
                "results": [
                    {
                        **_env(api, rid, r),
                        "summary": r["body"]["summary"],
                        "data": r["body"]["data"],
                    }
                    for rid, r in reports
                    if r["refs"].get("protocol") == pid
                ],
            }
        )
    return {"protocols": protocols}


def _experiment_state(api: Api, pid: str) -> dict[str, Any]:
    """The experiment process's state of an approved protocol (WP11 D403; as
    ``kta experiment status``): status, runs started, runs finished, last error."""
    st = get_state(api.ops, f"experiment:{pid}")
    slots: dict[str, Any] = st.get("slots", {})
    return {
        "status": st.get("status"),
        "started": len(slots),
        "finished": sum(1 for s in slots.values() if s.get("run")),
        "report": st.get("report"),
        "error": get_state(api.ops, f"experiment:{pid}:error").get("error"),
    }


def improvements(api: Api) -> dict[str, Any]:
    approvals = _actions_on(api, "approve_improvement", "improvement")
    rejections = _actions_on(api, "reject_improvement", "improvement")
    gaps = [d["body"] for _, d in _rows(api, "Gap")]
    by_day: dict[str, Counter[str]] = {}
    for _, d in _rows(api, "Gap"):
        by_day.setdefault(d["created_at"][:10], Counter())[d["body"]["attribution"]] += 1
    reports = []
    for rid, d in _rows(api, "Report"):
        b = d["body"]
        if b["title"] != "improvement":
            continue
        data = b.get("data", {})
        reports.append(
            {
                **_env(api, rid, d),
                "summary": b.get("summary", ""),
                "batch": data.get("batch"),
                "attribution": data.get("attribution", {}),
                "items": data.get("items", []),
                "approved": approvals.get(rid, []),
                "rejected": rejections.get(rid, []),
                "system_items": sum(1 for i in data.get("items", []) if i.get("kind") == "system"),
            }
        )
    return {
        "attribution": dict(Counter(g["attribution"] for g in gaps)),
        "missing": dict(Counter(g["missing"] for g in gaps)),
        "attribution_by_day": [{"day": k, **v} for k, v in sorted(by_day.items())],
        "reports": reports,
    }
