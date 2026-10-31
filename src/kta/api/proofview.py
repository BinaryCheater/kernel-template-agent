"""Proof views (WP24, Part G): assurance distribution, per-obligation status, and the split
of a result's evidence into what is proved (Template level) and what is tested (code level).

Every obligation of a Template is listed: checked ones with outcome, stated domain and the
run that checked them, the rest as unchecked or never checked. Nothing is hidden because
it was not checked.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from kta.api.common import Api, body, incoming
from kta.checks.proof.prove import find_template
from kta.core.errors import SchemaError
from kta.store.assurance import effective, proof_summary

LEVELS = ("declared", "tested", "bounded", "proved")
NOT_COVERED = (
    "whether the agent-written generator code implements the Template's abstract program "
    "(guarded by code markers, expected SASS and sanitizers; IR-level extraction later)",
    "floating-point rounding (proofs treat products as symbolic terms; numeric tests cover it)",
    "schedules beyond the bounded event model (warps and phases named in each domain)",
)


def template_proof(api: Api, tid: str) -> dict[str, Any]:
    """The latest whole-Template summary and the status of every obligation of Σ."""
    t = body(api, tid)
    s = proof_summary(api.store, tid)
    checked = {(c["condition"], c["obligation"]): c for c in (s["checked"] if s else [])}
    unchecked = {u["obligation"]: u for u in (s["unchecked"] if s else [])}
    last = api.store.get_json(s["id"])["created_at"] if s else None
    rows: list[dict[str, Any]] = []
    for ob in t.get("obligations", []):
        c = checked.get(("K2", ob["name"]))
        if c is not None:
            rows.append(
                {
                    "name": ob["name"],
                    "method": ob["method"],
                    "status": c["outcome"],
                    "domain": c["domain"],
                    "result": c["result"],
                    "reused": c.get("reused", False),
                    "reason": c.get("reason", ""),
                }
            )
        elif ob["name"] in unchecked:
            rows.append(
                {
                    "name": ob["name"],
                    "method": ob["method"],
                    "status": "unchecked",
                    "domain": "",
                    "result": None,
                    "reused": False,
                    "reason": unchecked[ob["name"]]["reason"],
                }
            )
        else:
            rows.append(
                {
                    "name": ob["name"],
                    "method": ob["method"],
                    "status": "never checked",
                    "domain": "",
                    "result": None,
                    "reused": False,
                    "reason": "no checker run yet" if s is None else "",
                }
            )
    contract = [
        {
            "condition": k[0],
            "name": k[1],
            "status": c["outcome"],
            "domain": c["domain"],
            "result": c["result"],
            "reason": c.get("reason", ""),
        }
        for k, c in checked.items()
        if k[0] != "K2"
    ]
    return {
        "summary": s["id"] if s else None,
        "level": s.get("level") if s else None,
        "live": s["live"] if s else None,
        "last_run": last,
        "checker": s.get("checker") if s else None,
        "solver": s.get("solver") if s else None,
        "obligations": rows,
        "contract": contract,
        "gaps": s.get("gaps", []) if s else [],
    }


def distribution(api: Api, template_ids: list[str]) -> dict[str, int]:
    got = Counter(effective(api.store, t) for t in template_ids)
    return {lv: got.get(lv, 0) for lv in LEVELS}


def _step_rows(api: Api, fam: str, chain: str | None) -> list[dict[str, Any]]:
    s = proof_summary(api.store, fam)
    parts = {}
    if s:
        for sid in api.store.get_json(s["id"])["refs"].get("steps", []):
            b = body(api, sid)
            parts[b["part"].removeprefix("step:")] = {
                "summary": sid,
                "level": b["level"],
                "gaps": len(b["gaps"]),
            }
    rows: list[dict[str, Any]] = []
    for it in body(api, chain)["items"] if chain else []:
        try:
            tid: str | None = find_template(api.store, it["template"])
        except SchemaError:  # a chain naming a Template the store lacks is shown as declared
            tid = None
        rows.append(
            {
                "step": it["key"],
                "template": it["template"],
                "assurance": effective(api.store, tid) if tid else "declared",
                "in_chain": parts.get(it["key"]),
            }
        )
    return rows


def result_assurance(
    api: Api,
    fam: str | None,
    cand: str | None,
    numeric: str | None,
    sass: str | None,
    build: str | None,
) -> dict[str, Any]:
    """Which parts of a result are proved (Template level) and which are tested (code)."""
    fs = proof_summary(api.store, fam) if fam else None
    chain = None
    if fam:
        c = api.store.get_json(fam)["refs"].get("chain")
        chain = c if isinstance(c, str) else None
    cand_check = None
    if cand:
        found = [r for r in incoming(api, cand, "subject", "ProofResult")]
        if found:
            b = body(api, found[-1])
            cand_check = {
                "result": found[-1],
                "outcome": b["outcome"],
                "claims": b["data"].get("claims"),
                "reason": b["reason"],
            }
    checks = incoming(api, build, "build", "Check") if build else []
    sanitizer = [
        c for c in checks if api.store.get_json(c)["actor"]["name"].startswith("check_sanitizer")
    ]
    code = [
        {
            "check": "numeric differential tests (edge and hidden shapes)",
            "record": numeric,
            "status": _passed(api, numeric),
        },
        {"check": "SASS intent", "record": sass, "status": _passed(api, sass)},
        {
            "check": "compute-sanitizer",
            "record": sanitizer[0] if sanitizer else None,
            "status": _passed(api, sanitizer[0] if sanitizer else None),
        },
        {
            "check": "canaries (planted faults the checkers must catch)",
            "record": None,
            "status": "not linked to this result",
        },
    ]
    return {
        "family": {
            "summary": fs["id"] if fs else None,
            "level": fs.get("level") if fs and fs["live"] else None,
            "gaps": fs.get("gaps", []) if fs else ["no composition check recorded"],
        },
        "steps": _step_rows(api, fam, chain) if fam else [],
        "candidate": cand_check,
        "code_level": code,
        "not_covered": list(NOT_COVERED),
    }


def _passed(api: Api, rid: str | None) -> str:
    if rid is None:
        return "missing"
    return "tested" if body(api, rid).get("passed") else "failed"


def proofs_of(api: Api, subject: str) -> dict[str, Any]:
    """GET /proofs/{id}: every ProofSummary and ProofResult about a record, newest first."""
    summaries = incoming(api, subject, "subject", "ProofSummary")
    results = incoming(api, subject, "subject", "ProofResult")

    def row(rid: str) -> dict[str, Any]:
        d = api.store.get_json(rid)
        return {
            "id": rid,
            "created_at": d["created_at"],
            "status": api.store.status(rid)[0],
            **d["body"],
        }

    return {
        "subject": subject,
        "assurance": effective(api.store, subject),
        "summaries": sorted(
            (row(s) for s in summaries), key=lambda r: r["created_at"], reverse=True
        ),
        "results": sorted((row(r) for r in results), key=lambda r: r["created_at"], reverse=True),
    }
