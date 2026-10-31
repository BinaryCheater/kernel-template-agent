"""Verification inside the solve loop (WP24, paper §4.2–§4.3; D638, D639).

- ``candidate_gate``: every solver candidate is checked once (its assignment in C, its
  context in Γ, under the chain bindings) before it is built or measured. A failing
  candidate is dropped and recorded; a passing one is marked ``correctness`` or
  ``exploration`` by the family's composition level.
- ``relaxed_prediction``: the model optimum when each chain Template whose recorded
  weakest-precondition search says "stronger" is replaced by its relaxed form. The gap
  decomposition reads it as ``relaxed_pred_us`` (Δpre = P_opt − P_relaxed).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from kta.checks.proof import candidate as cand
from kta.checks.proof import wp
from kta.checks.proof.prove import find_template
from kta.core.errors import SchemaError
from kta.core.schemas.evaluation import Candidate as CandidateBody
from kta.core.schemas.evaluation import Case
from kta.core.schemas.knowledge import Template
from kta.formulation import assemble
from kta.knowledge import FamilyPackage
from kta.models import UnitModel
from kta.orchestrator.pipeline import Services, bucket_shape
from kta.solve import solve
from kta.store.sqlite import SqliteStore


def _sqlite(svc: Services) -> SqliteStore:
    if not isinstance(svc.store, SqliteStore):
        raise SchemaError("the proof records need the SQLite store")
    return svc.store


def candidate_gate(
    svc: Services,
    family_id: str,
    fp: FamilyPackage,
    templates: Mapping[str, Template],
    hw: Mapping[str, Any],
    bucket: Mapping[str, Any],
    candidate_ids: list[str],
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """(kept candidate IDs, candidate -> ProofResult ID, candidate -> claims)."""
    store = _sqlite(svc)
    steps = [(it, templates[it.template]) for it in fp.chain.items]
    shape = bucket_shape(bucket)
    kept: list[str] = []
    results: dict[str, str] = {}
    claims: dict[str, str] = {}
    for cid in candidate_ids:
        a = CandidateBody.model_validate(store.get(cid).body).assignment
        chk = cand.check_candidate(steps, a, shape, hw)
        rid = cand.record_candidate(store, cid, family_id, chk, a, shape, svc.ctx.clock.now())
        results[cid] = rid
        claims[cid] = str(store.get(rid).body["data"]["claims"])
        if chk.ok:
            kept.append(cid)
    return kept, results, claims


def relaxed_templates(
    store: SqliteStore, fp: FamilyPackage, templates: Mapping[str, Template]
) -> tuple[dict[str, Template], dict[str, str]]:
    """Templates with recorded "stronger" preconditions relaxed, and per Template what the
    weakest-precondition record says (exact, stronger, unsound, or absent)."""
    out = dict(templates)
    status: dict[str, str] = {}
    for it in fp.chain.items:
        tid = find_template(store, it.template)
        w = wp.latest(store, tid)
        if w is None:
            status[it.template] = "not_assessed"
            continue
        data = w["data"]
        status[it.template] = str(data["relation"])
        if (
            data["relation"] == "stronger"
            and data.get("weaker")
            and data.get("verified") in ("bounded", "proved")
        ):
            out[it.template] = wp.relaxed(
                templates[it.template], str(w["obligation"]), str(data["weaker"])
            )
    return out, status


def relaxed_prediction(
    svc: Services,
    fp: FamilyPackage,
    templates: Mapping[str, Template],
    case: Case,
    bucket: Mapping[str, Any],
    model: UnitModel,
    hw: Any,
    top_pred_us: float | None,
    shortlist: int,
    time_limit_s: float,
    rng: random.Random,
) -> tuple[float | None, str]:
    """(P_relaxed, note). None when no step of the chain has been assessed."""
    store = _sqlite(svc)
    relaxed, status = relaxed_templates(store, fp, templates)
    assessed = {k: v for k, v in status.items() if v != "not_assessed"}
    if not assessed:
        return None, "no weakest-precondition search recorded for any chain step"
    note = "; ".join(f"{k}: {v}" for k, v in sorted(status.items()))
    if relaxed == dict(templates):
        return top_pred_us, note
    f = assemble(fp, fp.chain, relaxed, case, hw.hw)
    res = solve(
        f, case, dict(bucket), model, hw, shortlist=shortlist, time_limit_s=time_limit_s, rng=rng
    )
    if not res.candidates:
        return None, note + "; the relaxed formulation has no candidate"
    return res.candidates[0].prediction.total, note
