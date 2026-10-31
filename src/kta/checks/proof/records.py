"""Proof records, their premises, reuse and invalidation (paper §4.4; D632, D633).

Premises are content hashes of the named groups a result uses (Γ, X, C, each Pre, the
program, the outline's assumptions, the model binding and the fixed bindings) plus the
solver and checker versions. Equal premises -> the same ``premise_key`` -> the earlier
certificate may be reused (written again for the new subject with ``refs.reused``).

The dependency graph is the records' refs: a chain summary references its step summaries,
a summary its results, a result its outline. ``invalidate`` marks a node stale and every
dependant through ``kta.store.stale.propagate``; ``drift`` invalidates every summary or
result whose checker, solver or outline is no longer current. A stale record is never
reused and never sets a level (kta.store.assurance).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from importlib import resources
from typing import Any

from kta.core.ids import record_id
from kta.core.schemas import Actor, Record, RefValue
from kta.core.schemas.knowledge import Template
from kta.core.schemas.process import StatusChange
from kta.core.schemas.proof import ObligationModel, ProofOutline, ProofResult, ProofSummary
from kta.store import stale
from kta.store.sqlite import SqliteStore

from .smt import solver_version
from .types import Draft

DEAD = frozenset({"stale", "rejected", "deprecated", "suspect"})
SYSTEM = Actor(type="system", name="proof.invalidate")


def h(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def checker_version() -> str:
    """Content hash of the trusted base in this repository: the condition builder, the
    models, the replay interpreter and the expression evaluator it relies on."""
    digest = hashlib.sha256()
    pkg = resources.files("kta.checks.proof")
    files = sorted(
        [p for p in pkg.iterdir() if p.name.endswith(".py")]
        + [p for p in pkg.joinpath("models").iterdir() if p.name.endswith(".py")],
        key=lambda p: p.name,
    )
    for p in [*files, resources.files("kta.core").joinpath("expr.py")]:
        digest.update(p.name.encode())
        digest.update(p.read_bytes())
    return f"proof@{digest.hexdigest()[:12]}"


def groups(
    t: Template, outline: ProofOutline | None, restrict: Mapping[str, Any]
) -> dict[str, str]:
    g = {
        "gamma": h(t.scope.predicate),
        "X": h([v.model_dump(mode="json") for v in t.variables]),
        "C": h([c.expr for c in t.constraints]),
        "pre": h([p.expr for p in t.preconditions]),
        "program": h(t.program.description),
        "assumes": h([a.model_dump(mode="json") for a in (outline.assumes if outline else [])]),
        "restrict": h(dict(restrict)),
    }
    g.update({f"pre:{p.id}": h(p.expr) for p in t.preconditions})
    return g


def premises(
    d: Draft,
    g: Mapping[str, str],
    t: Template,
    binding: ObligationModel | None,
    versions: Mapping[str, str],
) -> dict[str, str]:
    out = {k: g[k] for k in d.groups if k in g}
    out["restrict"] = g["restrict"]
    out["condition"] = f"{d.condition}:{d.obligation}"
    if d.condition == "K2":
        ob = next((o for o in t.obligations if o.name == d.obligation), None)
        out["obligation"] = h(ob.model_dump(mode="json") if ob else d.obligation)
        out["model"] = h(binding.model_dump(mode="json") if binding else None)
    out.update(versions)
    return out


def versions() -> dict[str, str]:
    return {"checker": checker_version(), "solver": solver_version()}


def alive(store: SqliteStore, rid: str) -> bool:
    return store.status(rid)[0] not in DEAD


def reusable(store: SqliteStore, key: str) -> Record | None:
    """The newest live result with this premise key and a positive outcome."""
    for rec in store.query(
        "ProofResult", where={"premise_key": key}, order="-created_at", limit=20
    ):
        if rec.body["outcome"] in ("proved", "bounded") and alive(store, record_id(rec)):
            return rec
    return None


def result_record(
    d: Draft,
    subject: str,
    scope: str,
    prem: dict[str, str],
    actor: Actor,
    now: datetime,
    outline_id: str | None,
    reused: str | None = None,
) -> Record:
    body = ProofResult(
        subject=subject,
        scope=scope,  # type: ignore[arg-type]
        condition=d.condition,  # type: ignore[arg-type]
        obligation=d.obligation,
        outcome=d.outcome,  # type: ignore[arg-type]
        domain=d.domain,
        premises=prem,
        premise_key=h(prem),
        queries=d.queries,
        seconds=round(d.seconds, 6),
        counterexample=_jsonable(d.counterexample),
        replay=_jsonable(d.replay),
        reason=d.reason,
        data=_jsonable({**d.data, **({"reused": True} if reused else {})}) or {},
    )
    refs: dict[str, RefValue] = {"subject": subject}
    if outline_id:
        refs["outline"] = outline_id
    if reused:
        refs["reused"] = reused
    return Record.new("ProofResult", body, actor, now, refs=refs)


def summary_record(
    subject: str,
    scope: str,
    level: str | None,
    results: list[tuple[str, Record]],
    unchecked: list[dict[str, Any]],
    gaps: list[str],
    actor: Actor,
    now: datetime,
    steps: Iterable[str] = (),
    extra: Mapping[str, Any] | None = None,
    part: str = "",
) -> Record:
    checked = [
        {
            "condition": r.body["condition"],
            "obligation": r.body["obligation"],
            "outcome": r.body["outcome"],
            "domain": r.body["domain"],
            "result": rid,
            "reused": bool(r.refs.get("reused")),
            **({"reason": r.body["reason"]} if r.body["reason"] else {}),
        }
        for rid, r in results
    ]
    v = versions()
    body = ProofSummary(
        subject=subject,
        scope=scope,  # type: ignore[arg-type]
        part=part,
        level=level,  # type: ignore[arg-type]
        checked=checked + [dict(x) for x in (extra or {}).get("checked", [])],
        unchecked=unchecked,
        gaps=gaps,
        premise_key=h(sorted(r.body["premise_key"] for _, r in results)),
        checker=v["checker"],
        solver=v["solver"],
    )
    refs: dict[str, RefValue] = {"subject": subject, "results": [rid for rid, _ in results]}
    if steps:
        refs["steps"] = list(steps)
    return Record.new("ProofSummary", body, actor, now, refs=refs)


def _jsonable(x: Any) -> Any:
    return None if x is None else json.loads(json.dumps(x, default=str))


def latest_summary(store: SqliteStore, subject: str) -> Record | None:
    """The newest live summary of ``subject`` (None if there is none or it went stale)."""
    for rec in store.query(
        "ProofSummary", where={"subject": subject, "part": ""}, order="-created_at", limit=5
    ):
        return rec if alive(store, record_id(rec)) else None
    return None


def invalidate(store: SqliteStore, node: str, reason: str, now: datetime) -> list[str]:
    """Mark ``node`` stale (unless already dead) and every dependant of it."""
    marked: list[str] = []
    if alive(store, node):
        store.put(
            Record.new(
                "StatusChange",
                StatusChange(subject=node, to_status="stale", reason=reason),
                SYSTEM,
                now,
            )
        )
        marked.append(node)
    return marked + stale.propagate(store, node, reason, now)


def drift(store: SqliteStore, now: datetime, current: Mapping[str, str] | None = None) -> list[str]:
    """Invalidate live results whose checker or solver version changed, and those built on
    an outline that a newer version of the same outline replaced."""
    cur = dict(current or versions())
    newest: dict[str, tuple[int, str]] = {}
    for o in store.query("ProofOutline", limit=10000):
        key = o.body["template"]
        if key not in newest or o.body["version"] > newest[key][0]:
            newest[key] = (o.body["version"], record_id(o))
    marked: list[str] = []
    for r in store.query("ProofResult", limit=10000):
        rid = record_id(r)
        if rid in marked or not alive(store, rid):
            continue
        prem = r.body["premises"]
        why = [k for k in ("checker", "solver") if prem.get(k) and prem[k] != cur[k]]
        outline = r.refs.get("outline")
        if isinstance(outline, str):
            o = store.get(outline).body
            if newest.get(o["template"], (0, outline))[1] != outline:
                why.append("outline")
        if why:
            marked += invalidate(store, rid, f"premise changed: {', '.join(why)}", now)
    return marked
