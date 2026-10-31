"""Event-order models for storage and transfers.

- ``event_order``: a value written at ``first_write`` must survive until ``last_read``:
  under Pre, the next destructive write into its buffer comes after the last read. Event
  times are distinct integers (a total order of events); ``last_read``, ``next_write`` are
  the Template's p.* names. Replay: execute the three events in time order on one buffer.
- ``resident_capacity``: the bytes resident at once (copies x size) fit the capacity,
  under Γ and C. Replay: recompute the product with the interpreter.
- ``async_readiness``: every read of a destination follows the completion of every piece
  it reads, on every interleaving of W issuers. What the wait at each ``complete_point``
  covers ("own" pieces or "all" pieces) is declared in the outline; a value declared
  ``null`` (decided by a compiler) is not modelled.
"""

from __future__ import annotations

import itertools
import time
from typing import Any

import z3

from .. import interp
from ..encode import Unsupported
from ..smt import enumerated, run
from ..types import Draft
from .base import ModelInput


def _pre(m: ModelInput) -> Any:
    tr = m.space.tr
    return z3.And(*[tr.bool(p.expr) for p in m.space.template.preconditions] or [z3.BoolVal(True)])


def event_order(m: ModelInput) -> Draft:
    sp = m.space
    lr, nw = sp.env[m.var("last_read")].term, sp.env[m.var("next_write")].term
    fw = z3.Int("first_write")
    d = m.draft("proved", f"all event times; {sp.describe()}", ("gamma", "assumes", "pre"))
    wf = z3.And(fw < lr, fw < nw, z3.Distinct(fw, lr, nw))
    q = run(
        z3.And(sp.domain, sp.gamma(), sp.assumptions(), _pre(m), wf, nw < lr),
        {**sp.decls(), "first_write": fw},
        m.timeout_ms,
    )
    d.add(q.record())
    if q.result == "unsat":
        return d
    if q.result == "unknown":
        d.outcome, d.reason = "unresolved", f"solver unknown: {q.reason}"
        return d
    w = sp.decode(q.model)
    events = sorted(
        [
            (w["first_write"], "write", "v"),
            (w[m.var("next_write")], "write", "w"),
            (w[m.var("last_read")], "read", None),
        ]
    )
    buf: str | None = None
    seen = None
    for _, kind, val in events:
        if kind == "write":
            buf = val
        else:
            seen = buf
    pre_ok = all(interp.holds(p.expr, w) is True for p in sp.template.preconditions)
    d.replay = {
        "events": [list(e) for e in events],
        "read_sees": seen,
        "pre_holds": pre_ok,
        "agrees": seen == "v",
    }
    if seen != "v" and pre_ok:
        d.outcome, d.counterexample = "counterexample", w
        d.reason = "the next write lands before the last read although Pre holds"
    else:
        d.outcome, d.reason = "unresolved", "solver witness not confirmed by the replay"
    return d


def resident_capacity(m: ModelInput) -> Draft:
    sp = m.space
    size, copies, cap = (sp.env[m.var(k)].term for k in ("size", "copies", "capacity"))
    d = m.draft("proved", f"all {sp.describe()} satisfying C", ("gamma", "assumes", "C"))
    q = run(
        z3.And(sp.domain, sp.gamma(), sp.assumptions(), sp.constraints(), copies * size > cap),
        sp.decls(),
        m.timeout_ms,
    )
    d.add(q.record())
    if q.result == "unsat":
        return d
    if q.result == "unknown":
        d.outcome, d.reason = "unresolved", f"solver unknown: {q.reason}"
        return d
    w = sp.decode(q.model)
    over = interp.holds(f"{m.var('copies')} * {m.var('size')} > {m.var('capacity')}", w)
    d.replay = {"agrees": over is True}
    d.outcome = "counterexample" if over else "unresolved"
    d.counterexample = w if over else None
    return d


def _readiness_witness(warps: int, waits: str) -> list[tuple[str, int]] | None:
    """An interleaving where a read precedes a completion it needs, by enumeration of the
    per-warp programs ISSUE(w) < COMPLETE(w) and ISSUE(w) < WAIT(w) < READ(w)."""
    events = [(k, w) for w in range(warps) for k in ("ISSUE", "COMPLETE", "WAIT", "READ")]
    for order in itertools.permutations(events):
        pos = {e: i for i, e in enumerate(order)}
        ok = all(
            pos[("ISSUE", w)] < pos[("COMPLETE", w)]
            and pos[("ISSUE", w)] < pos[("WAIT", w)] < pos[("READ", w)]
            for w in range(warps)
        )
        waited = [j for j in range(warps)] if waits == "all" else None
        for w in range(warps):
            deps = waited if waited is not None else [w]
            ok = ok and all(pos[("COMPLETE", j)] < pos[("WAIT", w)] for j in deps)
        if not ok:
            continue
        if any(pos[("READ", w)] < pos[("COMPLETE", j)] for w in range(warps) for j in range(warps)):
            return list(order)
    return None


def _replay_readiness(order: list[tuple[str, int]], warps: int) -> dict[str, Any]:
    done: set[int] = set()
    stale: list[dict[str, Any]] = []
    for kind, w in order:
        if kind == "COMPLETE":
            done.add(w)
        elif kind == "READ":
            missing = [j for j in range(warps) if j not in done]
            if missing:
                stale.append({"reader": w, "pieces_not_complete": missing})
    return {"agrees": not stale, "stale_reads": stale, "order": [f"{k}{w}" for k, w in order]}


def async_readiness(m: ModelInput) -> Draft:
    t0 = time.perf_counter()
    cv = m.var("complete_point")
    waits: dict[str, str | None] = dict(m.param("waits") or {})
    warps = int(m.param("warps", 2))
    d = m.draft("bounded", "", ("program", "X"))
    unmodelled = [v for v in m.menu(cv) if waits.get(str(v)) not in ("own", "all")]
    for v in m.menu(cv):
        how = waits.get(str(v))
        if how not in ("own", "all"):
            continue
        order = _readiness_witness(warps, how)
        if order is not None:
            rep = _replay_readiness(order, warps)
            if not rep["agrees"]:
                d.outcome, d.counterexample, d.replay = (
                    "counterexample",
                    {cv: v, "waits": how, "warps": warps},
                    rep,
                )
                d.reason = f"{cv}={v}: a warp reads another warp's piece before it completes"
                break
    d.add(
        enumerated(
            f"async_readiness {warps} {sorted(waits.items(), key=str)}", time.perf_counter() - t0
        )
    )
    modelled = [v for v in m.menu(cv) if v not in unmodelled]
    d.domain = f"every interleaving of {warps} issuing warps, one piece each; {cv} in {modelled}"
    if d.outcome != "counterexample" and unmodelled:
        d.outcome = "unresolved"
        d.reason = f"{cv} in {unmodelled}: completion decided outside the Template (not modelled)"
    if not m.menu(cv):
        raise Unsupported("empty complete_point menu")
    return d
