"""``slot_rotation``: readiness and lifetime of generations rotating through S slots.

The bounded event model of paper §4.2 / Appendix A: W warps, KT reduction phases; per
phase a warp issues the load of generation p + P (after a prologue of P loads), crosses a
leading barrier, consumes slot p mod S and, with beta, crosses a trailing barrier. Event
times are integers constrained only by program order and barrier semantics (a barrier
instance is one point every warp crosses), so a model of the negated obligation is an
admissible interleaving. Every such interleaving is replayed by an interpreter that
re-derives admissibility from the program text and executes formal tokens A[w,k] B[k,j].

Reimplemented after the read-only reference
/Users/dingyi/Projects/agentic-codesign/backend/ventus_compiled/proof_templates/upstream/smem_verify.py
(``schedule_obligations``, ``replay_schedule``); not imported.

KT is max(phases, S + P + 2) for each point so that a slot is reused within the model.
"""

from __future__ import annotations

import itertools
import time
from collections import Counter
from typing import Any

import z3

from ..smt import run
from ..types import Draft
from .base import ModelInput

Event = tuple[Any, ...]
TN = 2  # B columns in the event model's block


def program(
    s: int, p: int, beta: int, warps: int, phases: int, leading: bool = True
) -> list[list[Event]]:
    progs: list[list[Event]] = []
    for w in range(warps):
        prog: list[Event] = [("LOAD", w, u) for u in range(min(p, phases))]
        for ph in range(phases):
            if p == 0:
                prog.append(("LOAD", w, ph))
            elif ph + p < phases:
                prog.append(("LOAD", w, ph + p))
            if leading:
                prog.append(("BAR", "A", ph))
            prog.append(("COMP", w, ph))
            if beta:
                prog.append(("BAR", "B", ph))
        progs.append(prog)
    _ = s
    return progs


def _name(e: Event) -> str:
    return f"t_{e[0]}_{e[1]}_{e[2]}"


def _elements(warps: int) -> list[tuple[str, int]]:
    return [("A", r) for r in range(warps)] + [("B", c) for c in range(TN)]


def schedule_queries(
    s: int, p: int, beta: int, warps: int, phases: int, timeout_ms: int, leading: bool = True
) -> dict[str, Any]:
    progs = program(s, p, beta, warps, phases, leading)
    ev: dict[Event, Any] = {}
    for prog in progs:
        for e in prog:
            ev.setdefault(e, z3.Int(_name(e)))
    adm: list[Any] = []
    for prog in progs:
        adm += [ev[a] < ev[b] for a, b in itertools.pairwise(prog)]
    adm.append(z3.Distinct(*[v for e, v in ev.items() if e[0] != "BAR"]))
    elems = _elements(warps)
    ready: list[Any] = []
    life: list[Any] = []
    for w in range(warps):
        for ph in range(phases):
            for i, (mat, idx) in enumerate(elems):
                if mat == "A" and idx != w:
                    continue
                owner = i % warps
                d, c = ev[("LOAD", owner, ph)], ev[("COMP", w, ph)]
                ready.append(c < d)
                for u in range(phases):
                    if u != ph and u % s == ph % s:
                        life.append(z3.And(d < ev[("LOAD", owner, u)], ev[("LOAD", owner, u)] < c))
    decls = {_name(e): v for e, v in ev.items()}
    out: dict[str, Any] = {"progs": progs}
    for name, bad in (("readiness", ready), ("lifetime", life)):
        q = run(z3.And(*adm, z3.Or(*bad) if bad else z3.BoolVal(False)), decls, timeout_ms)
        out[name] = q
    return out


def linearize(model: dict[str, int], progs: list[list[Event]]) -> list[Event]:
    seen: dict[str, Event] = {}
    for prog in progs:
        for e in prog:
            seen.setdefault(_name(e), e)
    return [e for _, e in sorted((model[n], e) for n, e in seen.items())]


def replay(
    s: int, warps: int, phases: int, progs: list[list[Event]], order: list[Event]
) -> dict[str, Any]:
    """Execute the interleaving with formal tokens; compare with the reference products."""
    pos = {e: i for i, e in enumerate(order)}
    for prog in progs:
        for a, b in itertools.pairwise(prog):
            if pos[a] >= pos[b]:
                return {"admissible": False, "agrees": False, "reason": f"program order {a} !< {b}"}
    elems = _elements(warps)
    slot: dict[tuple[int, int], tuple[Any, ...]] = {}
    acc = {(w, j): Counter[tuple[Any, ...]]() for w in range(warps) for j in range(TN)}
    trace: list[str] = []
    for e in order:
        if e[0] == "LOAD":
            _, w, u = e
            for i, (mat, idx) in enumerate(elems):
                if i % warps == w:
                    slot[(u % s, i)] = (mat, idx, u) if mat == "A" else (mat, u, idx)
            trace.append(f"LOAD warp{w} gen{u} -> slot{u % s}")
        elif e[0] == "COMP":
            _, w, ph = e
            a = slot.get((ph % s, w))
            for j in range(TN):
                b = slot.get((ph % s, warps + j))
                if a is None or b is None:
                    acc[(w, j)][("uninitialised",)] += 1
                else:
                    acc[(w, j)][(a[1], a[2], b[1], b[2])] += 1
            trace.append(f"COMP warp{w} gen{ph} reads slot{ph % s}: A={a}")
        else:
            trace.append(f"BARRIER {e[1]}{e[2]}")
    diffs: list[dict[str, Any]] = []
    for (w, j), got in acc.items():
        want = Counter({(w, k, k, j): 1 for k in range(phases)})
        for tok in sorted(set(want) | set(got), key=str):
            if want[tok] != got[tok]:
                diffs.append(
                    {"output": [w, j], "term": list(tok), "expected": want[tok], "actual": got[tok]}
                )
    return {
        "admissible": True,
        "agrees": not diffs,
        "differences": len(diffs),
        "sample": diffs[:4],
        "trace": trace,
    }


def safe_point(
    s: int, p: int, beta: int, m: ModelInput, d: Draft
) -> tuple[bool | None, dict[str, Any]]:
    """(safe, evidence) for one (S, P, beta); None when the solver did not decide."""
    warps = int(m.param("warps", 2))
    phases = max(int(m.param("phases", 5)), s + p + 2)
    qs = schedule_queries(s, p, beta, warps, phases, m.timeout_ms, bool(m.param("leading", True)))
    ev: dict[str, Any] = {"phases": phases, "warps": warps}
    for name in m.param("properties", ["readiness", "lifetime"]):
        q = qs[name]
        d.add(q.record())
        if q.result == "unknown":
            return None, {**ev, "unknown": name, "reason": q.reason}
        if q.result == "sat":
            order = linearize(q.model, qs["progs"])
            rep = replay(s, warps, phases, qs["progs"], order)
            ev.update(failed=name, interleaving=[list(e) for e in order], replay=rep)
            # the interpreter must exhibit a wrong term, else the encoding is suspect
            return (None if rep["agrees"] else False), ev
    return True, ev


def slot_rotation(m: ModelInput) -> Draft:
    sv, pv, bv = m.var("S"), m.var("P"), m.var("beta")
    t0 = time.perf_counter()
    d = m.draft("bounded", "", ("pre", "X", "program"))
    table: list[dict[str, Any]] = []
    first_bad: dict[str, Any] | None = None
    undecided: list[dict[str, Any]] = []
    for s in m.menu(sv):
        for p in m.menu(pv):
            for b in m.menu(bv):
                point = {sv: s, pv: p, bv: b}
                admitted = m.pre_holds(point)
                safe, ev = safe_point(int(s), int(p), int(b), m, d)
                table.append({**point, "pre": admitted, "safe": safe})
                if safe is None:
                    undecided.append(
                        {**point, **{k: v for k, v in ev.items() if k != "interleaving"}}
                    )
                elif admitted and not safe and first_bad is None:
                    first_bad = {"point": point, **ev}
    exact = all(r["pre"] == r["safe"] for r in table)
    stronger = [r for r in table if r["safe"] and not r["pre"]]
    warps = int(m.param("warps", 2))
    d.domain = (
        f"{sv} in {m.menu(sv)}, {pv} in {m.menu(pv)}, {bv} in {m.menu(bv)} admitted by Pre; "
        f"every interleaving of the event model with {warps} warps and KT = max("
        f"{int(m.param('phases', 5))}, S+P+2) phases; leading barrier; other variables abstracted"
    )
    d.data = {"table": table, "pre_exact": exact, "safe_not_admitted": stronger}
    d.seconds = time.perf_counter() - t0
    if first_bad is not None:
        d.outcome = "counterexample"
        d.counterexample = {k: v for k, v in first_bad.items() if k != "replay"}
        d.replay = first_bad["replay"]
        d.reason = f"{first_bad['failed']} fails at {first_bad['point']} although Pre admits it"
    elif any(r["pre"] and r["safe"] is None for r in table):
        d.outcome = "unresolved"
        d.reason = "solver did not decide, or a witness did not replay, for an admitted point"
        d.data["undecided"] = undecided
    return d


def slot_progress(m: ModelInput) -> Draft:
    """Progress and completion agreement: for every point Pre admits, the program-order
    and barrier constraints have a solution (no circular wait: some complete run exists,
    and every stuck prefix can be extended because all warps cross the same barriers),
    and that run, replayed with tokens, gives exactly the reference products."""
    sv, pv, bv = m.var("S"), m.var("P"), m.var("beta")
    t0 = time.perf_counter()
    warps = int(m.param("warps", 2))
    d = m.draft("bounded", "", ("pre", "X", "program"))
    runs: list[dict[str, Any]] = []
    for s in m.menu(sv):
        for p in m.menu(pv):
            for b in m.menu(bv):
                point = {sv: s, pv: p, bv: b}
                if not m.pre_holds(point):
                    continue
                phases = max(int(m.param("phases", 5)), int(s) + int(p) + 2)
                progs = program(
                    int(s), int(p), int(b), warps, phases, bool(m.param("leading", True))
                )
                ev: dict[Event, Any] = {}
                for prog in progs:
                    for e in prog:
                        ev.setdefault(e, z3.Int(_name(e)))
                adm = [ev[x] < ev[y] for prog in progs for x, y in itertools.pairwise(prog)]
                adm.append(z3.Distinct(*[v for e, v in ev.items() if e[0] != "BAR"]))
                q = run(z3.And(*adm), {_name(e): v for e, v in ev.items()}, m.timeout_ms)
                d.add(q.record())
                if q.result == "unknown":
                    d.outcome, d.reason = "unresolved", f"solver unknown at {point}: {q.reason}"
                    break
                if q.result == "unsat":
                    d.outcome, d.counterexample = (
                        "counterexample",
                        {"point": point, "circular_wait": True},
                    )
                    d.replay = {"agrees": False, "reason": "no admissible complete run"}
                    d.reason = f"circular wait at {point}"
                    break
                rep = replay(int(s), warps, phases, progs, linearize(q.model, progs))
                runs.append({**point, "agrees": rep["agrees"]})
                if not rep["agrees"]:
                    d.outcome, d.counterexample, d.replay = "counterexample", {"point": point}, rep
                    d.reason = f"the witness run at {point} does not give the reference products"
                    break
            if d.outcome != "bounded":
                break
        if d.outcome != "bounded":
            break
    d.domain = (
        f"{sv} in {m.menu(sv)}, {pv} in {m.menu(pv)}, {bv} in {m.menu(bv)} admitted by Pre; "
        f"event model with {warps} warps and KT = max({int(m.param('phases', 5))}, S+P+2) phases"
    )
    d.data["runs"] = runs
    d.seconds = time.perf_counter() - t0
    return d
