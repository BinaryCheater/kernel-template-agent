"""Critical-path attribution and the assumption ledger (docs/design/27 §3 items 4 and 6).

A simulator run with `trace=True` keeps, per warp, every issue with the earliest time the warp could
have issued it and the condition that was last found blocking it (dependency counter, resource,
barrier, mbarrier hand-off, register hand-off), and, per dependency counter, each release with the
producing instruction's issue time and its path stages (queue, address stage, latency, return
port, array write, ...). `critical_path` walks back from the end of the run along the constraint
that bound each step:

- a dependency counter: through the producer's path stages to the producer's issue (same warp);
- a block barrier: the release latency, then to the last warp that arrived;
- an mbarrier hand-off: the hand-off delay, then to the TMA box that landed last or the arrive of
  the producer, or to the consumer warp whose release the producer waited for;
- a busy resource, or another warp issuing first: that wait, then the warp's own stall count;
- a block's first instruction: the replacement of the previous block on its slot.

Segments are contiguous from the start of the run to its end, so their durations add up to the
predicted cycles. What is not on the path is what overlap hid. A wait on a busy resource is charged
to that resource without following who held it (a queue, not a chain).
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from typing import Any

from .sm import SM

Seg = tuple[float, float, str, str, str | None]  # start, end, resource, mechanism, parameter


def _index_at(times: list[float], t: float) -> int:
    """Index of the last issue at or before t."""
    return bisect.bisect_right(times, t) - 1


def _stages(t0: float, t1: float, stages: list[Seg]) -> list[Seg]:
    """The stages of one release clipped to [t0, t1], gaps filled."""
    out: list[Seg] = []
    cur = t0
    for a, b, res, mech, par in stages:
        a, b = max(a, cur), min(b, t1)
        if b <= a:
            continue
        if a > cur:
            out.append((cur, a, "issue", "path gap", None))
        out.append((a, b, res, mech, par))
        cur = b
    if t1 > cur:
        out.append((cur, t1, "issue", "path gap", None))
    return out


def _ws_index(sm: Any) -> dict[tuple, list[tuple]]:
    """WSSM's event log by (kind, block, k-tile or None), in time order."""
    block_of = {w.wid: w.block for w in sm.warps}
    idx: dict[tuple, list[tuple]] = defaultdict(list)
    for e in getattr(sm, "log", []):
        if e[0] in ("tma", "arrive"):
            idx[(e[0], block_of[e[1]], e[2])].append(e)
        elif e[0] == "release":
            idx[("release", block_of[e[1]], None)].append(e)
        elif e[0] == "dealloc":
            idx[("dealloc", e[2], None)].append(e)
    return idx


def _ws_binding(
    sm: Any, idx: dict[tuple, list[tuple]], why: tuple, ti: float
) -> tuple[list[Seg], int, float] | None:
    """WSSM waits: (segments back to the jump point, warp to continue on, its time)."""
    wake = sm.P["mbar_wake_clk"]
    if why[0] == "mbar_full":
        b, k = why[1], why[2]
        arr = sm.arrived.get((b, k))
        landed = sm.landed.get((b, k))
        boxes = idx.get(("tma", b, k), [])
        if landed is not None and boxes and (arr is None or landed >= arr):
            _, wid, _, t_issue, start, land = max(boxes, key=lambda e: e[5])
            lat_end = min(land, start + sm.P["tma_l2_latency_clk"])
            mech = "transfer (engine rate)" if not sm.miss_frac else "transfer / DRAM queue"
            par = "tma_r_GBs_alone" if not sm.miss_frac else None
            segs = [
                (landed + wake, ti, "issue", "after mbarrier hand-off", None),
                (landed, landed + wake, "mbarrier", "hand-off (full)", "sim.mbar_wake_clk"),
                (land, landed, "smem_array", "TMA box write (incl. queue)", None),
                (lat_end, land, "tma_engine", mech, par),
                (start, lat_end, "tma_engine", "L2 latency (TMA)", "sim.tma_l2_latency_clk"),
                (t_issue, start, "tma_engine", "queue", None),
            ]
            return segs, wid, t_issue
        arrives = idx.get(("arrive", b, k), [])
        if arr is not None and arrives:
            wid = arrives[-1][1]
            segs = [
                (arr + wake, ti, "issue", "after mbarrier hand-off", None),
                (arr, arr + wake, "mbarrier", "hand-off (full)", "sim.mbar_wake_clk"),
            ]
            return segs, wid, arr
    elif why[0] == "mbar_empty":
        b, n = why[1], why[2]
        rel = idx.get(("release", b, None), [])
        if len(rel) >= n:
            _, wid, _, t = rel[n - 1]
            segs = [
                (t + wake, ti, "issue", "after mbarrier hand-off", None),
                (t, t + wake, "mbarrier", "hand-off (empty)", "sim.mbar_wake_clk"),
            ]
            return segs, wid, t
    elif why[0] == "regalloc":
        deal = idx.get(("dealloc", why[1], None), [])
        if deal:
            _, wid, _, t = deal[-1]
            return [(t, ti, "regalloc", "register hand-off", None)], wid, t
    return None


def critical_path(sm: SM, cycles: float) -> list[Seg]:
    """The binding chain from 0 to `cycles`, latest first."""
    hist = sm.hist
    if not hist:
        return [(0.0, cycles, "issue", "untraced", None)]
    times = {wid: [r[0] for r in recs] for wid, recs in hist.items()}
    # the compiler's stall count after a pipe instruction covers that pipe's hold or latency
    hold_of = {c.name: c.hold[0][0] for _, c in sm.m.classes if c.hold and c.hold[0][0] != "either"}
    warps = {w.wid: w for w in sm.warps}
    wid = max(hist, key=lambda q: hist[q][-1][0])
    t_last = hist[wid][-1][0]
    drain = getattr(sm, "end", 0) > t_last
    segs: list[Seg] = [
        (
            t_last,
            cycles,
            "store_port" if drain else "issue",
            "drain after the last instruction" if drain else "last instruction",
            None,
        )
    ]
    i = len(hist[wid]) - 1
    ws_idx: dict[tuple, list[tuple]] | None = None
    for _ in range(10_000_000):
        if i < 0:
            break
        ti, _pc, ready, why, _cname = hist[wid][i]
        tag = why[0] if why else None
        jump: tuple[int, float] | None = None
        if tag == "sb":
            rels = [r for r in warps[wid].sbsrc.get(why[1], []) if r[0] <= ti]
            if rels:
                t_rel, t_p, stages = max(rels, key=lambda r: r[0])
                if ti > t_rel:
                    segs.append((t_rel, ti, "issue", "after counter release", None))
                segs += reversed(_stages(t_p, t_rel, stages))
                jump = (wid, t_p)
        elif tag == "bar":
            src = [r for r in warps[wid].bar_src if r[0] <= ti]
            if src:
                rel, t_arr, last = max(src, key=lambda r: r[0])
                if ti > rel:
                    segs.append((rel, ti, "issue", "after barrier release", None))
                segs.append((t_arr, rel, "barrier", "release latency", "sim.bar_base_clk"))
                jump = (last, t_arr)
        elif tag in ("mbar_full", "mbar_empty", "regalloc"):
            if ws_idx is None:
                ws_idx = _ws_index(sm)
            got = _ws_binding(sm, ws_idx, why, ti)
            if got is not None:
                more, w2, t2 = got
                segs += [s for s in more if s[1] > s[0]]
                jump = (w2, t2)
        if jump is not None:
            wid = jump[0]
            i = _index_at(times[wid], jump[1])
            continue
        if ti > ready:
            if tag == "res":
                segs.append((ready, ti, why[1], f"queue ({why[2]})", None))
            elif tag is None:
                segs.append((ready, ti, "issue", "scheduler (another warp issued)", None))
            else:
                segs.append((ready, ti, "issue", f"blocked ({tag})", None))
        if i > 0:
            prev = hist[wid][i - 1]
            if ready > prev[0]:
                res = hold_of.get(prev[4], "issue")
                segs.append((prev[0], ready, res, f"stall count after {prev[4]}", None))
            i -= 1
            continue
        pred = sm.block_pred.get(warps[wid].block)
        if pred is None:
            if ready > 0:
                segs.append((0.0, ready, "cta_dispatch", "block start", None))
            break
        b_prev, t_exit = pred
        cands = [
            (hist[q][_index_at(times[q], t_exit)][0], q)
            for q in hist
            if warps[q].block == b_prev and _index_at(times[q], t_exit) >= 0
        ]
        segs.append((t_exit, ready, "cta_dispatch", "block replacement", None))
        if not cands:
            break
        _, wid = max(cands)
        i = _index_at(times[wid], t_exit)
    return segs


def aggregate(segs: list[Seg], total: float) -> list[dict[str, Any]]:
    """Shares of `total` per (resource, mechanism), largest first."""
    acc: dict[tuple[str, str], float] = defaultdict(float)
    params: dict[tuple[str, str], str | None] = {}
    for a, b, res, mech, par in segs:
        acc[(res, mech)] += b - a
        params.setdefault((res, mech), par)
    out = [
        {"resource": r, "mechanism": m, "cycles": v, "share": v / total, "param": params[(r, m)]}
        for (r, m), v in acc.items()
        if v > 0
    ]
    return sorted(out, key=lambda d: -d["cycles"])


def ledger(sm: SM, path: list[dict[str, Any]]) -> dict[str, Any]:
    """Defaults the run used and how often; assumed or fitted parameters on the critical path."""
    m = sm.m
    on_path: dict[str, float] = defaultdict(float)
    for d in path:
        if d["param"] and m.status(d["param"]) in ("assumed", "fitted", "override"):
            on_path[d["param"]] += d["share"]
    return {
        "assumed_latency": dict(sm.assumed_latency),
        **{k: dict(v) for k, v in sm.defaults.items()},
        "assumed_or_fitted_on_path": {
            k: {"status": m.status(k), "share": v} for k, v in on_path.items()
        },
    }


def analyse(
    sm: SM, cycles: float, extra: dict[str, tuple[float, str | None]] | None = None
) -> dict[str, Any]:
    """Attribution and ledger of a traced run; `extra` adds terms outside the SM, name ->
    (cycles, parameter), e.g. the launch, to the total."""
    segs = critical_path(sm, cycles)
    total = cycles + sum(v for v, _ in (extra or {}).values())
    covered = sum(b - a for a, b, *_ in segs)
    path = aggregate(segs, total)
    for name, (v, par) in (extra or {}).items():
        path.append(
            {"resource": name, "mechanism": name, "cycles": v, "share": v / total, "param": par}
        )
    path.sort(key=lambda d: -d["cycles"])
    return {"path": path, "covered_cycles": covered, "cycles": cycles, "ledger": ledger(sm, path)}


def default_counts(led: dict[str, Any]) -> dict[str, int]:
    """One number per kind of default, for the error table's column."""
    out = {
        "default latency": sum(led["assumed_latency"].values()),
        "default rbar": sum(led["rbar_default"].values()),
        "issue-only": sum(led["issue_only"].values()),
        "branch fallback": sum(led["branch_fallback"].values()),
        "front end": sum(led["front_end"].values()),
    }
    return {k: v for k, v in out.items() if v}
