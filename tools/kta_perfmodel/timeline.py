"""Export one replay as a Scope trace: the timeline `perfmodel/scope/` draws.

    python -m kta_perfmodel.timeline ROW [ROW ...] [--set KEY=VALUE ...] [--sens N] [--window A:B]
    python -m kta_perfmodel.timeline --demo

ROW is a row id of the error table (`fa_fwd_s1024_h16_d64 kernel`). The trace holds what the engine
decided, in engine time: every issue with the time its warp could have issued and why it waited,
every path stage of every transaction, each pipe's occupancy, the critical path, the machine's
parameters with their source and status, and ±30 % sensitivities. `--set` writes a what-if trace
beside its base. The engine is not changed: `recording` wraps two of its methods for the length of
one replay, and nothing here is part of `table.engine_hash`.

What a trace does not hold: activity on other SMs (a replay is one SM, the chip enters as shares),
bank-level shared memory, and data values. A row replayed on another engine round is another trace.
"""

from __future__ import annotations

import argparse
import bisect
import itertools
import json
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kta.models.gb10sim import attribution
from kta.models.gb10sim.machine import Machine, default_machine
from kta.models.gb10sim.sm import SM

from . import atlas, table, upstream

SCHEMA = 1
HERE = Path(__file__).resolve().parents[2]
MAX_ISSUES = 1_500_000  # a browser tab's worth; larger rows need --window
MIN_REGION = 12  # instructions; a shorter region between two barriers joins the one before it
STEP = 0.3
DEMO = (
    "fa_fwd_s1024_h16_d64 kernel",
    "tma 1024 full",
    "hgemm_m1024n1024k1024_c128x128x32_s4_t128 kernel",
    "gemm1024 c128x128x32_st3_th128 full",
    "softmax_m1024_n2048_warm kernel",
    "wi12b_dec_s1_warm kernel",
)
# parameters that belong to a resource by name, beyond those machine.yaml names
BY_NAME = {
    "scheduler": r"older_block|bar_|rbar|default_var|branch_redirect",
    "tensor": r"hmma|mma_|tensor",
    "fma": r"fma|ffma|alu_half",
    "alu": r"^alu",
    "xu": r"mufu|sfu",
    "mio": r"^mio",
    "smem_array": r"smem|ldsm|stsm|shfl",
    "lsu": r"lsu|ldg",
    "l1": r"^l1|lat_l1",
    "l2_return": r"l2_return|l2_aggregate|l2_rate|cp_|copy",
    "l2_chip": r"^l2|stg_read",
    "dram_queue": r"dram",
    "store_port": r"^st_|^stg|store",
    "store_queue": r"st_queue",
    "tma_engine": r"tma",
    "instruction_buffer": r"fetch|branch",
}


# what the page's unit interiors draw (js/units.js), whether or not this replay read them
UNIT_PARAMS = (
    "tensor_rates_by_type",
    "tensor_flop_per_clk_per_sm",
    "tensor_eff_pure_mma",
    "sim.hmma_16816_pipe_clk",
    "sim.hmma_latency_clk",
    "sim.tensor_beside_fma_clk",
    "sim.tensor_beside_alu_clk",
    "sim.tensor_beside_copy_clk",
    "sim.mma_operand_clk_per_set",
    "sim.reg_reads_per_clk",
    "smem_bytes_per_clk_per_sm",
    "smem_per_sm_bytes",
    "smem_reserved_per_block_bytes",
    "sim.mio_ldsm_clk",
    "sim.mio_stsm_clk",
    "sim.mio_ldgsts_clk",
    "sim.mio_smem_op_clk",
    "sim.smem_op_min_clk",
    "sim.smem_array_clk_per_512B",
    "sim.ldsm_latency_clk",
    "sim.ldsm_sm_clk_per_x4",
    "sim.lds_latency_clk",
    "sim.lds_latency_B_per_clk",
    "sim.shfl_latency_clk",
    "sim.sts_read_clk_per_128B",
)


# ports booked ahead, at an arrival after the issue: their time held is the engine's busy increment,
# not the span from the issue
LATER = {"l2_return": "fill", "store_port": "stg"}


@contextmanager
def recording(rec: dict[int, dict[str, list[Any]]]) -> Iterator[None]:
    """For one replay, per SM (a replay may run helper SMs first, the L2 pass): `holds`, every
    change of a sub-partition resource's free time at an issue (start, end, resource,
    sub-partition, warp, pc); `ports`, every advance of an SM or TPC port's free time (start, end,
    port, warp, pc, issue time), for all instructions, where the engine's stages exist only for
    those with a dependency counter; `copies`, the whole path of every cp.async (the engine keeps
    only the last of a group, with the commit that waits for it)."""
    from kta.models.gb10sim.ws import WSSM

    issues = {c: c.__dict__["issue"] for c in (SM, WSSM)}
    ev = SM.ev
    ws_event = WSSM.on_event
    depth = [0]

    def of(sm: Any) -> dict[str, list[Any]]:
        return rec.setdefault(id(sm), {"holds": [], "ports": [], "copies": []})

    def moved(
        sm: Any, before: dict[str, float], busy: dict[str, float], wid: int, pc: int, now: float
    ) -> None:
        for r, old in before.items():
            new = sm.t[r]
            if new > old and new > now:
                start = max(old, now)
                key = LATER.get(r)
                if key and sm.busy.get(key, 0.0) > busy.get(key, 0.0):
                    start = max(start, new - (sm.busy[key] - busy.get(key, 0.0)))
                of(sm)["ports"].append((start, new, r, wid, pc, now))

    def ws_on_event(self: Any, kind: str, w: Any, key: Any, now: float) -> None:
        ports, busy = dict(self.t), dict(self.busy)
        ws_event(self, kind, w, key, now)
        moved(self, ports, busy, w.wid, -1, now)

    def wrap(real: Any) -> Any:
        def issue(self: Any, w: Any, now: float) -> Any:
            if depth[0]:
                return real(self, w, now)
            depth[0] = 1
            pc, s = w.pc, w.smsp
            before = {r: self.free[r][s] for r in self.free}
            ports, busy = dict(self.t), dict(self.busy)
            try:
                return real(self, w, now)
            finally:
                depth[0] = 0
                for r, old in before.items():
                    new = self.free[r][s]
                    if new > old and new > now:
                        of(self)["holds"].append((max(old, now), new, r, s, w.wid, pc))
                moved(self, ports, busy, w.wid, pc, now)

        return issue

    def event(self: Any, t: float, kind: str, w: Any, arg: Any = None) -> None:
        if kind == "landed" and isinstance(arg, tuple):
            of(self)["copies"].append((w.wid, arg[1]))
        ev(self, t, kind, w, arg)

    try:
        for c, real in issues.items():
            c.issue = wrap(real)  # type: ignore[method-assign]
        SM.ev = event  # type: ignore[method-assign]
        WSSM.on_event = ws_on_event  # type: ignore[method-assign]
        yield
    finally:
        for c, real in issues.items():
            c.issue = real  # type: ignore[method-assign]
        SM.ev = ev  # type: ignore[method-assign]
        WSSM.on_event = ws_event  # type: ignore[method-assign]


def replay(
    cal: Path, job: list[Any], m: Machine
) -> tuple[Any, dict[str, Any], dict[str, list[Any]]]:
    """The SM of one traced replay, its table result and the recording."""
    rec: dict[int, dict[str, list[Any]]] = {}
    kept: list[Any] = []
    real = table._result

    def keep(predicted: float, sm: Any, extra: dict[str, Any], trace: bool) -> dict[str, Any]:
        kept.append(sm)
        return real(predicted, sm, extra, trace)

    table._result = keep  # type: ignore[assignment]
    try:
        with recording(rec):
            res = table.run_job(cal, job, m, trace=True)
    finally:
        table._result = real  # type: ignore[assignment]
    return kept[-1], res, rec[id(kept[-1])]


def regions(ins: list[Any], names: list[str]) -> list[dict[str, Any]]:
    """Phases of the program: stretches of code between synchronisation points (block barriers and
    mbarrier checks), named by what they hold. Their names say what the code is, not what it means:
    the replay has no source map."""
    cuts = [0]
    for i, x in enumerate(ins):
        if names[i] == "barrier" or x.op.startswith("SYNCS"):
            if i - cuts[-1] >= MIN_REGION:
                cuts.append(i)
    cuts.append(len(ins))
    out = []
    for a, b in itertools.pairwise(cuts):
        c = Counter(names[a:b])
        top = [f"{n} {k}" for n, k in c.most_common(4) if n not in ("other", "barrier")][:3]
        out.append({"start": a, "end": b, "top": top})
    return out


def phase_spans(hist: dict[int, list[tuple]], regs: list[dict[str, Any]]) -> list[list[float]]:
    """Per warp, the stretches spent in one region: [warp, region, start, end]."""
    starts = [r["start"] for r in regs]
    spans: list[list[float]] = []
    for wid, recs in hist.items():
        cur: list[float] | None = None
        for t, pc, *_ in recs:
            r = bisect.bisect_right(starts, pc) - 1
            if cur is None or cur[1] != r:
                if cur:
                    spans.append(cur)
                cur = [wid, r, t, t]
            cur[3] = t
        if cur:
            spans.append(cur)
    return spans


def transactions(
    sm: Any, copies: list[tuple], ports: list[tuple], pc_at: dict[tuple, int]
) -> list[list[Any]]:
    """[warp, pc, issue time, stages], one per instruction with a path; a stage is
    (start, end, resource, mechanism, parameter). Counters of one instruction share their stages."""
    by: dict[tuple, dict[tuple, None]] = defaultdict(dict)
    for w in sm.warps:
        for rels in w.sbsrc.values():
            for _, t_issue, stages in rels:
                for s in stages:
                    by[(w.wid, t_issue)][tuple(s)] = None
    for wid, path in copies:
        for s in path:
            by[(wid, path[0][0])][tuple(s)] = None
    for e in getattr(sm, "log", []):
        if e[0] == "tma":
            _, wid, _, t_issue, start, land = e
            if start > t_issue:
                by[(wid, t_issue)][(t_issue, start, "tma_engine", "queue (tma)", None)] = None
            by[(wid, t_issue)][(start, land, "tma_engine", "box load", None)] = None
    # time on a port: the stage the engine names, or `service` where it recorded none
    for a, b, res, wid, pc, now in ports:
        if pc >= 0 and not any(s[0] == a and s[1] == b and s[2] == res for s in by[(wid, now)]):
            by[(wid, now)][(a, b, res, "service", None)] = None
    return [
        [wid, pc_at.get((wid, t), -1), t, sorted(st, key=lambda s: (s[0], s[1]))]
        for (wid, t), st in sorted(by.items(), key=lambda kv: kv[0][1])
    ]


def locate(segs: list[tuple], hist: dict[int, list[tuple]], tx: list[list[Any]]) -> list[list[Any]]:
    """Critical-path segments in time order, each with the warp and pc it belongs to where one
    issue or one transaction matches it exactly: [start, end, resource, mechanism, param, warp, pc].
    """
    waits: dict[tuple, tuple[int, int]] = {}
    stalls: dict[tuple, tuple[int, int]] = {}
    for wid, recs in hist.items():
        for j, (t, pc, ready, *_) in enumerate(recs):
            waits[(ready, t)] = (wid, pc)
            if j:
                prev = recs[j - 1]
                stalls[(prev[0], ready)] = (wid, prev[1])
    stages = {(s[0], s[1], s[2], s[3]): (t[0], t[1]) for t in tx for s in t[3]}
    out = []
    for a, b, res, mech, par in sorted(segs, key=lambda s: (s[0], s[1])):
        hit = (
            stages.get((a, b, res, mech))
            or (waits.get((a, b)) if mech.startswith(("queue", "scheduler", "blocked")) else None)
            or (stalls.get((a, b)) if mech.startswith("stall") else None)
        )
        out.append([a, b, res, mech, par, *(hit or (None, None))])
    return out


def machine_view(m: Machine, reads: set[str], segs: list[tuple]) -> dict[str, Any]:
    """The machine's structure, and every parameter this replay read with its source."""
    raw = m.raw
    sim = raw["sim"]

    def entry(ref: str) -> dict[str, Any] | None:
        if ref.startswith("sim."):
            k = ref[4:]
            if k not in sim["value"]:
                return None
            v, src = sim["value"][k], sim.get("value_source", {}).get(k)
        elif ref in raw and not ref.startswith("_"):
            v, src = raw[ref]["value"], raw[ref].get("source")
        else:
            return None
        small = isinstance(v, int | float | str | bool) or len(json.dumps(v)) < 1500
        return {
            "value": v if small else json.dumps(v)[:120],
            "status": m.status(ref),
            "source": src,
        }

    links: dict[str, list[str]] = defaultdict(list)
    for name, r in m.desc["resources"].items():
        for v in r.values():
            for u in v.values() if isinstance(v, dict) else [v]:
                if isinstance(u, str) and entry(u) is not None and u not in links[name]:
                    links[name].append(u)
    for _, c in m.classes:
        for res, ref in (*c.hold, *c.beside):
            for name in c.either if res == "either" else (res,):
                if ref not in links[name]:
                    links[name].append(ref)
        if c.latency and c.hold:
            res = c.hold[0][0]
            for name in c.either if res == "either" else (res,):
                if c.latency not in links[name]:
                    links[name].append(c.latency)
    for _, _, res, _, par in segs:
        if par and entry(par) is not None and res in m.desc["resources"]:
            if par not in links[res]:
                links[res].append(par)
    every = sorted(reads - {"*"}) if reads else []
    for name, pat in BY_NAME.items():
        for ref in every:
            if re.search(pat, ref.removeprefix("sim.")) and ref not in links[name]:
                links[name].append(ref)
    refs = sorted({*every, *UNIT_PARAMS, *(r for v in links.values() for r in v)})
    return {
        "name": m.desc["name"],
        "hierarchy": m.desc["hierarchy"],
        "execution": m.desc["execution"],
        "resources": m.desc["resources"],
        "classes": [
            {
                "name": c.name,
                "gates": list(c.gates),
                "either": list(c.either),
                "hold": list(c.hold),
                "latency": c.latency,
            }
            for _, c in m.classes
        ],
        "facts": {
            "sms": m.sms,
            "smsp_per_sm": m.smsp_per_sm,
            "tpcs": m.desc["hierarchy"]["tpc"]["count"],
            "regs_per_sm": m.C["regs_per_sm"],
            "smem_per_sm_bytes": m.C["smem_per_sm_bytes"],
            "max_threads_per_sm": m.C["max_threads_per_sm"],
            "max_blocks_per_sm": m.C["max_blocks_per_sm"],
            "l2_bytes": m.C["l2_bytes"],
            "l2_effective_bytes": m.C["l2_effective_bytes"],
            "dram_read_GBs": m.C["dram_read_GBs"],
        },
        "params": {r: e for r in refs if (e := entry(r)) is not None},
        "links": {k: v for k, v in links.items() if v},
    }


def _sensitivity_one(
    a: tuple[Path, list[Any], str, float, float],
) -> tuple[str, float, float | None]:
    cal, job, ref, value, f = a
    try:
        m = default_machine().with_overrides({ref: value * f})
        return ref, f, table.run_job(cal, job, m, trace=False)["predicted"]
    except (ArithmeticError, IndexError, KeyError, TypeError, ValueError):
        # a value the rule cannot take (a count, a table): no number, not a failure
        return ref, f, None


def sensitivity(
    cal: Path,
    job: list[Any],
    base: float,
    view: dict[str, Any],
    attr: list[dict[str, Any]],
    n: int,
    jobs: int = 3,
) -> list[dict[str, Any]]:
    """Time change for each parameter ±30 %: those on the critical path first, then each pipe's and
    port's own. Replays whose value does not reach a rule change nothing and say so."""
    picked: list[str] = []
    for a in attr:
        if a.get("param") and a["param"] not in picked:
            picked.append(a["param"])
    # resources by their share of the path: most of a tensor queue names no parameter of its own
    share: dict[str, float] = defaultdict(float)
    for a in attr:
        share[a["resource"]] += a["share"]
    for res in sorted(view["links"], key=lambda r: -share[r]):
        picked += [r for r in view["links"][res][:3] if r not in picked]
    numeric = [
        r
        for r in picked
        if isinstance(view["params"].get(r, {}).get("value"), int | float)
        and not isinstance(view["params"][r]["value"], bool)
        and view["params"][r]["value"]
    ][:n]
    tasks = [
        (cal, job, r, float(view["params"][r]["value"]), f)
        for r in numeric
        for f in (1 - STEP, 1 + STEP)
    ]
    got: dict[str, dict[float, float | None]] = defaultdict(dict)
    with ProcessPoolExecutor(jobs) as pool:
        for ref, f, v in pool.map(_sensitivity_one, tasks):
            got[ref][f] = v
    out = []
    for r in numeric:
        lo, hi = got[r][1 - STEP], got[r][1 + STEP]
        out.append(
            {
                "param": r,
                "minus_pct": None if lo is None else round((lo / base - 1) * 100, 2),
                "plus_pct": None if hi is None else round((hi / base - 1) * 100, 2),
            }
        )
    return sorted(out, key=lambda d: -abs(d["minus_pct"] or 0) - abs(d["plus_pct"] or 0))


def _r(x: float) -> float:
    return round(float(x), 1)


def build(
    spec: dict[str, Any], cal: Path, over: dict[str, Any], sens: int, window: tuple | None
) -> dict[str, Any]:
    m = default_machine()
    m = m.with_overrides(over) if over else m
    tracked = m.tracked()
    sm, res, rec = replay(cal, spec["job"], tracked)
    hist: dict[int, list[tuple]] = sm.hist
    issued = sum(len(v) for v in hist.values())
    if issued > MAX_ISSUES and window is None:
        raise SystemExit(f"{spec['id']}: {issued} issues; give --window A:B in cycles")
    names = [c.name for c in sm.icls]
    regs = regions(sm.ins, names)
    pc_at = {(wid, r[0]): r[1] for wid, recs in hist.items() for r in recs}
    tx = transactions(sm, rec["copies"], rec["ports"], pc_at)
    segs = attribution.critical_path(sm, sm.cycles)
    crit = locate(segs, hist, tx)
    view = machine_view(m, set(tracked.reads), segs)
    busy_check = checks(sm, rec, crit)
    a, b = window or (0.0, float(sm.cycles))

    why_names: list[str] = []
    why_ix: dict[str, int] = {}

    def why_code(y: Any) -> int:
        k = "ready" if y is None else ":".join(str(v) for v in y)
        if k not in why_ix:
            why_ix[k] = len(why_names)
            why_names.append(k)
        return why_ix[k]

    rows = sorted(
        (
            (r[0], wid, r[1], r[2], r[3])
            for wid, recs in hist.items()
            for r in recs
            if a <= r[0] <= b
        ),
        key=lambda r: (r[0], r[1]),
    )
    stats: dict[int, list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
    for t, _, pc, ready, _ in rows:
        stats[pc][0] += 1
        stats[pc][1] += t - ready
    for s in crit:
        if s[6] is not None:
            stats[s[6]][2] += s[1] - s[0]
    pool: dict[str, int] = {}

    def label(s: str) -> int:
        return pool.setdefault(s, len(pool))

    stages: dict[str, list[Any]] = {k: [] for k in ("s", "e", "r", "m", "p", "x")}
    txs: dict[str, list[Any]] = {"w": [], "pc": [], "t": []}
    for t in tx:
        if not any(s[1] > a and s[0] < b and s[1] > s[0] for s in t[3]):
            continue
        n = len(txs["w"])
        txs["w"].append(t[0])
        txs["pc"].append(t[1])
        txs["t"].append(_r(t[2]))
        for s in t[3]:
            if s[1] > s[0]:
                stages["s"].append(_r(s[0]))
                stages["e"].append(_r(s[1]))
                stages["r"].append(label(s[2]))
                stages["m"].append(label(s[3]))
                stages["p"].append(-1 if s[4] is None else label(s[4]))
                stages["x"].append(n)
    holds = sorted(h for h in rec["holds"] if h[1] > a and h[0] < b)
    ports = sorted(h for h in rec["ports"] if h[1] > a and h[0] < b)
    rels = [
        (w.wid, c, _r(t_rel), _r(t_issue))
        for w in sm.warps
        for c, rs in w.sbsrc.items()
        for t_rel, t_issue, _ in rs
    ]
    blocks = defaultdict(lambda: [float("inf"), 0.0])
    for w in sm.warps:
        h = hist.get(w.wid)
        if h:
            blocks[w.block] = [min(blocks[w.block][0], h[0][0]), max(blocks[w.block][1], h[-1][0])]
    measured = spec.get("measured")
    if not isinstance(measured, int | float) or measured != measured:
        measured = None
    clock = m.param(m.desc["clock"]["kernel_GHz"])
    return {
        "schema": SCHEMA,
        "meta": {
            "row": spec["id"],
            "set": spec.get("group"),
            "unit": spec.get("unit"),
            "measured": measured,
            "predicted": res["predicted"],
            "note": spec.get("note"),
            "cycles": _r(sm.cycles),
            "clock_GHz": clock,
            "issued": issued,
            "blocks": sm.blocks,
            "grid_blocks": getattr(sm, "grid_blocks", None),
            "resident": getattr(sm, "resident", None),
            "active_sms": getattr(sm, "active_sms", None),
            "engine_class": type(sm).__name__,
            "engine": table.engine_hash()[:12],
            "commit": _git("rev-parse", "--short", "HEAD"),
            "dirty": bool(_git("status", "--porcelain", "src/kta/models/gb10sim")),
            "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "overrides": over,
            "window": [a, b] if window else None,
            "checks": busy_check,
            "counts": {
                k: v
                for k, v in res["counts"].items()
                if k in ("issued", "HMMA", "LDGSTS", "copy_sectors", "ldg_sectors", "stg_sectors")
            },
            "busy": res["counts"]["busy"],
            "coverage": res["coverage"],
            "defaults": res["defaults"],
            "attribution": res["attribution"],
            "counters": spec.get("counters"),
        },
        "machine": view,
        "ins": [
            {
                "a": x.addr,
                "t": (x.pred + " " if x.pred else "") + x.text,
                "c": names[i],
                "n": stats[i][0] if i in stats else 0,
                "w": _r(stats[i][1]) if i in stats else 0,
                "k": _r(stats[i][2]) if i in stats else 0,
            }
            for i, x in enumerate(sm.ins)
        ],
        "warps": [
            {"id": w.wid, "block": w.block, "smsp": w.smsp, "role": w.role} for w in sm.warps
        ],
        "blocks": {str(k): [_r(v[0]), _r(v[1])] for k, v in sorted(blocks.items())},
        "regions": regs,
        "spans": [
            [s[0], s[1], _r(s[2]), _r(s[3])]
            for s in phase_spans(hist, regs)
            if s[3] >= a and s[2] <= b
        ],
        "why": why_names_after(rows, why_code, why_names),
        "issues": {
            "t": [_r(r[0]) for r in rows],
            "w": [r[1] for r in rows],
            "pc": [r[2] for r in rows],
            "r": [_r(r[3]) for r in rows],
            "y": [why_code(r[4]) for r in rows],
        },
        "labels": list(pool),
        "tx": txs,
        "stages": stages,
        "holds": {
            k: [h[i] if i > 1 else _r(h[i]) for h in holds]
            for k, i in (("s", 0), ("e", 1), ("r", 2), ("p", 3), ("w", 4), ("pc", 5))
        }
        | {"own": [int(_own(sm, h[5], h[2])) for h in holds]},
        "ports": {
            k: [h[i] if i > 1 else _r(h[i]) for h in ports]
            for k, i in (("s", 0), ("e", 1), ("r", 2), ("w", 3), ("pc", 4))
        },
        "rels": rels,
        "crit": [[_r(c[0]), _r(c[1]), c[2], c[3], c[4], c[5], c[6]] for c in crit],
        "sens": [],
    }


def why_names_after(rows: list[tuple], code: Any, names: list[str]) -> list[str]:
    """Fills the reason table first, so `why` and the `y` column come out in one order."""
    for r in rows:
        code(r[4])
    return names


def checks(sm: Any, rec: dict[str, list[Any]], crit: list[list[Any]]) -> dict[str, Any]:
    """The trace against the engine's own totals. Pipes differ by the part of a clock a hold
    carries to the next one (D963): the trace has the time held, the engine's busy the sum of c."""
    total: Counter = Counter()
    for s, e, r, _, _, pc in rec["holds"]:
        if _own(sm, pc, r):
            total[r] += e - s
    pipes = {}
    for r, k in (("tensor", "tensor"), ("fma", "fma"), ("alu", "alu"), ("xu", "sfu")):
        pipes[r] = {"trace": round(total[r], 1), "engine": round(sm.busy.get(k, 0.0), 1)}
    port_total: Counter = Counter()
    for s, e, r, *_ in rec["ports"]:
        port_total[r] += e - s
    return {
        "pipes": pipes,
        "ports": {
            r: {"trace": round(port_total[r], 1), "engine": round(sm.busy.get(k, 0.0), 1)}
            for r, k in (("store_port", "stg"), ("lsu", "lsu"), ("l2_return", "fill"))
        },
        "path_sum": round(sum(c[1] - c[0] for c in crit), 1),
        "cycles": round(sm.cycles, 1),
        "path_located": round(
            sum(c[1] - c[0] for c in crit if c[5] is not None) / max(1.0, sm.cycles), 3
        ),
    }


def _own(sm: Any, pc: int, res: str) -> bool:
    """Whether the instruction at pc holds `res` itself; otherwise it is the delay it adds to the
    pipe beside it (machine.yaml `beside`), which the engine's busy counters leave out."""
    c = sm.icls[pc]
    return res == "mio" or any(r == res or (r == "either" and res in c.either) for r, _ in c.hold)


def _git(*a: str) -> str:
    return subprocess.run(
        ["git", *a], cwd=upstream.ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def slug(row: str, over: dict[str, Any]) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", row).strip("-")
    return s + "".join(f"~{k.removeprefix('sim.')}={v}" for k, v in sorted(over.items()))


def find_specs(rows: list[str], cal: Path) -> dict[str, dict[str, Any]]:
    want = set(rows)
    out: dict[str, dict[str, Any]] = {}
    for it in (table.gemm_specs(cal), table.tma_specs(cal), table.latest(table.batch_specs())):
        for r in it:
            if r["id"] in want:
                out[r["id"]] = r
    missing = want - out.keys()
    if missing:
        raise SystemExit(f"no such row: {sorted(missing)}")
    return out


def write_index(out: Path) -> None:
    """index.json: what the page lists."""
    items = []
    for p in sorted(out.glob("*.json")):
        if p.name in ("index.json", "atlas.json"):
            continue
        d = json.loads(p.read_text())
        meta = d["meta"]
        items.append(
            {
                "file": p.name,
                "row": meta["row"],
                "set": meta["set"],
                "predicted": meta["predicted"],
                "measured": meta["measured"],
                "unit": meta["unit"],
                "cycles": meta["cycles"],
                "issued": meta["issued"],
                "overrides": meta["overrides"],
                "attribution": meta["attribution"],
                "busy": meta["busy"],
                "mb": round(p.stat().st_size / 1e6, 1),
            }
        )
    (out / "index.json").write_text(json.dumps(items, indent=1))


def all_params(m: Machine) -> dict[str, dict[str, Any]]:
    """Every value of gb10.json with its status and source, for the page's unit interiors."""
    raw = m.raw
    out: dict[str, dict[str, Any]] = {}
    for k, v in raw["sim"]["value"].items():
        ref = f"sim.{k}"
        out[ref] = {
            "value": v,
            "status": m.status(ref),
            "source": raw["sim"].get("value_source", {}).get(k),
        }
    for k, e in raw.items():
        if not k.startswith("_") and k != "sim":
            out[k] = {"value": e["value"], "status": m.status(k), "source": e.get("source")}
    return out


def write_atlas(out: Path) -> None:
    """atlas.json: the mechanisms, from the records of the tree whose engine replays."""
    m = default_machine()
    view = {
        "classes": m.desc["classes"],
        "resources": m.desc["resources"],
        "params": all_params(m),
    }
    (out / "atlas.json").write_text(
        json.dumps(atlas.build(upstream.ROOT, view), separators=(",", ":"))
    )


def export(
    row: str,
    spec: dict[str, Any],
    cal: Path,
    over: dict[str, Any],
    sens: int,
    window: Any,
    out: Path,
) -> Path:
    d = build(spec, cal, over, sens, window)
    if sens and not over:
        d["sens"] = sensitivity(
            cal, spec["job"], d["meta"]["predicted"], d["machine"], d["meta"]["attribution"], sens
        )
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{slug(row, over)}.json"
    p.write_text(json.dumps(d, separators=(",", ":")))
    c = d["meta"]["checks"]
    print(
        f"{p.name}: {d['meta']['issued']} issues, {d['meta']['cycles']:.0f} clk, "
        f"{p.stat().st_size / 1e6:.1f} MB, path {c['path_sum']:.0f}/{c['cycles']:.0f}, "
        f"located {c['path_located']:.0%}, pipes {c['pipes']}"
    )
    return p


def parse_set(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k] = json.loads(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("rows", nargs="*")
    ap.add_argument("--demo", action="store_true", help="the rows the page ships with")
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE, gb10.json name")
    ap.add_argument("--sens", type=int, default=14, help="parameters to vary ±30 %% (0: none)")
    ap.add_argument("--window", help="A:B in cycles, for rows too long for one tab")
    ap.add_argument("--atlas", action="store_true", help="rewrite atlas.json only, no replay")
    ap.add_argument("--out", type=Path, default=HERE / "perfmodel" / "scope" / "traces")
    a = ap.parse_args()
    rows = [*a.rows, *(DEMO if a.demo else ())]
    a.out.mkdir(parents=True, exist_ok=True)
    if a.atlas:
        write_atlas(a.out)
        return
    if not rows:
        ap.error("give a row id or --demo")
    cal, _ = upstream.calibration_dir(upstream.UPSTREAM, upstream.UPSTREAM_REV)
    specs = find_specs(rows, cal)
    win = tuple(float(v) for v in a.window.split(":")) if a.window else None
    over = parse_set(a.set)
    for r in rows:
        export(r, specs[r], cal, over, a.sens, win, a.out)
    write_index(a.out)
    write_atlas(a.out)


if __name__ == "__main__":
    main()
