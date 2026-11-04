"""The error table (docs/design/28 §4.7): every validation kernel and probe replayed on CPU,
measured against predicted, one row each.

A row is a spec (identity, measured value) plus a job (what to replay). Jobs run in a process
pool, and each result is cached under a key made of the replay code (`engine_hash`), the upstream
input commit and the job. A result also records the values of the parameters the replay read, and
a cached result counts only when those values still hold, with any override applied: a changed or
ablated `sim.value` entry replays exactly the rows that read it. Parameters outside `sim` are in
`engine_hash`, because `model.C` reads them behind the tracked machine. The kernel, b2b and graph
rows of a test kernel share one replay."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import signal
import statistics
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from kta.models.gb10sim import attribution, probes, probeset, ws
from kta.models.gb10sim.kernel import generic_kernel, machine_trips, sass_trips, tile_launch
from kta.models.gb10sim.machine import Machine, default_machine
from kta.models.gb10sim.model import C
from kta.models.gb10sim.sass import parse
from kta.models.regression import spearman

SETS = ("gemm", "tma", "probes", "batches")
BATCHES = Path(__file__).resolve().parents[2] / "perfmodel" / "batches"
# rows left out of the table: `tma 8192 full` (replay 150-300 s, far above the 28 §4 budget;
# deleted at the user's request, round 1)
DROPPED = {"tma 8192 full"}
SLOW: dict[str, str] = {}
ENGINE = Path(probeset.__file__).parent
CACHE = Path(__file__).resolve().parents[2] / "data" / "perfmodel" / "cache"
BUDGET_S = 600.0  # CPU seconds of one replay; the slowest row takes about 90
LOCAL_JOBS = 6
# LOOP §4.4 "mechanisms only": every fitted parameter the replays read, replaced by the probe value
# of the same component (gb10.json) or by nothing when no probe measures it (D-number in
# docs/decisions.md)
MECH_ONLY: dict[str, Any] = {
    "launch_stream_us": 2.05,  # launch_costs_us.stream_empty_back_to_back (probe launch.cu)
    "tma_r_GBs_alone": 80.7,  # port_GBs_per_sm (probe mlp_stream.cu)
    "tma_l2_aggregate_GBs": 2250,  # cpasync_l2_aggregate_GBs (probe tileload.cu)
    "block_setup_us": 0.0,  # no probe
}


def _spec(rid, group, level, unit, measured, job, excluded=None, note=None) -> dict[str, Any]:
    ok = measured == measured and measured > 0
    return {
        "id": rid,
        "set": group,
        "level": level,
        "unit": unit,
        "measured": measured,
        "excluded": excluded if ok else (excluded or "no measured value"),
        "note": note,
        "job": job,
    }


def gemm_specs(cal: Path) -> Iterator[dict[str, Any]]:
    """The 22 cp.async GEMMs at 1024^3, warm: the whole kernel, its loads-only variant at the grid,
    and its mma-only variant on one block, all through kernel.generic_kernel."""
    m = json.loads((cal / "ablation_cpasync" / "mechanisms.json").read_text())
    abl = [
        json.loads(line)
        for line in (cal / "ablation_cpasync" / "ablation.jsonl").read_text().splitlines()
        if line
    ]
    for name in sorted(m):
        r = m[name]
        g = "gemm1024"
        yield _spec(
            f"{g} {name} full", f"{g} full", 7, "us", r["full_event_us"], ["gemm", name, "full"]
        )
        nc = [
            x
            for x in abl
            if x["name"] == name
            and x["variant"] == "nocompute"
            and x.get("ok")
            and int(x["blocks"]) == r["blocks"]
        ]
        yield _spec(
            f"{g} {name} nocompute",
            f"{g} nocompute",
            7,
            "us",
            nc[0]["median_us"] if nc else float("nan"),
            ["gemm", name, "nocompute"],
        )
        yield _spec(
            f"{g} {name} noload 1 block",
            f"{g} noload 1 block",
            7,
            "us",
            r["compute_per_ktile_us"] * r["KT"] + r["overlap"]["setup_us"],
            ["gemm_block", name, "noload"],
        )


def tma_specs(cal: Path) -> Iterator[dict[str, Any]]:
    """The 7 warp-specialised TMA picks (ws.py): whole kernel at its grid (warm),
    and the mma-only variant on one block."""
    m = json.loads((cal / "ablation" / "mechanisms.json").read_text())
    for name, r in m.items():
        note = (
            "measured warm in a session where part of B stayed in the L2; upstream compares the "
            "decode picks cold against a later flush measurement (gb10 design note §7)"
            if name in ("m32", "m128")
            else None
        )
        if f"tma {name} full" not in DROPPED:
            yield _spec(
                f"tma {name} full",
                "tma full",
                7,
                "us",
                r["full_event_us"],
                ["tma", name],
                note=note,
            )
        setup = r.get("block_setup_us") or C["block_setup_us"]
        yield _spec(
            f"tma {name} noload 1 block",
            "tma noload 1 block",
            7,
            "us",
            r["compute_per_ktile_us"] * r["KT"] + setup,
            ["tma_block", name],
        )


def probe_specs(cal: Path) -> Iterator[dict[str, Any]]:
    for case in probes.all_cases(cal):
        yield _spec(
            case.id,
            f"probe {case.probe}",
            case.level,
            case.unit,
            case.measured,
            ["probe", case.id],
            excluded=case.excluded,
        )


def _batch_key(d: Path) -> str:
    """What a batch row's replay depends on beyond the engine: its SASS, launch files and the
    clock it measured."""
    h = hashlib.sha256()
    for p in sorted((d / "build").rglob("*")):
        if p.name in ("k.sass", "launch.json", "res-usage.txt", "kernel.cu"):
            h.update(str(p.relative_to(d)).encode())
            h.update(p.read_bytes())
    h.update(str(probeset.load(d).clock_GHz).encode())
    return h.hexdigest()[:16]


def batch_specs(root: Path = BATCHES) -> Iterator[dict[str, Any]]:
    """KTA's own measurement batches (perfmodel/batches/<batch>/): probes and test-set kernels.
    A batch that is not valid (gb10-measure) contributes no measured values."""
    dirs = [p for p in root.iterdir() if (p / "build").is_dir()] if root.exists() else []
    for d in sorted(dirs, key=lambda p: batch_order(p.name)):
        b = probeset.load(d)
        meas = d / "measured.json"
        mdoc = json.loads(meas.read_text()) if meas.exists() else {}
        valid = mdoc.get("valid", False)
        counters = mdoc.get("counters", {})
        key = _batch_key(d)
        for builder, cases in probeset.cases_by_builder(d):
            code = probeset.code_key(builder)
            for case in cases:
                spec = _spec(
                    case.id,
                    case.probe if case.probe.startswith("testset") else f"probe {case.probe}",
                    case.level,
                    case.unit,
                    case.measured if valid else float("nan"),
                    ["batch", d.name, case.id, key, code],
                    excluded=case.excluded
                    if valid
                    else ("batch not valid" if meas.exists() else None),
                    note=f"batch {b.name}",
                )
                unit = case.id.removesuffix(" kernel")
                if unit in counters:
                    spec["counters"] = counters[unit]
                yield spec


def latest(rows: Iterator[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per id across batches: the latest batch (by name order) that measured it, else the
    latest that has it."""
    by: dict[str, dict[str, Any]] = {}
    for r in rows:
        old = by.get(r["id"])
        if old is None or r["measured"] == r["measured"] or old["measured"] != old["measured"]:
            by[r["id"]] = r
    return list(by.values())


def specs(cal: Path, sets: tuple[str, ...] = SETS) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if "gemm" in sets:
        out += gemm_specs(cal)
    if "tma" in sets:
        out += tma_specs(cal)
    if "probes" in sets:
        out += probe_specs(cal)
    if "batches" in sets:
        out += latest(batch_specs())
    return out


_cases: dict[str, probes.Case] = {}
_batch_cases: dict[tuple[str, str], probes.Case] = {}


def _regs(cal: Path) -> dict[str, str]:
    text = (cal / "ablation_cpasync" / "regs.txt").read_text()
    return dict(line.split() for line in text.splitlines() if line.strip())


def run_job(
    cal: Path,
    job: list[Any],
    machine: Machine | None = None,
    program: dict | None = None,
    trace: bool = True,
    also: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Replay one job: `predicted`, and with `trace` the critical-path attribution and the ledger.
    `program` overrides the kernel's model inputs (`h.<key>` in its tile config, `regs`); the SASS
    stays the compiled one. `also` names batch cases that share this replay; their results come
    back under "also"."""
    m = machine or default_machine()
    program = program or {}

    def pick(r: dict) -> dict:
        h = dict(r["pick"])
        h.update({k[2:]: v for k, v in program.items() if k.startswith("h.")})
        return h

    got: list[Any] = []
    kind = job[0]
    extra: dict[str, tuple[float, str | None]] = {}
    if kind in ("gemm", "gemm_block"):
        f = m.param(m.desc["clock"]["block_level_GHz"]) * 1e3
        r = json.loads((cal / "ablation_cpasync" / "mechanisms.json").read_text())[job[1]]
        h = pick(r)
        one = {"M": h["block_M"], "N": h["block_N"]} if kind == "gemm_block" else {}
        regs = int(program.get("regs", _regs(cal)[job[1]]))
        shape = {**r["shape"], **one}
        ins = parse((cal / "sass" / f"{job[1]}__{job[2]}.sass").read_text(), kernel="matmul_kernel")
        L = tile_launch(h, shape, regs, None, True, "matmul_kernel")
        kt = math.ceil(shape["K"] / h["block_K"])
        trips = machine_trips(ins) or sass_trips(ins, h, kt - (max(1, h["num_stages"]) - 1))
        got.append(generic_kernel(ins, L, regs, trips, machine=m, trace=trace))
        got[-1].run()
        predicted = got[-1].cycles / f
        if kind == "gemm":
            predicted += m.C["launch_stream_us"]
            extra["launch"] = (m.param(m.desc["launch_us"]) * f, m.desc["launch_us"])
    elif kind in ("tma", "tma_block"):
        f = m.param(m.desc["clock"]["kernel_GHz"]) * 1e3
        r = json.loads((cal / "ablation" / "mechanisms.json").read_text())[job[1]]
        if kind == "tma":
            predicted = ws.kernel_us(
                cal, job[1], pick(r), r["shape"], machine=m, sm_hook=got.append, trace=trace
            )
            extra["launch"] = (m.param(m.desc["launch_us"]) * f, m.desc["launch_us"])
        else:
            res = ws.ws_sim(
                cal,
                job[1],
                "noload",
                r["pick"],
                r["shape"],
                machine=m,
                sm_hook=got.append,
                trace=trace,
            )
            predicted = res["us"]
    elif kind == "probe":
        if not _cases:
            _cases.update((c.id, c) for c in probes.all_cases(cal))
        predicted, sm, _ = probes.run_case(_cases[job[1]], machine=m, trace=trace)
        got.append(sm)
    elif kind == "batch":
        if (job[1], job[2]) not in _batch_cases:
            d = BATCHES / job[1]
            _batch_cases.update(((job[1], c.id), c) for c in probeset.all_cases(d))
        ids = (job[2], *also)
        cases = [_batch_cases[(job[1], i)] for i in ids]
        values, sm, _ = probeset.run_cases(cases, machine=m, trace=trace)
        out = {}
        for i, predicted in zip(ids, values, strict=True):
            extra = {}
            if i.endswith(" kernel"):
                ref = m.desc.get("launch_event_us", m.desc["launch_us"])
                k = m.param(m.desc["clock"]["kernel_GHz"])
                extra["launch"] = (m.param(ref) * k * 1e3, ref)
            out[i] = _result(predicted, sm, extra, trace)
        return {**out[job[2]], "also": {i: out[i] for i in also}}
    else:
        raise ValueError(f"unknown job {job}")
    return _result(predicted, got[-1], extra, trace)


def _dynamic_ops(sm: Any) -> dict[str, list[Any]]:
    """Per full opcode: instructions issued, its class, and whether a default rule times it (the
    default class: an issue slot, no pipe)."""
    out: dict[str, list[Any]] = {}
    for x, cls, n in zip(sm.ins, sm.icls, sm.pc_count, strict=True):
        if n:
            d = out.setdefault(x.op, [0, cls.name, cls.name == "other"])
            d[0] += n
    return out


def _result(
    predicted: float, sm: Any, extra: dict[str, tuple[float, str | None]], trace: bool
) -> dict[str, Any]:
    a = attribution.analyse(sm, sm.cycles, extra) if trace else None
    return {
        "predicted": predicted,
        "counts": {
            "issued": sm.count["issued"],
            "HMMA": sm.count.get("HMMA", 0),
            "LDGSTS": sm.count.get("LDGSTS", 0),
            "ldg_sectors": sm.count.get("ldg_sectors", 0),
            "ldg_miss_sectors": sm.count.get("ldg_miss_sectors", 0),
            "lane_track": getattr(sm, "lane_track", False),
            "blocks": sm.blocks,
            "busy": {k: v / max(1.0, sm.cycles) for k, v in sm.busy.items()},
            "ops": _dynamic_ops(sm),
        },
        "attribution": [
            {k: d[k] for k in ("resource", "mechanism", "share", "param")}
            for d in a["path"]
            if d["share"] >= 0.005
        ]
        if a
        else [],
        "coverage": (a["covered_cycles"] / a["cycles"] if a["cycles"] else 1.0) if a else None,
        "defaults": attribution.default_counts(a["ledger"]) if a else {},
        "ledger": a["ledger"] if a else [],
    }


def engine_hash() -> str:
    """sha256 over what every replay shares: the engine's code except probeset.py (`job` carries
    the code of a row's own case builder, probeset.code_key), the description, the parameter file
    without the `sim` values (a row compares those it read, `_read_values`) and the replay functions
    of this file; a change to the report or the scorecard replays nothing."""
    h = hashlib.sha256()
    files = sorted(
        p for p in ENGINE.iterdir() if p.suffix in (".py", ".yaml") and p.name != "probeset.py"
    )
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    raw = json.loads((ENGINE / "gb10.json").read_text())
    raw["sim"] = {k: v for k, v in raw["sim"].items() if not k.startswith("value")}
    h.update(json.dumps(raw, sort_keys=True).encode())
    for f in (run_job, _regs, _result, _dynamic_ops, _timed, _read_values):
        h.update(inspect.getsource(f).encode())
    return h.hexdigest()


def _read_values(m: Machine, reads: list[str]) -> dict[str, Any]:
    """What the parameters in `reads` hold on `m`: a `sim` entry with its status, which attribution
    reports. An absent entry reads None. "*" (the whole table was read) is a digest of it."""
    sim = m.raw["sim"]
    out: dict[str, Any] = {}
    for k in reads:
        if k == "*":
            out[k] = hashlib.sha256(json.dumps(sim["value"], sort_keys=True).encode()).hexdigest()
        elif k.startswith("sim."):
            out[k] = [sim["value"].get(k[4:]), sim["value_status"].get(k[4:])]
        else:
            out[k] = m.raw.get(k, {}).get("value")
    return json.loads(json.dumps(out))


class _OverBudget(BaseException):  # not an Exception: no handler in the engine takes it for its own
    pass


def _over_budget(*_: Any) -> None:
    raise _OverBudget


def _timed(
    cal: Path,
    job: list[Any],
    overrides: dict[str, Any] | None = None,
    trace: bool = True,
    also: tuple[str, ...] = (),
    budget_s: float = BUDGET_S,
) -> dict[str, Any]:
    t, cpu = time.perf_counter(), time.process_time()
    m = (default_machine().with_overrides(overrides) if overrides else default_machine()).tracked()
    signal.signal(signal.SIGVTALRM, _over_budget)
    signal.setitimer(signal.ITIMER_VIRTUAL, budget_s)
    try:
        out = run_job(cal, job, m, trace=trace, also=also)
    except _OverBudget:
        out = {"predicted": None, "excluded": f"over budget: replay above {budget_s:.0f} CPU s"}
        out["budget_s"] = budget_s
    finally:
        signal.setitimer(signal.ITIMER_VIRTUAL, 0)
    reads = sorted(m.__dict__["reads"])
    wall = round((time.perf_counter() - t) / (1 + len(also)), 2)
    extra = {
        "reads": reads,
        "values": _read_values(m, reads),
        "wall_s": wall,
        "cpu_s": round((time.process_time() - cpu) / (1 + len(also)), 2),
    }
    for o in (out, *out.get("also", {}).values()):
        o.update(extra)
    return out


def _worker_init() -> None:
    """Pool worker: lowest priority, and exit when the table process is gone (killed or stopped),
    so that no worker outlives it."""
    os.nice(19)
    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(2)
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def _replay_group(r: dict[str, Any]) -> tuple[Any, ...]:
    """Rows that one replay serves: the launch modes of one test kernel."""
    job = r["job"]
    m = re.fullmatch(r"(.*) (kernel|b2b|graph)", job[2]) if job[0] == "batch" else None
    return (job[1], m[1]) if m else (r["id"],)


# replays and CPU of this process's `run` calls, for the run log
STATS = {"replayed": 0, "cpu_s": 0.0}


def run(
    cal: Path,
    upstream_rev: str,
    rows: list[dict[str, Any]],
    jobs: int,
    use_cache: bool = True,
    hint: dict[str, float] | None = None,
    run_slow: bool = False,
    cached_only: bool = False,
    overrides: dict[str, Any] | None = None,
    trace: bool = True,
    budget_s: float = BUDGET_S,
) -> Iterator[dict[str, Any]]:
    """Fill each spec with its result, cached ones first, the rest from a pool of `jobs` processes
    (longest previous runs first); yields rows as they complete. `trace=False` skips the
    attribution: faster replays, for ablations. A replay past `budget_s` CPU seconds comes back
    excluded, predicted None."""
    eng = engine_hash()
    CACHE.mkdir(parents=True, exist_ok=True)
    machine = default_machine().with_overrides(overrides) if overrides else default_machine()

    def key(r: dict[str, Any], traced: bool) -> Path:
        k = [eng, upstream_rev, r["job"]] + ([] if traced else ["fast"])
        return CACHE / hashlib.sha256(json.dumps(k).encode()).hexdigest()

    def lookup(r: dict[str, Any]) -> dict[str, Any] | None:
        for traced in (True,) if trace else (False, True):
            d = key(r, traced)
            for p in sorted(d.iterdir()) if d.is_dir() else ():
                res = json.loads(p.read_text())
                if res.get("budget_s", budget_s) < budget_s:
                    continue
                if _read_values(machine, res["reads"]) == res["values"]:
                    return res
        return None

    def store(r: dict[str, Any], res: dict[str, Any]) -> None:
        d = key(r, trace)
        d.mkdir(exist_ok=True)
        name = hashlib.sha256(json.dumps(res["values"], sort_keys=True).encode()).hexdigest()[:16]
        (d / f"{name}.json").write_text(json.dumps(res))

    todo = []
    for r in rows:
        res = lookup(r) if use_cache else None
        if res is not None:
            r.update(res, cached=True)
            yield _finish(r)
        elif r["id"] in SLOW and not run_slow:
            r.update(predicted=None, excluded=SLOW[r["id"]], cached=False, wall_s=0.0)
            yield _finish(r)
        else:
            todo.append(r)
    if cached_only and todo:
        raise SystemExit(f"{len(todo)} rows not in the cache, e.g. {todo[0]['id']!r}")
    todo.sort(key=lambda r: -(hint or {}).get(r["id"], 0.0))
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in todo:
        groups.setdefault(_replay_group(r), []).append(r)
    if not groups:
        return
    with ProcessPoolExecutor(max_workers=max(1, jobs), initializer=_worker_init) as pool:
        futs = {
            pool.submit(
                _timed,
                cal,
                g[0]["job"],
                overrides,
                trace,
                tuple(r["job"][2] for r in g[1:]),
                budget_s,
            ): g
            for g in groups.values()
        }
        for f in as_completed(futs):
            g = futs[f]
            lead = f.result()
            also = lead.pop("also", {})
            STATS["replayed"] += 1
            STATS["cpu_s"] += lead["cpu_s"] * len(g)
            for r in g:
                res = also.get(r["job"][2], lead) if r is not g[0] else lead
                store(r, res)
                r.update(res, cached=False)
                yield _finish(r)


def _finish(r: dict[str, Any]) -> dict[str, Any]:
    ok = r["measured"] == r["measured"] and r["measured"] > 0
    ok = ok and r["predicted"] is not None
    r["error"] = r["predicted"] / r["measured"] - 1 if ok else None
    return r


def batch_order(name: str) -> tuple[int, str]:
    """Batches in the order they ran: r2c before r10."""
    return (int(re.sub(r"\D", "", name)), name)


def first_scores(root: Path = BATCHES) -> dict[str, dict[str, Any]]:
    """Per row, the first score it ever had (LOOP §5, blind error): the prediction committed before
    the batch that first measured it, against that measurement. Later rows' engines have seen it."""
    out: dict[str, dict[str, Any]] = {}
    done = (p for p in root.iterdir() if (p / "predicted.json").exists())
    for d in sorted(done, key=lambda p: batch_order(p.name)):
        mp = d / "measured.json"
        if not mp.exists():
            continue
        doc = json.loads(mp.read_text())
        if not doc.get("valid"):
            continue
        pred = json.loads((d / "predicted.json").read_text())["predicted"]
        for cid, v in doc["results"].items():
            if cid not in out and cid in pred and v["median"] > 0:
                out[cid] = {
                    "batch": d.name,
                    "predicted": pred[cid],
                    "measured": v["median"],
                    "error": pred[cid] / v["median"] - 1,
                }
    return out


def inversions(rs: list[dict[str, Any]], noise: float = 0.02) -> int:
    """Pairs of design variants the model orders the other way round from the measurement (pairs
    whose measured times differ by less than `noise` are not counted)."""
    n = 0
    for i, a in enumerate(rs):
        for b in rs[i + 1 :]:
            if abs(a["measured"] / b["measured"] - 1) < noise:
                continue
            if (a["measured"] - b["measured"]) * (a["predicted"] - b["predicted"]) < 0:
                n += 1
    return n


def kendall(x: list[float], y: list[float]) -> float:
    """Kendall tau-b of predicted against measured (28 §1)."""
    con = dis = tx = ty = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            a, b = x[i] - x[j], y[i] - y[j]
            if a == 0 or b == 0:
                tx += a == 0
                ty += b == 0
                continue
            con += a * b > 0
            dis += a * b < 0
    n = con + dis
    return (con - dis) / math.sqrt((n + tx) * (n + ty)) if n else 0.0


def best_variant(rs: list[dict[str, Any]], noise: float = 0.02) -> dict[str, Any]:
    """Does the model's fastest variant measure as the fastest? `gap`: how much slower the model's
    pick measures than the true best; within `noise` it counts as identified."""
    pick = min(rs, key=lambda r: r["predicted"])
    true = min(rs, key=lambda r: r["measured"])
    gap = pick["measured"] / true["measured"] - 1
    return {"ok": gap < noise, "gap": gap, "pick": pick["id"], "best": true["id"]}


# the good-enough targets of 28 §1 (D828): pairs at least this far apart must be ordered right
PAIR_GAP = 0.03
TOP_N = 3


def goal(rs: list[dict[str, Any]]) -> dict[str, Any]:
    """28 §1 on one group: pairs measured `PAIR_GAP` or more apart that the model orders right, the
    true best inside the model's first `TOP_N`, and what picking the model's first loses."""
    right = total = 0
    for i, a in enumerate(rs):
        for b in rs[i + 1 :]:
            if abs(a["measured"] / b["measured"] - 1) >= PAIR_GAP:
                total += 1
                right += (a["measured"] - b["measured"]) * (a["predicted"] - b["predicted"]) > 0
    order = sorted(rs, key=lambda r: r["predicted"])
    true = min(rs, key=lambda r: r["measured"])
    return {
        "pairs_right": right,
        "pairs": total,
        "best_in_top": true in order[:TOP_N],
        "first_loss": order[0]["measured"] / true["measured"] - 1,
    }


def _ranks(rs: list[dict[str, Any]]) -> dict[str, Any]:
    p, m = [r["predicted"] for r in rs], [r["measured"] for r in rs]
    return {
        "spearman": spearman(p, m),
        "kendall": kendall(p, m),
        "inversions": inversions(rs),
        "best": best_variant(rs),
        "goal": goal(rs),
    }


def measurements(root: Path = BATCHES) -> dict[str, list[dict[str, Any]]]:
    """Per row, every valid batch's measurement of it, in batch order."""
    out: dict[str, list[dict[str, Any]]] = {}
    for d in sorted(root.iterdir(), key=lambda p: batch_order(p.name)):
        mp = d / "measured.json"
        if not mp.exists():
            continue
        doc = json.loads(mp.read_text())
        if not doc.get("valid"):
            continue
        for cid, v in doc["results"].items():
            if v.get("median", 0) > 0:
                out.setdefault(cid, []).append({"batch": d.name, **v})
    return out


def _spread(ms: list[dict[str, Any]]) -> tuple[float, float]:
    """A row's measured range: its batches' medians, each widened by half its IQR."""
    return (
        min(m["median"] - m.get("iqr", 0.0) / 2 for m in ms),
        max(m["median"] + m.get("iqr", 0.0) / 2 for m in ms),
    )


def repeatability(
    rs: list[dict[str, Any]], meas: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """The measurement's own ranking limit (round 9), a record beside τ: τ between every two batches
    that measured the whole group, or between odd and even samples when one batch did; and how many
    of the pairs the model orders the other way lie inside the rows' measured spread. The raw τ and
    the 0.8 target are not changed by it."""
    ids = [r["id"] for r in rs]
    per = {i: {m["batch"]: m for m in meas.get(i, [])} for i in ids}
    common = [b for b in (per[ids[0]] if ids else {}) if all(b in per[i] for i in ids)]
    taus = [
        kendall([per[i][a]["median"] for i in ids], [per[i][b]["median"] for i in ids])
        for x, a in enumerate(common)
        for b in common[x + 1 :]
    ]
    how = "batches"
    if not taus and common:
        odd, even = [], []
        for i in ids:
            sm = per[i][common[-1]].get("samples") or []
            odd.append(statistics.median(sm[0::2]) if len(sm) > 1 else float("nan"))
            even.append(statistics.median(sm[1::2]) if len(sm) > 1 else float("nan"))
        if all(x == x for x in odd + even):
            taus, how = [kendall(odd, even)], "samples"
    within = above = 0
    for i, a in enumerate(rs):
        for b in rs[i + 1 :]:
            if (a["measured"] - b["measured"]) * (a["predicted"] - b["predicted"]) >= 0:
                continue
            la, ha = _spread(meas.get(a["id"]) or [{"median": a["measured"]}])
            lb, hb = _spread(meas.get(b["id"]) or [{"median": b["measured"]}])
            if la <= hb and lb <= ha:
                within += 1
            else:
                above += 1
    item: dict[str, Any] = {
        "within_spread": within,
        "above_spread": above,
        "n_repeats": len(common),
    }
    if taus:
        item.update(
            tau_median=statistics.median(taus), tau_min=min(taus), tau_max=max(taus), how=how
        )
    return item


def _design_group(group: str) -> bool:
    """Design variants of one shape (the what-if sets are one change each, ablations among them)."""
    return group.endswith("full") or (group.startswith("testset") and "whatif" not in group)


# R1 (28 §4): instructions per block executed by the model's replay vs the profiler's count
R1_TOL = 0.02


def counts_check(r: dict[str, Any]) -> dict[str, Any]:
    c, k = r["counters"], r.get("counts") or {}
    grid = float(c.get("launch__grid_size") or 1)
    meas = float(c.get("smsp__inst_executed.sum") or 0) / grid
    model = k.get("issued", 0) / max(1, k.get("blocks", 1))
    mt = float(c.get("smsp__inst_executed_pipe_tensor.sum") or 0) / grid
    kt = k.get("HMMA", 0) / max(1, k.get("blocks", 1))
    ok = bool(meas) and abs(model / meas - 1) <= R1_TOL and (not mt or abs(kt / mt - 1) <= R1_TOL)
    out = {"inst_model": model, "inst_meas": meas, "mma_model": kt, "mma_meas": mt, "ok": ok}
    # round 3: global-load sectors per block where the replay counts them from the lanes'
    # addresses (LDG only; cp.async sectors are counted by the copy front end, not here), and the
    # L1 misses against the L2 read sectors (a record, not part of R1: ncu's replay refills the L2)
    if k.get("lane_track") and not k.get("LDGSTS"):
        blocks = max(1, k.get("blocks", 1))
        sm_, sx = k["ldg_sectors"] / blocks, float(c.get(SECT_LD) or 0) / grid
        out.update(sect_model=sm_, sect_meas=sx)
        out["sect_ok"] = bool(sx) and abs(sm_ / sx - 1) <= R1_TOL
        out["ok"] = ok and out["sect_ok"]
        out["l2_read_model"] = k["ldg_miss_sectors"] / blocks
        out["l2_read_meas"] = float(c.get("lts__t_sectors_op_read.sum") or 0) / grid
    return out


SECT_LD = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"


# counter pipes and the model resources they stand for (bottleneck agreement, LOOP §5)
PIPES = {
    "sm__pipe_tensor_cycles_active.sum": "tensor",
    "sm__pipe_fma_cycles_active.sum": "fma",
    "sm__pipe_alu_cycles_active.sum": "alu",
}


def bottleneck(r: dict[str, Any]) -> dict[str, Any]:
    """The busiest counter pipe against the model's top critical-path resource."""
    c = r["counters"]
    busiest = max(PIPES, key=lambda m: float(c.get(m) or 0))
    top = r["attribution"][0]["resource"] if r.get("attribution") else None
    return {"busiest_pipe": PIPES[busiest], "model_top": top, "agree": top == PIPES[busiest]}


def dynamic_weights(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Opcodes by share of the instructions issued, per kernel row, over the `kernel` rows of the
    test set: the mean and the largest share; `default`: a default rule times the opcode."""
    per: dict[str, dict[str, Any]] = {}
    kernels = [
        r for r in rows if r["id"].endswith(" kernel") and (r.get("counts") or {}).get("ops")
    ]
    for r in kernels:
        ops = r["counts"]["ops"]
        total = sum(d[0] for d in ops.values()) or 1
        for op, (n, cls, dflt) in ops.items():
            e = per.setdefault(op, {"op": op, "class": cls, "default": dflt, "sum": 0.0, "max": 0})
            e["sum"] += n / total
            e["max"] = max(e["max"], n / total)
    out = [
        {**{k: e[k] for k in ("op", "class", "default", "max")}, "mean": e["sum"] / len(kernels)}
        for e in per.values()
    ]
    return sorted(out, key=lambda e: -e["mean"])


UNITS = {  # model busy key -> (counter, instances per SM the counter sums over)
    "tensor": ("sm__pipe_tensor_cycles_active.sum", 1),
    "fma": ("sm__pipe_fma_cycles_active.sum", 1),
    "alu": ("sm__pipe_alu_cycles_active.sum", 1),
    "fill": ("lts__lts2xbar_cycles_active.sum", 1),
    "lsu": ("l1tex__m_l1tex2xbar_req_cycles_active.sum", 1),
}
STALLS = {
    "pipes": ("math_pipe_throttle", "short_scoreboard"),
    "memory": ("long_scoreboard", "mio_throttle", "lg_throttle"),
    "sync": ("barrier", "wait"),
}


def unit_compare(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Round 10 (LOOP §5, bottleneck): per unit the model's busy cycles against the counter's, both
    over the launch's cycles on the mean SM; per stall family the counters' share of warp-cycles
    against the model's critical-path share. Counters per launch, over all 48 SMs."""
    per: dict[str, list[float]] = {u: [] for u in UNITS}
    fam: dict[str, list[tuple[float, float]]] = {k: [] for k in STALLS}
    worst: list[dict[str, Any]] = []
    for r in rows:
        c = r.get("counters") or {}
        active = float(c.get("smsp__warps_active.sum") or 0)
        busy = (r.get("counts") or {}).get("busy")
        if not active or not busy:
            continue
        grid = float(c.get("launch__grid_size") or 1)
        blocks = max(1, r["counts"].get("blocks", 1))
        cyc = r["predicted"] * C["clock_under_load_GHz"] * 1e3
        for u, (key, _) in UNITS.items():
            if key in c and float(c[key]) and u in busy:
                meas = float(c[key]) / grid  # per block
                model = busy[u] * cyc / blocks
                per[u].append(model / meas)
                worst.append(
                    {"row": r["id"], "unit": u, "model": model, "counter": meas, "r": model / meas}
                )
        share = {"pipes": 0.0, "memory": 0.0, "sync": 0.0}
        for a in r.get("attribution", []):
            k = _unit_kind(a["resource"])
            share[
                "pipes" if k == "compute" else "memory" if k in ("loads", "stores") else "sync"
            ] += a["share"]
        total = sum(
            float(c.get(f"smsp__warps_issue_stalled_{x}.sum") or 0)
            for v in STALLS.values()
            for x in v
        )
        for k, names in STALLS.items():
            m = sum(float(c.get(f"smsp__warps_issue_stalled_{x}.sum") or 0) for x in names)
            fam[k].append((m / total if total else 0.0, share[k]))
    worst.sort(key=lambda d: -abs(math.log(max(d["r"], 1e-9))))
    return {
        "units": {u: statistics.median(v) for u, v in per.items() if v},
        "n": {u: len(v) for u, v in per.items()},
        "worst": worst[:10],
        "stalls": {
            k: (statistics.mean(x for x, _ in v), statistics.mean(y for _, y in v))
            for k, v in fam.items()
            if v
        },
    }


# Convergence record (LOOP §5, round 10): per kernel class, in the order first scored, the new
# mechanisms the class needed, counted by hand from docs/decisions.md and the inventory; the worst
# first-score error comes from the rows.
CLASSES = (
    ("HGEMM (TileLang)", r"^hgemm", 9),
    ("FA forward", r"^fa_", 4),
    ("decode attention", r"^decode_", 3),
    ("w4a16 GEMV", r"^gemv_", 3),
    ("RMSNorm, softmax", r"^(rmsnorm|softmax)_", 5),
    ("FP8 GEMM", r"^fp8gemm", 0),
    ("MoE GEMM", r"^moe_", 1),
    ("w4a16 GEMM", r"^w4gemm", 1),
    ("NVFP4 GEMM", r"^nvfp4gemm", 1),
    ("hand-written CUDA", r"^hand_", 5),
    ("generated mixes", r"^stress_", 0),
    ("library operators", r"^lib_", 0),
)


def convergence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for name, pat, needed in CLASSES:
        rs = [
            r
            for r in rows
            if r["id"].endswith(" kernel")
            and re.search(pat, r["id"])
            and r.get("blind") is not None
        ]
        if rs:
            worst = max(rs, key=lambda r: abs(r["blind"]))
            out.append(
                {
                    "class": name,
                    "first_batch": min((r["blind_batch"] for r in rs), key=batch_order),
                    "n": len(rs),
                    "worst_first": worst["blind"],
                    "median_first": statistics.median(abs(r["blind"]) for r in rs),
                    "mechanisms": needed,
                }
            )
    return out


def default_jobs() -> int:
    return min(LOCAL_JOBS, max(1, (os.cpu_count() or 2) - 2))


def summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    meas = measurements()
    for group in dict.fromkeys(r["set"] for r in rows):
        rs = [r for r in rows if r["set"] == group and not r["excluded"]]
        if not rs:
            continue
        e = [r["error"] for r in rs]
        worst = max(e, key=abs)
        item: dict[str, Any] = {
            "set": group,
            "n": len(rs),
            "excluded": sum(1 for r in rows if r["set"] == group and r["excluded"]),
            "median_abs": statistics.median(abs(x) for x in e),
            "worst": worst,
            "within_10": sum(abs(x) <= 0.1 for x in e),
        }
        em = [r["error_mech"] for r in rs if r.get("error_mech") is not None]
        if em:
            item["median_abs_mech"] = statistics.median(abs(x) for x in em)
            item["worst_mech"] = max(em, key=abs)
        if len(rs) >= 3 and _design_group(group):
            ranks = _ranks(rs)
            item.update(
                spearman=ranks["spearman"],
                inversions=ranks["inversions"],
                kendall=ranks["kendall"],
                best=ranks["best"],
                goal=ranks["goal"],
                repeat=repeatability(rs, meas),
            )
            if len(em) == len(rs):
                rm = [{**r, "predicted": r["predicted_mech"]} for r in rs]
                item["spearman_mech"] = spearman(
                    [r["predicted"] for r in rm], [r["measured"] for r in rm]
                )
                item["inversions_mech"] = inversions(rm)
        b = [r["blind"] for r in rs if r.get("blind") is not None]
        if b:
            item["blind_median"] = statistics.median(abs(x) for x in b)
            item["blind_worst"] = max(b, key=abs)
        r1 = [counts_check(r) for r in rs if r.get("counters")]
        if r1:
            item["r1_pass"] = sum(x["ok"] for x in r1)
            item["r1_n"] = len(r1)
            bn = [bottleneck(r) for r in rs if r.get("counters")]
            item["bottleneck_agree"] = sum(x["agree"] for x in bn)
        out.append(item)
    return out


def blind_summary(rows: list[dict[str, Any]], batch: str) -> list[dict[str, Any]]:
    """The rows `batch`'s round measured first, by set, at their first score (no engine had seen
    them)."""
    out = []
    round_no = batch_order(batch)[0]  # r9, r9b, r9c are one round
    picked = [
        r
        for r in rows
        if r.get("blind_batch")
        and batch_order(r["blind_batch"])[0] == round_no
        and not r["excluded"]
    ]
    for group in dict.fromkeys(r["set"] for r in picked):
        rs = [r for r in picked if r["set"] == group]
        b = [r["blind"] for r in rs]
        item: dict[str, Any] = {
            "set": group,
            "n": len(rs),
            "median_abs": statistics.median(abs(x) for x in b),
            "worst": max(b, key=abs),
            "now_median": statistics.median(abs(r["error"]) for r in rs),
            "now_worst": max((r["error"] for r in rs), key=abs),
        }
        if len(rs) >= 3 and _design_group(group):
            first = _ranks([{**r, "predicted": r["blind_predicted"]} for r in rs])
            now = _ranks(rs)
            item.update(first)
            item.update(
                now_spearman=now["spearman"],
                now_inversions=now["inversions"],
                now_kendall=now["kendall"],
                now_best=now["best"],
            )
        out.append(item)
    return out


# what-if scoring (28 §1, round 7): a change is scored against the base row of its own batch
TIE, MAG_MIN = 0.02, 0.05
GOAL_DIR, GOAL_MAG = 0.03, 0.10  # 28 §1: direction on changes of 3 %, magnitude on changes of 10 %
# round 8: one event timing of a short kernel steps by about 0.96 us (the same RMSNorm 8.6 or 9.6 us
# across batches); the graph row averages 50 launches and is scored beside it
SHORT_US = 30.0
ABLATIONS = {
    "compute removed": "compute",
    "loads removed": "loads",
    "stores removed": "stores",
}
# critical-path resources by what removing them means
LOAD_UNITS = ("l2", "dram", "lsu", "tma", "cp", "fill", "copy")
STORE_UNITS = ("store",)


def _unit_kind(resource: str) -> str:
    if any(resource.startswith(u) for u in STORE_UNITS):
        return "stores"
    if any(resource.startswith(u) for u in LOAD_UNITS):
        return "loads"
    return "compute"


def whatif_pairs(rows: list[dict[str, Any]], root: Path = BATCHES) -> list[dict[str, Any]]:
    """One record per variant marked `whatif` in its launch.json: predicted against measured delta
    from the `kernel` rows, at first score and now."""
    by_id = {r["id"]: r for r in rows}
    out: dict[str, dict[str, Any]] = {}  # a variant measured again keeps one record, the latest
    for d in sorted(root.iterdir(), key=lambda p: batch_order(p.name)):
        for lj in sorted((d / "build" / "testset").glob("*/launch.json")):
            w = json.loads(lj.read_text()).get("whatif")
            if not w:
                continue
            a, b = by_id.get(f"{w['base']} kernel"), by_id.get(f"{lj.parent.name} kernel")
            if not a or not b or a["excluded"] or b["excluded"]:
                continue
            meas = b["measured"] / a["measured"] - 1
            first = (b.get("blind_predicted") or 0) / (a.get("blind_predicted") or 1) - 1
            now = b["predicted"] / a["predicted"] - 1
            ga, gb = by_id.get(f"{w['base']} graph"), by_id.get(f"{lj.parent.name} graph")
            graph = None
            if ga and gb and a["measured"] < SHORT_US:
                gm, gn = gb["measured"] / ga["measured"] - 1, gb["predicted"] / ga["predicted"] - 1
                graph = {"measured": gm, "now": gn, "ok": _direction(gn, gm)}
            out[lj.parent.name] = {
                "graph": graph,
                "id": lj.parent.name,
                "base": w["base"],
                "axis": w["axis"],
                "measured": meas,
                "first": first,
                "now": now,
                "first_ok": _direction(first, meas),
                "now_ok": _direction(now, meas),
            }
    return list(out.values())


def _direction(pred: float, meas: float) -> bool:
    if abs(meas) < TIE:
        return abs(pred) < 2 * TIE  # no real change measured: the model must not claim one
    return pred * meas > 0


def _magnitude(p: dict[str, Any], key: str) -> float | None:
    if abs(p["measured"]) < MAG_MIN:
        return None
    return abs(p[key] - p["measured"]) / abs(p["measured"])


def whatif_summary(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for axis in ["all", *dict.fromkeys(p["axis"] for p in pairs)]:
        ps = [p for p in pairs if axis in ("all", p["axis"])]
        item: dict[str, Any] = {"axis": axis, "n": len(ps)}
        for key in ("first", "now"):
            big = [p for p in ps if abs(p["measured"]) >= GOAL_DIR]
            mag = [abs(p[key] / p["measured"] - 1) for p in ps if abs(p["measured"]) >= GOAL_MAG]
            item[f"{key}_goal_dir"] = (sum(p[key] * p["measured"] > 0 for p in big), len(big))
            item[f"{key}_goal_mag"] = (statistics.median(mag) if mag else None, len(mag))
            mags = [m for p in ps if (m := _magnitude(p, key)) is not None]
            item[f"{key}_ok"] = sum(p[f"{key}_ok"] for p in ps)
            item[f"{key}_mag_n"] = len(mags)
            item[f"{key}_mag_median"] = statistics.median(mags) if mags else None
            item[f"{key}_mag_within"] = sum(m <= 0.25 for m in mags)
        out.append(item)
    return out


def bottlenecks(rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per base with ablations: the critical-path share of compute, loads and stores in the replay,
    against what removing each measures and replays."""
    by_id = {r["id"]: r for r in rows}
    out = []
    for base in dict.fromkeys(p["base"] for p in pairs if p["axis"] in ABLATIONS):
        r = by_id[f"{base} kernel"]
        share: dict[str, float] = {}
        for a in r.get("attribution", []):
            if a["resource"] == "launch":
                continue
            k = _unit_kind(a["resource"])
            share[k] = share.get(k, 0.0) + a["share"]
        gains = {
            ABLATIONS[p["axis"]]: p for p in pairs if p["base"] == base and p["axis"] in ABLATIONS
        }
        if not share:  # a run without attribution (--fast)
            continue
        top = max(share, key=lambda k: share[k])
        best = min(gains, key=lambda k: gains[k]["measured"])
        out.append(
            {
                "base": base,
                "model_top": top,
                "share": share,
                "gains": {k: (p["now"], p["measured"]) for k, p in gains.items()},
                "measured_top": best,
                "agree": top == best,
            }
        )
    return out


def summary_line(s: dict[str, Any]) -> str:
    sp = f"{s['spearman']:.2f}" if "spearman" in s else ""
    kt = f"{s['kendall']:.2f}" if "kendall" in s else ""
    bv = _best(s.get("best"))
    inv = str(s["inversions"]) if "inversions" in s else ""
    r1 = f"{s['r1_pass']}/{s['r1_n']}" if "r1_n" in s else ""
    bn = f"{s['bottleneck_agree']}/{s['r1_n']}" if "r1_n" in s else ""
    mm = f"{100 * s['median_abs_mech']:.1f} %" if "median_abs_mech" in s else ""
    mw = _pct(s["worst_mech"]) if "worst_mech" in s else ""
    ms = f"{s['spearman_mech']:.2f} / {s['inversions_mech']}" if "spearman_mech" in s else ""
    bl = ""
    if "blind_median" in s:
        bl = f"{100 * s['blind_median']:.1f} % / {_pct(s['blind_worst'])}"
    rp = s.get("repeat") or {}
    ct = ""
    if "tau_median" in rp:
        ct = f"{rp['tau_median']:.2f} ({rp['tau_min']:.2f}–{rp['tau_max']:.2f}, {rp['n_repeats']}"
        ct += " batches)" if rp["how"] == "batches" else " batch, samples)"
    sprd = f"{rp['within_spread']} / {rp['above_spread']}" if rp else ""
    return (
        f"| {s['set']} | {s['n']} | {s['excluded']} | {100 * s['median_abs']:.1f} % | "
        f"{_pct(s['worst'])} | {mm} | {mw} | {bl} | {s['within_10']} | {sp} | {kt} | {ct} | {bv} | "
        f"{inv} | {sprd} | {ms} | {r1} | {bn} |"
    )


def _best(b: dict[str, Any] | None) -> str:
    if not b:
        return ""
    return "yes" if b["gap"] == 0 else f"{'yes' if b['ok'] else 'no'} (+{100 * b['gap']:.1f} %)"


def _fmt(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:,.2f}" if abs(x) < 1e6 else f"{x:,.0f}"


def _pct(x: float | None) -> str:
    return "—" if x is None or math.isnan(x) else f"{100 * x:+.1f} %"


def render(doc: dict[str, Any], top: int = 15) -> str:
    rows = doc["rows"]
    replay = sum(r["wall_s"] for r in rows)
    slowest = max(r["wall_s"] for r in rows)
    lines = [
        "# Error table",
        "",
        f"Produced by `scripts/perfmodel-table` from KTA commit `{doc['kta_commit']}`"
        f"{' (dirty)' if doc['kta_dirty'] else ''} on {doc['date']}; upstream inputs "
        f"`{doc['upstream']}` at `{doc['upstream_rev']}`. CPU only: replay time {replay:.0f} s "
        f"summed over rows (slowest row {slowest:.0f} s); this run {doc['wall_s']:.0f} s with "
        f"{doc['jobs']} processes, {doc['cached_rows']} rows from the cache. "
        "Error = predicted / measured − 1 (positive: the model is slow). "
        "Machine-readable copy: `error-table.json`. Mechanisms only (LOOP §4.4): every fitted "
        f"parameter replaced by its probe value or removed: {MECH_ONLY}.",
        "",
        "## Summary",
        "",
        "| Set | n | excluded | median abs error | worst | median, mechanisms only | "
        "worst, mechanisms only | blind median / worst | within 10 % | Spearman | Kendall τ | "
        "τ between repeats (ceiling, a record) | best variant | inversions | "
        "reversed pairs within / above the measured spread | "
        "Spearman / inversions, mechanisms only | counts check (R1) | bottleneck agrees |",
        "| --- " + "| ---: " * 17 + "|",
    ]
    for s in doc["summary"]:
        lines.append(summary_line(s))
    if doc.get("blind"):
        lines += [
            "",
            f"## Blind rows of batch {doc['blind_batch']}",
            "",
            "First score: the prediction committed before the batch, engine unchanged. Now: "
            "the current engine on the same rows, no longer blind.",
            "",
            "| Set | n | first: median | first: worst | first: Kendall τ / Spearman / "
            "inversions | first: best variant | now: median | now: worst | now: Kendall τ / "
            "Spearman / inversions | now: best variant |",
            "| --- " + "| ---: " * 9 + "|",
        ]
        for s in doc["blind"]:
            f = n = ""
            if "spearman" in s:
                f = f"{s['kendall']:.2f} / {s['spearman']:.2f} / {s['inversions']}"
                n = f"{s['now_kendall']:.2f} / {s['now_spearman']:.2f} / {s['now_inversions']}"
            lines.append(
                f"| {s['set']} | {s['n']} | {100 * s['median_abs']:.1f} % | {_pct(s['worst'])} | "
                f"{f} | {_best(s.get('best'))} | {100 * s['now_median']:.1f} % | "
                f"{_pct(s['now_worst'])} | {n} | {_best(s.get('now_best'))} |"
            )
    lines += _goal_lines(doc)
    lines += _whatif_lines(doc)
    lines += _dynamic_lines(doc)
    lines += _unit_lines(doc)
    lines += _convergence_lines(doc)
    ranked = sorted((r for r in rows if not r["excluded"]), key=lambda r: -abs(r["error"]))
    lines += ["", f"## Top {top} rows by absolute error", ""]
    lines += _table(ranked[:top])
    lines += ["", "## All rows", ""]
    lines += _table(rows)
    return "\n".join(lines) + "\n"


def _design(group: str) -> bool:
    """Groups of design variants: the breadth sets and the probe-like kernel sets are not."""
    return not any(k in group for k in ("breadth", "wr_", "atomic_bins", "store_rewrite"))


def goal_totals(doc: dict[str, Any]) -> dict[str, Any]:
    """28 §1 over the design groups of the test set, and the what-if numbers."""
    gs = [(s["set"], s["goal"]) for s in doc["summary"] if "goal" in s and _design(s["set"])]
    wi = next((s for s in doc.get("whatif_summary", []) if s["axis"] == "all"), None)
    return {
        "groups": len(gs),
        "pairs_right": sum(g["pairs_right"] for _, g in gs),
        "pairs": sum(g["pairs"] for _, g in gs),
        "top_missed": [n for n, g in gs if not g["best_in_top"]],
        "loss_above": [(n, g["first_loss"]) for n, g in gs if g["first_loss"] > PAIR_GAP],
        "worst_loss": max((g["first_loss"] for _, g in gs), default=0.0),
        "whatif_dir": wi and wi["now_goal_dir"],
        "whatif_mag": wi and wi["now_goal_mag"],
    }


def _goal_lines(doc: dict[str, Any]) -> list[str]:
    t = goal_totals(doc)
    if not t["groups"]:
        return []

    def pct(a: int, b: int) -> str:
        return f"{100 * a / max(1, b):.1f} %"

    lines = [
        "",
        "## Against 28 §1 (targets are the user's; a seen group meeting them is not enough)",
        "",
        f"| Metric | Target | Now ({t['groups']} design groups) |",
        "| --- | ---: | --- |",
        f"| Pairs measured ≥ 3 % apart, ordered right | ≥ 95 % | {t['pairs_right']}/{t['pairs']} "
        f"({pct(t['pairs_right'], t['pairs'])}) |",
        f"| True best in the model's top {TOP_N} | every group | "
        f"{t['groups'] - len(t['top_missed'])}/{t['groups']} |",
        f"| First pick's loss | ≤ 3 % | worst {100 * t['worst_loss']:.1f} %, "
        f"{len(t['loss_above'])} groups above |",
    ]
    if t["whatif_dir"]:
        (r, n), (m, k) = t["whatif_dir"], t["whatif_mag"]
        mag = "—" if m is None else f"{100 * m:.0f} %"
        lines += [
            f"| What-if direction, changes ≥ 3 % | ≥ 95 % | {r}/{n} ({pct(r, n)}) |",
            f"| What-if magnitude, changes ≥ 10 % | median ≤ 20–25 % | {mag} over {k} |",
        ]
    miss = [f"{n}: +{100 * g:.1f} %" for n, g in t["loss_above"]]
    if miss or t["top_missed"]:
        lines += ["", f"Loss above 3 %: {', '.join(miss) or 'none'}."]
        lines += [f"Best outside the top {TOP_N}: {', '.join(t['top_missed']) or 'none'}."]
    return lines


def _convergence_lines(doc: dict[str, Any]) -> list[str]:
    rows = doc.get("convergence") or []
    lines = [
        "",
        "## Convergence record",
        "",
        "Per class, in the order first scored. Not a claim of convergence: first-score errors must "
        "fall and mechanisms per class must reach zero on classes from outside.",
        "",
        "| Class | First batch | n | Median first | Worst first | New mechanisms needed |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for c in rows:
        lines.append(
            f"| {c['class']} | {c['first_batch']} | {c['n']} | {100 * c['median_first']:.1f} % | "
            f"{_pct(c['worst_first'])} | {c['mechanisms']} |"
        )
    return lines


def _unit_lines(doc: dict[str, Any]) -> list[str]:
    u = doc.get("units") or {}
    if not u.get("units"):
        return []
    lines = [
        "",
        "## Per unit: model against counters",
        "",
        "Median of model busy cycles / counter cycles per block, on rows with the round 10 "
        "counters: "
        + ", ".join(f"{k} {v:.2f} (n {u['n'][k]})" for k, v in u["units"].items())
        + ".",
        "Stall families, mean counter share of stalled warp-cycles / model critical-path share: "
        + ", ".join(f"{k} {a:.2f} / {b:.2f}" for k, (a, b) in u["stalls"].items())
        + ".",
        "",
        "| Row | Unit | Model | Counter | Ratio |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for d in u["worst"]:
        lines.append(
            f"| {d['row']} | {d['unit']} | {d['model']:.0f} | {d['counter']:.0f} | {d['r']:.2f} |"
        )
    return lines


def _dynamic_lines(doc: dict[str, Any]) -> list[str]:
    ops = doc.get("dynamic_ops") or []
    if not ops:
        return []
    dflt = [e for e in ops if e["default"]]
    lines = [
        "",
        "## Dynamic opcode weight",
        "",
        f"{len(ops)} opcodes with modifiers on the `kernel` rows; default rules time "
        f"{len(dflt)} of them, {100 * sum(e['mean'] for e in dflt):.1f} % of the instructions on "
        "the mean row. Share of a row's instructions, mean over rows and the largest.",
        "",
        "| Opcode | Class | Default rule | Mean | Largest |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for e in [*ops[:15], *[e for e in dflt[:15] if e not in ops[:15]]]:
        lines.append(
            f"| `{e['op']}` | {e['class']} | {'yes' if e['default'] else ''} | "
            f"{100 * e['mean']:.2f} % | {100 * e['max']:.1f} % |"
        )
    return lines


def _whatif_lines(doc: dict[str, Any]) -> list[str]:
    if not doc.get("whatif"):
        return []
    lines = [
        "",
        "## What-if: predicted against measured change (28 §1)",
        "",
        f"One compiled change from a base in the same batch. Direction right: signs agree, or "
        f"measured within {100 * TIE:.0f} % of zero and predicted within {200 * TIE:.0f} %. "
        f"Magnitude: |predicted − measured| / |measured| on changes of at least "
        f"{100 * MAG_MIN:.0f} %; target 25 %. First: the prediction committed before the batch. "
        "Now: the current engine.",
        "",
        "| Axis | n | first: direction | first: median magnitude error | first: within 25 % | "
        "now: direction | now: median magnitude error | now: within 25 % |",
        "| --- " + "| ---: " * 7 + "|",
    ]
    for s in doc["whatif_summary"]:

        def med(key: str, s: dict[str, Any] = s) -> str:
            v = s[f"{key}_mag_median"]
            return "—" if v is None else f"{100 * v:.0f} %"

        lines.append(
            f"| {s['axis']} | {s['n']} | {s['first_ok']}/{s['n']} | {med('first')} | "
            f"{s['first_mag_within']}/{s['first_mag_n']} | {s['now_ok']}/{s['n']} | {med('now')} | "
            f"{s['now_mag_within']}/{s['now_mag_n']} |"
        )
    ps = doc["whatif"]
    short = [p for p in ps if p.get("graph")]
    if short:
        ok = sum(p["graph"]["ok"] for p in short) + sum(p["now_ok"] for p in ps if not p["graph"])
        lines += [
            "",
            f"Kernels under {SHORT_US:.0f} µs read from their graph rows (50 launches): "
            f"{sum(p['graph']['ok'] for p in short)}/{len(short)} of those changes right, "
            f"{ok}/{len(ps)} over all; the counts above stay the `kernel` rows.",
        ]
    lines += [
        "",
        "| Change | Base | Axis | measured Δ | first Δ | now Δ | first direction | now direction |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for p in sorted(doc["whatif"], key=lambda p: (p["now_ok"], p["first_ok"], p["id"])):
        lines.append(
            f"| {p['id']} | {p['base']} | {p['axis']} | {_pct(p['measured'])} | {_pct(p['first'])} "
            f"| {_pct(p['now'])} | {'right' if p['first_ok'] else 'WRONG'} | "
            f"{'right' if p['now_ok'] else 'WRONG'} |"
            + (
                f" graph: {_pct(p['graph']['measured'])}, {_pct(p['graph']['now'])}, "
                f"{'right' if p['graph']['ok'] else 'WRONG'} |"
                if p.get("graph")
                else ""
            )
        )
    if doc.get("bottlenecks"):
        lines += [
            "",
            "## Bottleneck: what the model names against what removal gains",
            "",
            "Critical-path share by kind in the base's replay (launch left out) and the change "
            "from removing that kind, replayed / measured.",
            "",
            "| Base | model's top | shares | gains, replay / measured | largest measured gain | "
            "agree |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for b in doc["bottlenecks"]:
            sh = ", ".join(f"{k} {100 * v:.0f} %" for k, v in b["share"].items())
            ga = ", ".join(f"{k} {_pct(v[0])} / {_pct(v[1])}" for k, v in b["gains"].items())
            lines.append(
                f"| {b['base']} | {b['model_top']} | {sh} | {ga} | {b['measured_top']} | "
                f"{'yes' if b['agree'] else 'NO'} |"
            )
    return lines


def _table(rows: list[dict[str, Any]]) -> list[str]:
    out = [
        "| Row | Level | Unit | Measured | Predicted | Error | Blind error | "
        "Attribution (critical path) | Defaults | Note |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for r in rows:
        att = "; ".join(
            f"{a['resource']}: {a['mechanism']} {100 * a['share']:.0f} %"
            for a in r.get("attribution", [])[:3]
        )
        defaults = ", ".join(f"{k} {v}" for k, v in sorted(r.get("defaults", {}).items()))
        note = (r["excluded"] and f"excluded: {r['excluded']}") or r["note"] or ""
        blind = _pct(r["blind"]) + f" ({r['blind_batch']})" if r.get("blind") is not None else ""
        out.append(
            f"| {r['id']} | {r['level']} | {r['unit']} | {_fmt(r['measured'])} | "
            f"{_fmt(r['predicted'])} | {_pct(r['error'])} | {blind} | {att} | {defaults} | {note} |"
        )
    return out
