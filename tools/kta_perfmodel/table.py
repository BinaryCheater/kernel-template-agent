"""The error table (docs/design/28 §4.7): every validation kernel and probe replayed on CPU,
measured against predicted, one row each.

A row is a spec (identity, measured value) plus a job (what to replay). Jobs run in a process
pool, and each result is cached under a key made of the engine's source and parameter files, the
upstream input commit and the job, so a rerun after a change only replays what the change can
affect (any engine file change replays everything)."""

from __future__ import annotations

import hashlib
import json
import re
import math
import os
import statistics
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from kta.models.gb10sim import attribution, probes, probeset, validate, ws_validate
from kta.models.gb10sim.machine import Machine, default_machine
from kta.models.gb10sim.model import C
from kta.models.regression import spearman

SETS = ("gemm", "tma", "probes", "batches")
BATCHES = Path(__file__).resolve().parents[2] / "perfmodel" / "batches"
# rows left out of the table: `tma 8192 full` (replay 150-300 s, far above the 28 §4 budget;
# deleted at the user's request, round 1)
DROPPED = {"tma 8192 full"}
SLOW: dict[str, str] = {}
ENGINE = Path(validate.__file__).parent
CACHE = Path(__file__).resolve().parents[2] / "data" / "perfmodel" / "cache"
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
    """The 22 cp.async GEMMs at 1024^3 (upstream validate.kernel_level), warm: the whole kernel,
    its loads-only variant at the grid, and its mma-only variant on one block (block_level)."""
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
    """The 7 warp-specialised TMA picks (upstream ws_validate): whole kernel at its grid (warm),
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
    for d in sorted(p for p in root.iterdir() if (p / "build").is_dir()) if root.exists() else []:
        b = probeset.load(d)
        meas = d / "measured.json"
        mdoc = json.loads(meas.read_text()) if meas.exists() else {}
        valid = mdoc.get("valid", False)
        counters = mdoc.get("counters", {})
        key = _batch_key(d)
        for case in probeset.all_cases(d):
            spec = _spec(
                case.id,
                case.probe if case.probe.startswith("testset") else f"probe {case.probe}",
                case.level,
                case.unit,
                case.measured if valid else float("nan"),
                ["batch", d.name, case.id, key],
                excluded=case.excluded if valid else ("batch not valid" if meas.exists() else None),
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
    cal: Path, job: list[Any], machine: Machine | None = None, program: dict | None = None
) -> dict[str, Any]:
    """Replay one job with tracing: `predicted`, the critical-path attribution and the ledger.
    `program` overrides the kernel's model inputs (`h.<key>` in its tile config, `regs`); the SASS
    stays the compiled one."""
    m = machine or default_machine()
    program = program or {}

    def pick(r: dict) -> dict:
        h = dict(r["pick"])
        h.update({k[2:]: v for k, v in program.items() if k.startswith("h.")})
        return h

    got: list[Any] = []
    kind = job[0]
    launch = m.param(m.desc["launch_us"])
    extra: dict[str, tuple[float, str | None]] = {}
    if kind in ("gemm", "gemm_block"):
        f = m.param(m.desc["clock"]["block_level_GHz"]) * 1e3
        r = json.loads((cal / "ablation_cpasync" / "mechanisms.json").read_text())[job[1]]
        if kind == "gemm":
            regs = int(program.get("regs", _regs(cal)[job[1]]))
            rmax = validate._rmax(pick(r), regs, m)
            predicted = validate.kernel_us(
                cal, job[1], job[2], pick(r), r["blocks"], rmax, m, got.append, True
            )
            extra["launch"] = (launch * f, m.desc["launch_us"])
        else:
            res = validate.simulate(
                cal, job[1], job[2], 1, 1, machine=m, sm_hook=got.append, trace=True
            )
            predicted = res["us"]
    elif kind in ("tma", "tma_block"):
        f = m.param(m.desc["clock"]["kernel_GHz"]) * 1e3
        r = json.loads((cal / "ablation" / "mechanisms.json").read_text())[job[1]]
        if kind == "tma":
            predicted = ws_validate.kernel_us(
                cal, job[1], pick(r), r["shape"], machine=m, sm_hook=got.append, trace=True
            )
            extra["launch"] = (launch * f, m.desc["launch_us"])
        else:
            res = ws_validate.ws_sim(
                cal,
                job[1],
                "noload",
                r["pick"],
                r["shape"],
                machine=m,
                sm_hook=got.append,
                trace=True,
            )
            predicted = res["us"]
    elif kind == "probe":
        if not _cases:
            _cases.update((c.id, c) for c in probes.all_cases(cal))
        predicted, sm, _ = probes.run_case(_cases[job[1]], machine=m, trace=True)
        got.append(sm)
    elif kind == "batch":
        if (job[1], job[2]) not in _batch_cases:
            d = BATCHES / job[1]
            _batch_cases.update(((job[1], c.id), c) for c in probeset.all_cases(d))
        predicted, sm, _ = probeset.run_case(_batch_cases[(job[1], job[2])], machine=m, trace=True)
        if job[2].endswith(" kernel"):
            ref = m.desc.get("launch_event_us", m.desc["launch_us"])
            extra["launch"] = (m.param(ref) * m.param(m.desc["clock"]["kernel_GHz"]) * 1e3, ref)
        got.append(sm)
    else:
        raise ValueError(f"unknown job {job}")
    sm = got[-1]
    a = attribution.analyse(sm, sm.cycles, extra)
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
        },
        "attribution": [
            {k: d[k] for k in ("resource", "mechanism", "share", "param")}
            for d in a["path"]
            if d["share"] >= 0.005
        ],
        "coverage": a["covered_cycles"] / a["cycles"] if a["cycles"] else 1.0,
        "defaults": attribution.default_counts(a["ledger"]),
        "ledger": a["ledger"],
    }


def engine_hash() -> str:
    """sha256 over the engine's code, description and parameter files and this table code."""
    h = hashlib.sha256()
    files = sorted(p for p in ENGINE.iterdir() if p.suffix in (".py", ".json", ".yaml"))
    for p in [*files, Path(__file__)]:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _timed(cal: Path, job: list[Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    t = time.perf_counter()
    out = run_job(cal, job, default_machine().with_overrides(overrides) if overrides else None)
    out["wall_s"] = round(time.perf_counter() - t, 2)
    return out


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
) -> Iterator[dict[str, Any]]:
    """Fill each spec with its result, cached ones first, the rest from a pool of `jobs`
    processes (longest previous runs first); yields rows as they complete."""
    eng = engine_hash()
    CACHE.mkdir(parents=True, exist_ok=True)

    def key(r: dict[str, Any]) -> Path:
        k = [eng, upstream_rev, r["job"]] + ([overrides] if overrides else [])
        k = hashlib.sha256(json.dumps(k).encode()).hexdigest()
        return CACHE / f"{k}.json"

    todo = []
    for r in rows:
        p = key(r)
        if r["id"] in SLOW and not run_slow and not (use_cache and p.exists()):
            r.update(predicted=None, excluded=SLOW[r["id"]], cached=False, wall_s=0.0)
            yield _finish(r)
            continue
        if use_cache and p.exists():
            r.update(json.loads(p.read_text()), cached=True)
            yield _finish(r)
        else:
            todo.append(r)
    if cached_only and todo:
        raise SystemExit(f"{len(todo)} rows not in the cache, e.g. {todo[0]['id']!r}")
    todo.sort(key=lambda r: -(hint or {}).get(r["id"], 0.0))
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        futs = {pool.submit(_timed, cal, r["job"], overrides): r for r in todo}
        for f in as_completed(futs):
            r = futs[f]
            res = f.result()
            key(r).write_text(json.dumps(res))
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


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
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
        if len(rs) >= 3 and (group.endswith("full") or group.startswith("testset")):
            item["spearman"] = spearman([r["predicted"] for r in rs], [r["measured"] for r in rs])
            item["inversions"] = inversions(rs)
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
    """The rows `batch` measured first, by set, at their first score (no engine had seen them)."""
    out = []
    picked = [r for r in rows if r.get("blind_batch") == batch and not r["excluded"]]
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
        if len(rs) >= 3 and group.startswith("testset"):
            first = [{**r, "predicted": r["blind_predicted"]} for r in rs]
            item["spearman"] = spearman([r["predicted"] for r in first], [r["measured"] for r in rs])
            item["inversions"] = inversions(first)
            item["now_spearman"] = spearman([r["predicted"] for r in rs], [r["measured"] for r in rs])
            item["now_inversions"] = inversions(rs)
        out.append(item)
    return out


def summary_line(s: dict[str, Any]) -> str:
    sp = f"{s['spearman']:.2f}" if "spearman" in s else ""
    inv = str(s["inversions"]) if "inversions" in s else ""
    r1 = f"{s['r1_pass']}/{s['r1_n']}" if "r1_n" in s else ""
    bn = f"{s['bottleneck_agree']}/{s['r1_n']}" if "r1_n" in s else ""
    mm = f"{100 * s['median_abs_mech']:.1f} %" if "median_abs_mech" in s else ""
    mw = _pct(s["worst_mech"]) if "worst_mech" in s else ""
    ms = f"{s['spearman_mech']:.2f} / {s['inversions_mech']}" if "spearman_mech" in s else ""
    bl = ""
    if "blind_median" in s:
        bl = f"{100 * s['blind_median']:.1f} % / {_pct(s['blind_worst'])}"
    return (
        f"| {s['set']} | {s['n']} | {s['excluded']} | {100 * s['median_abs']:.1f} % | "
        f"{_pct(s['worst'])} | {mm} | {mw} | {bl} | {s['within_10']} | {sp} | {inv} | {ms} | {r1} "
        f"| {bn} |"
    )


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
        "worst, mechanisms only | blind median / worst | within 10 % | Spearman | inversions | "
        "Spearman / inversions, mechanisms only | counts check (R1) | bottleneck agrees |",
        "| --- " + "| ---: " * 13 + "|",
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
            "| Set | n | first: median | first: worst | first: Spearman / inversions | "
            "now: median | now: worst | now: Spearman / inversions |",
            "| --- " + "| ---: " * 7 + "|",
        ]
        for s in doc["blind"]:
            f = f"{s['spearman']:.2f} / {s['inversions']}" if "spearman" in s else ""
            n = f"{s['now_spearman']:.2f} / {s['now_inversions']}" if "now_spearman" in s else ""
            lines.append(
                f"| {s['set']} | {s['n']} | {100 * s['median_abs']:.1f} % | {_pct(s['worst'])} | "
                f"{f} | {100 * s['now_median']:.1f} % | {_pct(s['now_worst'])} | {n} |"
            )
    ranked = sorted((r for r in rows if not r["excluded"]), key=lambda r: -abs(r["error"]))
    lines += ["", f"## Top {top} rows by absolute error", ""]
    lines += _table(ranked[:top])
    lines += ["", "## All rows", ""]
    lines += _table(rows)
    return "\n".join(lines) + "\n"


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
