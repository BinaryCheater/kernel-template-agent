"""The one whole-kernel front end (round 10): any kernel from its SASS, its launch and its register
count. The busiest SM of the grid on the SM simulator with block replacement, the chip's shares by
blocks in flight, what each wave of blocks misses from the L2 replay of the kernel's own addresses
(l2.py). The launch-order helpers `wave_dram`, `dram_split_kslice` stay for the warp-specialised TMA
kernels (ws.py), the only rows that do not come here."""

from __future__ import annotations

import itertools
import json
import math
import re
from typing import Any

from .l2 import replay_walk, walk
from .l2 import rotating_miss as l2_rotating_miss
from .l2 import warm_miss as l2_warm_miss
from .machine import Machine, default_machine
from .model import bank_conflict_ways
from .sass import Ins, main_loop
from .sm import SM


def copy_lines(h: dict) -> list:
    """Lines each copy (32 lanes x 16 B) touches, per warp, in the order TileLang issues them: the A
    tile's copies (rows of block_K*2 bytes) then the B tile's (rows of block_N*2 bytes). A 512-byte
    copy spans 512/row rows; each row covers max(1, row/128) lines."""

    def lines(row):
        rows = max(1, 512 // row)
        return rows * max(1, min(row, 512) // 128)

    nA = h["block_M"] * h["block_K"] * 2 // 16 // h["threads"]
    nB = h["block_K"] * h["block_N"] * 2 // 16 // h["threads"]
    return [lines(h["block_K"] * 2)] * max(1, nA) + [lines(h["block_N"] * 2)] * max(1, nB)


def sass_trips(ins, h: dict, src_trips: int) -> int:
    """The machine loop may be unrolled: its trip count is the source count over the k-tiles one
    iteration covers, found from the mma (or copies) one iteration holds against what one k-tile
    needs per warp."""
    lp = main_loop(ins)
    if not lp:
        return 1
    body = ins[lp[0] : lp[1] + 1]
    warps = h["threads"] // 32
    n_mma = sum(x.base in ("HMMA", "IMMA", "QMMA", "OMMA") for x in body)
    n_cp = sum(x.base == "LDGSTS" for x in body)
    if n_mma:
        u = n_mma / (h["block_M"] * h["block_N"] * h["block_K"] / (16 * 8 * 16) / warps)
    elif n_cp:
        u = n_cp / ((h["block_M"] + h["block_N"]) * h["block_K"] * 2 / 16 / h["threads"])
    else:
        u = 1
    return max(1, round(src_trips / max(1, round(u))))


def occupancy(h: dict, regs: int, smem: int, machine: Machine) -> int:
    MC = machine.C
    warps = h["threads"] // 32
    unit = MC["reg_alloc_unit_per_warp"]
    rw = math.ceil(regs * 32 / unit) * unit
    return max(
        1,
        min(
            MC["regs_per_sm"] // (warps * rw),
            MC["smem_per_sm_bytes"] // (smem + MC["smem_reserved_per_block_bytes"]),
            MC["max_threads_per_sm"] // h["threads"],
            MC["max_blocks_per_sm"],
        ),
    )


def dram_peak(run_bytes: float, machine: Machine) -> float:
    """DRAM read rate for misses that arrive as contiguous runs of run_bytes (probe tileload.cu,
    DRAM source, 48-96
    blocks: 64-byte runs 201-204 GB/s, 256-byte 230-234; mlp_stream contiguous 237)."""
    MC = machine.C
    pts = MC["dram_peak_by_segment_GBs"]
    peak = pts[-1][1]
    if run_bytes <= pts[0][0]:
        peak = pts[0][1]
    for (x0, y0), (x1, y1) in itertools.pairwise(pts):
        if x0 < run_bytes <= x1:
            peak = y0 + (y1 - y0) * (math.log(run_bytes) - math.log(x0)) / (
                math.log(x1) - math.log(x0)
            )
            break
    return max(MC["dram_rate_floor_GBs"], peak)


def wave_dram(
    h: dict, shape: dict, wave: dict, warm: bool, machine: Machine
) -> tuple[float, float]:
    """One wave's DRAM work: (read-miss bytes, seconds of DRAM time for the whole chip).
    A misses come as block_K*2-byte pieces of rows far apart; B misses come as pieces of the same B
    rows that the
    neighbouring column tiles in flight read together, one run of (columns in flight) x block_N*2
    bytes. C tiles go
    back to DRAM when operands and output do not stay in the L2 together (a write-back cache: a line
    written in one run
    is evicted before the next run writes it), at the write rate; the controller serves the mix as
    each part at its own
    rate (probe dram.cu: 1 / (f_read/242 + f_write/197) within 4 %)."""
    MC, P = machine.C, machine.P
    M, N, K = shape["M"], shape["N"], shape["K"]
    bm, bn, bk = h["block_M"], h["block_N"], h["block_K"]
    if warm and not P["l2_replay"] and (M * K + K * N + M * N) * 2 <= MC["l2_effective_bytes"]:
        return 0.0, 0.0
    # round 4 (batch r4, HGEMM warm vs rotating): a warm launch misses by the measured warm-miss
    # curve on the operands' read footprint; the cold wave's traffic scaled by that fraction.
    # Round 5: the one rule of l2.py for every warm launch, small footprints included
    if warm and (P["hgemm_warm_reads_by_curve"] or P["l2_replay"]):
        f = l2_warm_miss((M * K + K * N) * 2, "ca", machine)
        r_, s_ = wave_dram(h, shape, wave, False, machine)
        return r_ * f, s_ * f
    run_a = min(bk, K) * 2 if K > bk else min(bm, M) * K * 2
    run_b = min(N, wave["cols"] * bn) * 2
    c_w = wave["blocks"] * min(bm, M) * min(bn, N) * 2
    if not warm:
        # timed after a flush: C ends the kernel dirty in the L2 and goes back to DRAM later (in the
        # next flush), except what the kernel itself pushes out because C does not fit beside the
        # stream
        cap = MC["l2_effective_bytes"]
        c_all = M * N * 2
        c_w *= max(0.0, c_all - cap) / c_all if c_all else 0.0
    wpeak = max(MC["dram_rate_floor_GBs"], MC["dram_write_GBs"])
    secs = (
        wave["A"] / dram_peak(run_a, machine) + wave["B"] / dram_peak(run_b, machine) + c_w / wpeak
    ) / 1e9
    return wave["A"] + wave["B"], secs


def replay_waves(
    r: Any, scale: dict, write_bytes: float, trips: int, eq: float, m: Machine
) -> list:
    """Round 10: what one wave's blocks meet, per base, from the L2 replay `r`: the fraction of the
    distinct bytes that miss and the blocks that read each byte; the DRAM clock over the wave's
    misses (each contiguous run at its rate, the launch's writes in proportion); the bytes a step
    sends to DRAM. `scale`: per base the part of the cold misses this launch's cache state keeps."""
    total = sum(p[0] * scale.get(b, 1.0) for w in r.parts for b, p in w.items())
    out = []
    for parts, runs in zip(r.parts, r.runs, strict=True):
        got = sum(p[0] * scale.get(b, 1.0) for b, p in parts.items())
        cold = sum(p[0] for p in parts.values())
        keep = got / cold if cold else 0.0
        secs = sum(v * keep / dram_peak(run, m) for run, v in runs.items()) / 1e9
        if total:
            wr = write_bytes * got / total if m.P["dram_writes_paced"] else 0.0
            secs += wr / (m.C["dram_write_GBs"] * 1e9)
        slices = {
            b: (p[0] * scale.get(b, 1.0) / p[1], p[2] / p[1]) for b, p in parts.items() if p[1]
        }
        out.append((slices, dram_clk(got, secs, eq, m), got / max(1, trips)))
    return out


def dram_mix_secs(read_bytes: float, write_bytes: float, m: Machine) -> float:
    """Seconds of DRAM time for a rotating stream of reads and writes (round 4, probe dram_rw):
    the reads at dram_read_GBs, the writes at dram_write_GBs when `sim.dram_writes_paced`, else
    not DRAM traffic (the rule before round 4)."""
    secs = read_bytes / (m.C["dram_read_GBs"] * 1e9)
    if m.P["dram_writes_paced"]:
        secs += write_bytes / (m.C["dram_write_GBs"] * 1e9)
    return secs


def dram_clk(read_bytes: float, secs: float, blocks_eq: float, machine: Machine) -> float:
    """Cycles per 512 B of read misses for one SM: the wave's DRAM time spread over its read misses
    (writes folded in),
    shared by the blocks in flight."""
    MC = machine.C
    if read_bytes <= 0:
        return 512 / (MC["dram_read_GBs"] * 1e9 / (MC["clock_under_load_GHz"] * 1e9) / blocks_eq)
    return 512 * (secs / read_bytes) * (MC["clock_under_load_GHz"] * 1e9) * blocks_eq


def ldsm_ways(h: dict) -> tuple:
    """Bank-conflict ways of ldmatrix on A (rows block_K*2 bytes apart) and B (read transposed, rows
    block_N*2 apart):
    TileLang's inferred and swizzled layouts are conflict-free; a linear layout keeps the row
    stride."""
    if h.get("smem_layout") != "linear":
        return (1, 1)
    return (bank_conflict_ways(h["block_K"] * 2), bank_conflict_ways(h["block_N"] * 2))


P0_SLOT = 0x380  # the kernel's first parameter in constant bank 0


def machine_trips(ins: list[Ins]) -> int | None:
    """The main loop's passes from its counter when the compiler left one (`ISETP R, limit` with
    `IADD R, R, step`, round 7: a partly unrolled source loop), else None."""
    lp = main_loop(ins)
    if not lp:
        return None
    body = ins[lp[0] : lp[1] + 1]
    for x in reversed(body):
        m = re.search(r"(R\d+), (0x[0-9a-f]+), PT", x.args) if x.base == "ISETP" else None
        if m:
            for y in body:
                st = re.fullmatch(rf"{m[1]}, {m[1]}, (0x[0-9a-f]+)", y.args)
                if y.base == "IADD" and st:
                    return math.ceil(int(m[2], 16) / int(st[1], 16))
    return None


def grid_bank(L: dict[str, Any]) -> dict[tuple[int, int], int]:
    """gridDim as the kernel reads it (constant bank 0, TileLang's raster panels divide by it),
    and the launch's arguments where they were captured (`const_bank`: offset -> word)."""
    bank = {(0, int(k)): int(v) for k, v in L.get("const_bank", {}).items()}
    return {**bank, **{(0, 0x370 + 4 * i): int(v) for i, v in enumerate((*L["grid"], 1, 1)[:3])}}


_l2_cache: dict[tuple, Any] = {}


def _l2_replay(ins: list[Ins], L: dict[str, Any], trips: int, in_flight: int, m: Machine) -> Any:
    cap = m.C["l2_effective_bytes"]
    key = (L["entry"], json.dumps(L, sort_keys=True), len(ins), trips, in_flight, cap)
    if key not in _l2_cache:
        block = tuple(L.get("block", [L["threads"]]))
        acc = walk(ins, L["threads"], trips, tuple(L["grid"]), block, m, grid_bank(L))
        _l2_cache[key] = replay_walk(acc, in_flight, cap)
    return _l2_cache[key]


def tile_launch(h: dict, shape: dict, regs: int, smem: int | None, warm: bool, entry: str) -> dict:
    """R5: the launch of a TileLang GEMM from its template parameters."""
    M, N, K = shape["M"], shape["N"], shape["K"]
    bm, bn, bk = h["block_M"], h["block_N"], h["block_K"]
    return {
        "entry": entry,
        "grid": [math.ceil(N / bn), math.ceil(M / bm), 1],
        "threads": h["threads"],
        "block": [h["threads"]],
        "smem": smem if smem is not None else h["num_stages"] * (bm + bn) * bk * 2,
        "footprint_bytes": (M * K + K * N) * 2,
        "write_bytes": M * N * 2,
        "pick": dict(h),
        "cache": "warm" if warm else "cold",
    }


def generic_kernel(
    ins: list[Ins],
    L: dict[str, Any],
    regs: int,
    trips: int,
    carried: tuple[str, ...] = (),
    **kw: Any,
) -> SM:
    """The busiest SM of the grid with block replacement, the chip's shares by blocks in flight.
    `trips`: the main loop's passes when its counter is not a value the replay knows. The L2 (round
    5, l2.py): the loads' own addresses over the grid through one cache in launch order; rotating
    launches read that traffic at the DRAM rate shared by the blocks in flight (as the dram_rotate
    probe), warm launches the part the measured curve of the read footprint loses (plain loads;
    evict-first loads their own). Copies meet that per wave and base (`replay_waves`, round 10).
    LDG lines, sectors and L1 hits from the lanes' addresses (sm.SM lane tracking, round 3).
    `carried`: buffers an earlier kernel of a sequence left in the L2 (round 6): loads from them
    hit."""
    tiles = math.prod(L["grid"])
    m = kw.get("machine") or default_machine()
    h = {"threads": L["threads"]}
    rmax = occupancy(h, regs, L["smem"], m)
    per_sm = math.ceil(tiles / m.sms)
    eq = min(tiles, m.sms * rmax) / min(rmax, per_sm)
    fb = L["footprint_bytes"]
    cache = L.get("cache")
    rotating = cache in ("rotating", "cold")
    once = False
    ldg = [x for x in ins if x.base == "LDG"]
    kind = "cs" if ldg and all(".EF" in x.op for x in ldg) else "ca"
    # round 4: the written buffers (gen.py `write_bytes`) are DRAM traffic beside the reads
    wb = L.get("write_bytes", 0) if cache == "rotating" else 0
    if m.P["l2_replay"]:
        r = _l2_replay(ins, L, trips, m.sms * rmax, m)
        # round 7 (rot_depth): a rotation of `rot` sets is a re-read of everything it touches
        # round 12: unknown addresses read what the known ones leave of the buffers
        rest = max(0, fb - r.footprint - L.get("write_bytes", 0))
        seen = r.footprint + min(r.loose, rest)
        # round 12 (rmw_*): lines stores rewrite after reading them count once
        written = 0 if r.rewrites else L.get("write_bytes", 0)
        window = L.get("rot", 1) * (seen + written)
        if cache == "cold":
            scale = 1.0
        elif rotating:
            scale = l2_rotating_miss(window, kind, m)
        else:
            # round 11 (wr_*): reads a launch uses once stay on the L2's probation list beside the
            # stores; re-read lines are protected (HGEMM)
            once = m.P["warm_window_writes"] and r.loaded < 1.1 * seen and not r.rewrites
            scale = l2_warm_miss(seen + (written if once else 0), kind, m)
        # the kernel's arguments, 8 bytes each from the first parameter slot, by name
        slots = {P0_SLOT + 8 * i: n for i, n in enumerate(L.get("params", ()))}
        byname = {
            pc: slots.get(sym[2])
            for pc, base in r.base_by_pc.items()
            for sym, _ in base or ()
            if sym[0] == "c"
        }
        miss_pc = {
            pc: 0.0 if byname.get(pc) in carried else v * scale for pc, v in r.miss_by_pc.items()
        }
        kept = sum(miss_pc.values()) / max(1e-9, sum(r.miss_by_pc.values()))
        rb, miss = r.reads * kept, 1.0 if kept > 0 else 0.0
        kw["miss_by_pc"] = miss_pc
        kw["burst_bytes"] = rb / max(1, len(r.waves)) / max(1, trips)
        if m.P["miss_by_wave"] and any(x.base == "LDGSTS" for x in ins):
            bases = {b for w in r.parts for b in w}
            names = {
                b: next((slots.get(s[2]) for s, _ in b or () if s[0] == "c"), None) for b in bases
            }
            sc = {b: 0.0 if names[b] in carried else scale for b in bases}
            kw["wave_miss"] = replay_waves(r, sc, wb, trips, eq, m)
    elif rotating:
        rb, miss = fb - wb, 1.0
    else:  # warm (round 3, probe l2reuse): the re-read misses by the footprint
        rb, miss = fb, l2_warm_miss(fb, kind, m)
    kw["dram_writes_counted"] = wb > 0
    # round 11 (wr_*, r11b): repeated launches writing past the limit find it full from the start:
    # W 6 MiB pays nothing, 7 most of it, beside single-use reads; beside re-read lines (HGEMM's
    # 9.4 MB of rewritten C) only past 1.5 times the limit, up to 16 MiB unmeasured
    w_limit = m.P["stg_dirty_once_bytes"] if once else 1.5 * m.P["stg_dirty_bytes"]
    kw["dirty_full"] = not rotating and L.get("write_bytes", 0) > w_limit
    if any(x.base in ("RED", "REDG", "ATOMG") for x in ins):
        kw["atomic_bytes"] = L.get("write_bytes", 0)
    elif "write_bytes" in L:
        kw["write_footprint"] = L["write_bytes"]
    if miss > 0:
        kw = {
            **kw,
            "miss_frac": miss,
            "dram_clk_per_512B": dram_clk(rb, dram_mix_secs(rb, wb, m), eq, m),
        }
    return SM(
        ins,
        L["threads"],
        per_sm,
        trips if main_loop(ins) else 1,
        active_sms=eq,
        resident=rmax,
        grid_blocks=tiles,
        grid_dim=tuple(L["grid"]),
        block_dim=tuple(L.get("block", [L["threads"]])),
        l1_bytes=m.C["l1_unified_bytes"] - rmax * L["smem"],
        const_bank=grid_bank(L),
        ldsm_ways=ldsm_ways(L["pick"]) if "smem_layout" in L.get("pick", {}) else (1, 1),
        **kw,
    )
