"""Simulate a whole TileLang cp.async GEMM kernel on the GB10: occupancy from the driver's limits,
the busiest SM on
the SM simulator with block replacement, its share of the chip's L2 and store limits (blocks in
flight), and its DRAM misses from the launch-order L2 simulation (model.dram_split_kslice, checked
against ncu). Plus the launch. Timing condition: warm back-to-back runs (operands that fit the L2
stay there)."""

from __future__ import annotations

import itertools
import math

from .l2 import warm_miss as l2_warm_miss
from .machine import Machine, default_machine
from .model import bank_conflict_ways, dram_split_kslice
from .sass import main_loop
from .sm import SM, kernel_sass


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


def wave_slices(h: dict, KT: int, wave: dict, reads: float, cold: float, clk: float) -> tuple:
    """Round 9 (probe first_reader): what one wave's blocks meet, per tile (A, B): the fraction of
    the distinct slices that miss, the blocks that read each slice together; the DRAM clock, the
    bytes a step sends to DRAM. A warm launch scales the cold misses by `reads / cold`."""
    bm, bn, bk = h["block_M"], h["block_N"], h["block_K"]
    f = reads / cold if cold else 0.0
    return (
        min(1.0, f * wave["A"] / (wave["rows"] * KT * bm * bk * 2)),
        min(1.0, f * wave["B"] / (wave["cols"] * KT * bk * bn * 2)),
        wave["blocks"] / wave["rows"],
        wave["blocks"] / wave["cols"],
        clk,
        reads / KT,
    )


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


def simulate_kernel(
    sass_path,
    h: dict,
    shape: dict,
    regs: int,
    smem: int | None = None,
    warm: bool = True,
    kernel: str = "matmul_kernel",
    machine: Machine | None = None,
    sm_hook=None,
    trace: bool = False,
) -> dict:
    m = machine or default_machine()
    MC, SMS = m.C, m.sms
    F = MC["clock_under_load_GHz"] * 1e3  # cycles per us under load (clock probes 2.52-2.54 GHz)
    ins = kernel_sass(sass_path, kernel=kernel)
    M, N, K = shape["M"], shape["N"], shape["K"]
    KT = math.ceil(K / h["block_K"])
    smem = (
        smem
        if smem is not None
        else h["num_stages"] * (h["block_M"] + h["block_N"]) * h["block_K"] * 2
    )
    tiles = math.ceil(M / h["block_M"]) * math.ceil(N / h["block_N"])
    Rmax = occupancy(h, regs, smem, m)
    trips = sass_trips(ins, h, KT - (max(1, h["num_stages"]) - 1))
    sp = dram_split_kslice(h, shape, SMS * Rmax, MC["l2_effective_bytes"])
    # block replacement on the busiest SM: a new block starts when one exits
    rb = secs = 0.0
    per_wave = []
    for wv in sp["waves"]:
        r_, s_ = wave_dram(h, shape, wv, warm, m)
        rb += r_
        secs += s_
        per_wave.append((wv, r_, s_))
    copy_bytes = tiles * KT * (h["block_M"] + h["block_N"]) * h["block_K"] * 2
    miss = min(1.0, rb / copy_bytes) if copy_bytes else 0.0
    eq = min(tiles, SMS * Rmax) / min(Rmax, math.ceil(tiles / SMS))
    # round 9 (probe first_reader): the wave that first reads a slice takes its misses
    wave_miss = [
        wave_slices(h, KT, wv, r_, wv["A"] + wv["B"], dram_clk(r_, s_, eq, m))
        for wv, r_, s_ in per_wave
    ]
    sm = SM(
        ins,
        h["threads"],
        math.ceil(tiles / SMS),
        trips,
        active_sms=eq,
        copy_lines=copy_lines(h),
        miss_frac=miss,
        dram_clk_per_512B=dram_clk(rb, secs, eq, m),
        wave_miss=wave_miss if m.P["miss_by_wave"] else None,
        copies_a=max(1, h["block_M"] * h["block_K"] * 2 // 16 // h["threads"]),
        ctile_bytes=min(h["block_M"], M) * min(h["block_N"], N) * 2,
        ldsm_ways=ldsm_ways(h),
        resident=Rmax,
        grid_blocks=tiles,
        grid_dim=(math.ceil(N / h["block_N"]), math.ceil(M / h["block_M"]), 1),
        machine=m,
        trace=trace,
    )
    r = sm.run()
    if sm_hook is not None:  # KTA (PERFMODEL): the simulator for attribution
        sm_hook(sm)
    us = r["cycles"] / F
    return {
        "us": MC["launch_stream_us"] + us,
        "tiles": tiles,
        "blocks_per_sm": Rmax,
        "waves": math.ceil(tiles / (SMS * Rmax)),
        "dram_MB": rb / 1e6,
        "miss_frac": round(miss, 3),
        "dram_GBs": round(rb / secs / 1e9, 1) if secs else None,
        "busy": {k: round(v / max(1, r["cycles"]), 2) for k, v in r["busy"].items()},
    }
