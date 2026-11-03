"""Replays of KTA's own measurement batches (docs/design/perfmodel/LOOP.md products 5 and 6): the
probes under `perfmodel/probes/` and the test-set kernels under `perfmodel/testset/`, each from the
SASS the batch built (`perfmodel/batches/<batch>/build/<unit>/k.sass`) with the launch the batch
ran, against the medians the batch measured (`measured.json`, empty before the run: the replay
is then the prediction written into the batch's entry blocks).

Cycle-valued probes compare cycles directly. Microsecond-valued rows convert with the clock the
batch measured (`clock.chain_GHz`) when there is one, else with the description's kernel clock.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .kernel import (
    copy_lines,
    dram_clk,
    dram_mix_secs,
    launch_us,
    occupancy,
    sass_trips,
    simulate_kernel,
)
from .l2 import replay_walk, walk
from .l2 import warm_miss as l2_warm_miss
from .machine import Machine, default_machine
from .probes import Case, ClockMarked, marked_cycles
from .sass import Ins, main_loop, parse
from .sm import SM

P0_SLOT = 0x380  # constant-bank offset of a kernel's first argument (sm_121, seen in every SASS)
ITERS = 256  # smem_ldsts, shfl, ex2_ffma, l1hit tput
BAR_ITERS = 128  # namedbar
HOPS = 2048  # l1hit chase
DRAM_BYTES = 48 * 1024 * 16 * 85  # dram_rotate: bytes per launch
SMEM_WARPS = (1, 2, 3, 4, 5, 6, 8, 16, 32)  # smem_ldsts (2, 3, 5, 6 from round 2)
MMA_SKEWS = (0, 4096, 16384)  # mma_pair: FFMA iterations before the loop of blocks 48..95
L2SHARE_PICK = {
    "block_M": 128,
    "block_N": 256,
    "block_K": 64,
    "num_stages": 2,
    "threads": 256,
}


@dataclass
class Batch:
    name: str
    dir: Path
    measured: dict[str, float]  # "<probe> <point>" -> median
    clock_GHz: float | None

    @property
    def build(self) -> Path:
        return self.dir / "build"


def load(batch_dir: Path) -> Batch:
    mp = batch_dir / "measured.json"
    measured: dict[str, float] = {}
    clock = None
    if mp.exists():
        doc = json.loads(mp.read_text())
        measured = {k: v["median"] for k, v in doc["results"].items()}
        clock = measured.get("clock chain_GHz")
    return Batch(batch_dir.name, batch_dir, measured, clock)


def _sass(b: Batch, unit: str) -> str:
    return (b.build / unit / "k.sass").read_text()


def regs_of(b: Batch, unit: str, kernel: str) -> int:
    text = (b.build / unit / "res-usage.txt").read_text()
    m = re.search(rf"Function {re.escape(kernel)}:\s*REG:(\d+)", text)
    if not m:
        raise ValueError(f"{unit}: no register count for {kernel}")
    return int(m.group(1))


def _ghz(b: Batch, m: Machine) -> float:
    return b.clock_GHz or m.param(m.desc["clock"]["kernel_GHz"])


def _k(ins_text: str, kernel: str) -> list[Ins]:
    ins = parse(ins_text, kernel=kernel)
    if not ins:
        raise ValueError(f"no SASS for {kernel}")
    return ins


def _case(b: Batch, cid: str, probe: str, level: int, unit: str, build, read, **kw) -> Case:
    return Case(
        id=cid,
        probe=probe,
        level=level,
        unit=unit,
        measured=b.measured.get(cid, float("nan")),
        build=build,
        read=read,
        **kw,
    )


def per(div: float):
    return lambda sm, r: marked_cycles(sm, r) / div


def smem_cases(b: Batch) -> list[Case]:
    txt = _sass(b, "probe-smem_ldsts")
    out = []
    for kind, name in enumerate(["LDS.32", "LDS.64", "LDS.128", "STS.32", "STS.128"]):
        ins = _k(txt, f"_Z1kILi{kind}EEvjPyPj")
        for w in SMEM_WARPS:
            out.append(
                _case(
                    b,
                    f"smem_ldsts {name}_w{w}",
                    "smem_ldsts",
                    1 if w == 1 else 2,
                    "cycles/instr/SM",
                    lambda ins=ins, w=w, **kw: ClockMarked(ins, 32 * w, 1, 1, **kw),
                    per(ITERS * 8 * w),
                )
            )
    return out


def shfl_cases(b: Batch) -> list[Case]:
    txt = _sass(b, "probe-shfl")
    i1, i8 = _k(txt, "_Z1kILi1EEvPyPj"), _k(txt, "_Z1kILi8EEvPyPj")
    out = [
        _case(
            b,
            "shfl latency_w1",
            "shfl",
            1,
            "cycles/hop",
            lambda **kw: ClockMarked(i1, 32, 1, 1, **kw),
            per(ITERS * 5),
        )
    ]
    for w in (1, 4, 8, 16, 32):
        out.append(
            _case(
                b,
                f"shfl tput_w{w}",
                "shfl",
                2,
                "cycles/instr/SM",
                lambda w=w, **kw: ClockMarked(i8, 32 * w, 1, 1, **kw),
                per(ITERS * 5 * 8 * w),
            )
        )
    return out


def ex2_cases(b: Batch) -> list[Case]:
    txt = _sass(b, "probe-ex2_ffma")
    out = []
    for e, f in ((8, 0), (8, 8), (8, 16), (8, 32), (0, 32), (4, 32)):
        ins = _k(txt, f"_Z1kILi{e}ELi{f}EEvPyPf")
        cid = f"ex2_ffma E{e}_F{f}"
        # 8 warps per sub-partition issue at least E + F + 3 (loop) instructions per pass each
        floor = 8 * (e + f + 3)
        meas = b.measured.get(cid, float("nan"))
        out.append(
            _case(
                b,
                cid,
                "ex2_ffma",
                3,
                "cycles/pass/smsp",
                lambda ins=ins, **kw: ClockMarked(ins, 1024, 1, 1, **kw),
                per(ITERS),
                excluded=(
                    f"measured {meas:.0f} cycles is below the issue limit {floor} (one warp "
                    "instruction per cycle per sub-partition): open question"
                    if meas < floor
                    else None
                ),
            )
        )
    return out


def _pair_sum(sm: SM, r: dict) -> float:
    """namedbar arrive: warp 0 reads the clock before and after each bar.arrive."""
    marks = getattr(sm, "marks", [])
    pairs = list(zip(marks[0::2], marks[1::2], strict=False))
    return sum(t1 - t0 for t0, t1 in pairs) / BAR_ITERS


def bar_cases(b: Batch) -> list[Case]:
    txt = _sass(b, "probe-namedbar")
    sync = _k(txt, "_Z6sync_kPy")
    out = [
        _case(
            b,
            f"namedbar sync_w{w}",
            "namedbar",
            1,
            "cycles/barrier",
            lambda w=w, **kw: ClockMarked(sync, 32 * w, 1, 1, **kw),
            per(BAR_ITERS),
        )
        for w in (2, 4, 8, 16, 32)
    ]
    for L in (0, 64, 256):
        ins = _k(txt, f"_Z8arrive_kILi{L}EEvPyPf")
        out.append(
            _case(
                b,
                f"namedbar arrive_L{L}",
                "namedbar",
                3,
                "cycles/arrive",
                # the loop counter starts from an HFMA2 used as a move, which the value tracking
                # does not evaluate: the loop's trip count is given
                lambda ins=ins, **kw: ClockMarked(ins, 256, 1, BAR_ITERS, **kw),
                _pair_sum,
            )
        )
    return out


def l1_cases(b: Batch) -> list[Case]:
    txt = _sass(b, "probe-l1hit")
    chase, tput = _k(txt, "_Z5chasePKjPy"), _k(txt, "_Z4tputPK4int4PyPi")
    out = [
        _case(
            b,
            "l1hit chase",
            "l1hit",
            1,
            "cycles/hop",
            lambda **kw: ClockMarked(chase, 32, 1, 1, ldg_lines=[(1, 4, True)], **kw),
            per(HOPS),
        )
    ]
    for w in (4, 8, 16, 32):
        out.append(
            _case(
                b,
                f"l1hit tput_w{w}",
                "l1hit",
                2,
                "cycles/instr/SM",
                lambda w=w, **kw: ClockMarked(tput, 32 * w, 1, 1, ldg_lines=[(4, 512, True)], **kw),
                per(ITERS * 8 * w),
            )
        )
    return out


def store_cases(b: Batch) -> list[Case]:
    txt = _sass(b, "probe-store_tail")
    out = []
    for mode, name in ((0, "tl"), (1, "v16")):
        ins = _k(txt, f"_Z2stILi{mode}EEvPji")
        for blocks in (1, 2, 8, 16, 32, 48):
            out.append(
                _case(
                    b,
                    f"store_tail {name}_b{blocks}",
                    "store_tail",
                    4,
                    "us",
                    lambda ins=ins, blocks=blocks, **kw: SM(
                        ins, 256, 1, 1, active_sms=blocks, const_bank={(0, 0x388): 8}, **kw
                    ),
                    lambda sm, r: r["cycles"] / (_ghz(b, sm.m) * 1e3),
                )
            )
    return out


def l2share_cases(b: Batch) -> list[Case]:
    ins = _k(_sass(b, "probe-l2share_copy"), "matmul_kernel")
    h = L2SHARE_PICK
    trips = sass_trips(ins, h, 15)
    return [
        _case(
            b,
            f"l2share_copy b{blocks}",
            "l2share_copy",
            4,
            "us",
            lambda blocks=blocks, **kw: SM(
                ins, 256, 1, trips, active_sms=blocks, copy_lines=copy_lines(h), **kw
            ),
            lambda sm, r: r["cycles"] / (_ghz(b, sm.m) * 1e3),
        )
        for blocks in (1, 16, 32, 48)
    ]


def dram_cases(b: Batch) -> list[Case]:
    txt = _sass(b, "probe-dram_rotate")
    out = []
    for t in (32, 64, 128, 256, 512, 1024):
        ins = _k(txt, f"_Z6streamILi{t}EEvPK4int4Pi")

        def build(ins=ins, t=t, **kw):
            mm = kw.get("machine") or default_machine()
            secs = DRAM_BYTES / (mm.C["dram_read_GBs"] * 1e9)
            return SM(
                ins,
                t,
                1,
                1,
                active_sms=48,
                ldg_lines=[(4, 512, False)],
                miss_frac=1.0,
                dram_clk_per_512B=dram_clk(DRAM_BYTES, secs, 48, mm),
                **kw,
            )

        out.append(
            _case(
                b,
                f"dram_rotate t{t}",
                "dram_rotate",
                5,
                "us",
                build,
                lambda sm, r: r["cycles"] / (_ghz(b, sm.m) * 1e3) + _launch_event(sm.m),
                limit=200_000_000,
            )
        )
    return out


DRAM_RW_UN = 96 * 512 * 40  # dram_rw: int4 per plane
DRAM_RW_MIX = ((2, 0), (2, 1), (1, 1), (1, 2), (0, 2))


def dram_rw_cases(b: Batch) -> list[Case]:
    """Round 4 (probe dram_rw): NR read and NW written planes of 30 MiB, rotating (every read a
    miss, every written line back to DRAM), 96 blocks x 512 threads."""
    txt = _sass(b, "probe-dram_rw")
    out = []
    for nr, nw in DRAM_RW_MIX:
        ins = _k(txt, f"_Z2rwILi{nr}ELi{nw}EEvPK4int4PS0_Pi")

        def build(ins=ins, nr=nr, nw=nw, **kw):
            mm = kw.get("machine") or default_machine()
            rb, wb = nr * DRAM_RW_UN * 16, nw * DRAM_RW_UN * 16
            return SM(
                ins,
                512,
                2,
                1,
                active_sms=48,
                ldg_lines=[(4, 512, False)],
                miss_frac=1.0,
                dram_clk_per_512B=dram_clk(rb, dram_mix_secs(rb, wb, mm), 48, mm),
                **kw,
            )

        out.append(
            _case(
                b,
                f"dram_rw r{nr}_w{nw}",
                "dram_rw",
                5,
                "us",
                build,
                lambda sm, r: r["cycles"] / (_ghz(b, sm.m) * 1e3) + _launch_event(sm.m),
                limit=400_000_000,
            )
        )
    return out


class BlockMarked(SM):
    """The SM simulator that records, per block, when the block's first warp reads the clock
    (mma_pair: entry, after the skew, exit)."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.bmarks: dict[int, list[float]] = {}

    def issue(self, w, now):
        if w.wid % self.wpb == 0 and "SR_CLOCKLO" in self.ins[w.pc].args:
            self.bmarks.setdefault(w.block, []).append(now)
        super().issue(w, now)


def _span(sm: BlockMarked, r: dict) -> float:
    m = [v for v in sm.bmarks.values() if v]
    return max(v[-1] for v in m) - min(v[0] for v in m) if m else float(r["cycles"])


def _first(sm: BlockMarked, r: dict) -> float:
    m = [v for v in sm.bmarks.values() if v]
    return min(v[-1] for v in m) - min(v[0] for v in m) if m else float(r["cycles"])


def mma_pair_cases(b: Batch) -> list[Case]:
    """Round 2: the HGEMM warp tile's mma + ldmatrix loop from shared memory, one block, one per
    SM, two per SM with a start skew (probe mma_pair). The replay is one SM; the grid gives the
    block indices (blocks 48..95 run the skew) and the chip share."""
    txt = _sass(b, "probe-mma_pair")
    out = []
    for skew in MMA_SKEWS:
        ins = _k(txt, f"_Z1kILi{skew}EEvjPyPf")
        pts = [("b1", 1, 1), ("b48", 48, 1)] if skew == 0 else []
        pts.append((f"b96_s{skew}", 96, 2))
        for name, grid, per_sm in pts:

            def build(ins=ins, grid=grid, per_sm=per_sm, **kw):
                return BlockMarked(
                    ins,
                    128,
                    per_sm,
                    1,
                    active_sms=min(grid, 48),
                    resident=per_sm,
                    grid_blocks=grid,
                    grid_dim=(grid, 1, 1),
                    **kw,
                )

            reads = [("span", _span)] + ([("first", _first)] if per_sm == 2 else [])
            for what, read in reads:
                out.append(
                    _case(b, f"mma_pair {name}_{what}", "mma_pair", 3, "cycles", build, read)
                )
    return out


MMA_TYPES = (
    "f32_f16",
    "f16_f16",
    "f32_bf16",
    "f32_e4m3",
    "f16_e4m3",
    "f32_e5m2",
    "s32_s8",
    "s32_s4",
)
# the kernel template's type index per point (the probe's POINT list)
MMA_TYPE_INDEX = dict(zip(MMA_TYPES, (0, 1, 2, 4, 5, 6, 7, 8), strict=True))
ROW_POINTS = {
    "empty": (0, 0, 0),
    "ld": (1, 0, 0),
    "red": (0, 1, 0),
    "st": (0, 0, 1),
    "ld_st": (1, 0, 1),
    "ld_red": (1, 1, 0),
    "ld_red_st": (1, 1, 1),
}


def mma_types_cases(b: Batch) -> list[Case]:
    """Round 5: independent mma per opcode and type on one and four warps, and chained (probe
    mma_types); cycles per instruction on one sub-partition."""
    txt = _sass(b, "probe-mma_types")
    out = []
    for name in MMA_TYPES:
        t = MMA_TYPE_INDEX[name]
        i16 = _k(txt, f"_Z1kILi{t}ELi16EEvjPyPj")
        i1 = _k(txt, f"_Z1kILi{t}ELi1EEvjPyPj")
        for pt, ins, warps, na in (("w1", i16, 1, 16), ("w4", i16, 4, 16), ("chain", i1, 1, 1)):
            out.append(
                _case(
                    b,
                    f"mma_types {name}_{pt}",
                    "mma_types",
                    2,
                    "cycles/instr/subpartition",
                    lambda ins=ins, warps=warps, **kw: ClockMarked(ins, 32 * warps, 1, 1, **kw),
                    per(256 * na),
                    excluded=(
                        "s4 k64 has no native mma on sm_121a: the compiler expands it, no test row"
                        if name == "s32_s4"
                        else None
                    ),
                )
            )
    return out


def row_reduce_cases(b: Batch) -> list[Case]:
    """Round 5: the steps of a row kernel's block at 1 and 11 blocks per SM (probe row_reduce);
    the span of the busiest SM, warm (every load a hit), the chip's shares by blocks in flight."""
    txt = _sass(b, "probe-row_reduce")
    out = []
    for name, (ld, red, st) in ROW_POINTS.items():
        kernel = f"_Z1kILi{ld}ELi{red}ELi{st}EEvPyPKfPfPj"
        ins = _k(txt, kernel)
        regs = regs_of(b, "probe-row_reduce", kernel)
        for grid in (48, 512):

            def build(ins=ins, grid=grid, regs=regs, **kw):
                m = kw.get("machine") or default_machine()
                rmax = occupancy({"threads": 128}, regs, 512, m)
                per_sm = math.ceil(grid / m.sms)
                return BlockMarked(
                    ins,
                    128,
                    per_sm,
                    1,
                    active_sms=min(grid, m.sms * rmax) / min(rmax, per_sm),
                    resident=rmax,
                    grid_blocks=grid,
                    grid_dim=(grid, 1, 1),
                    **kw,
                )

            out.append(
                _case(
                    b,
                    f"row_reduce {name}_g{grid}_span",
                    "row_reduce",
                    4,
                    "cycles",
                    build,
                    _span,
                )
            )
    return out


ST256_GRIDS = (1, 2, 16, 48, 96, 512)


def st256_cases(b: Batch) -> list[Case]:
    """Round 5: 32-byte-per-lane global stores and loads at grids 1-512 (probe st256), warm."""
    txt = _sass(b, "probe-st256")
    out = []
    for name, (ld, st) in {"st": (0, 1), "ld": (1, 0), "ld_st": (1, 1)}.items():
        kernel = f"_Z1kILi{ld}ELi{st}EEvPyPKyS0_Pj"
        ins = _k(txt, kernel)
        regs = regs_of(b, "probe-st256", kernel)
        for grid in ST256_GRIDS:

            def build(ins=ins, grid=grid, regs=regs, **kw):
                m = kw.get("machine") or default_machine()
                rmax = occupancy({"threads": 128}, regs, 0, m)
                per_sm = math.ceil(grid / m.sms)
                return BlockMarked(
                    ins,
                    128,
                    per_sm,
                    1,
                    active_sms=min(grid, m.sms * rmax) / min(rmax, per_sm),
                    resident=rmax,
                    grid_blocks=grid,
                    grid_dim=(grid, 1, 1),
                    **kw,
                )

            out.append(_case(b, f"st256 {name}_g{grid}", "st256", 4, "cycles", build, _span))
    return out


BRANCH_LOOPS = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
BRANCH_FLATS = (64, 256, 1024, 4096)
BRANCH_PASSES = 512
FWD_MASK_ARG = (0, 0x380)  # the `z` argument of the branch probe's fwd kernel


def _warps_case(b, cid, probe, ins, warps, div, unit, const=None) -> Case:
    return _case(
        b,
        cid,
        probe,
        2,
        unit,
        lambda ins=ins, warps=warps, **kw: ClockMarked(
            ins, 32 * warps, 1, 1, const_bank=const, **kw
        ),
        per(div),
    )


def branch_cases(b: Batch) -> list[Case]:
    """Round 6: taken-branch cost and instruction fetch (probe branch), FFMA loops and straight
    lines on 1-32 warps, a forward branch taken and not."""
    txt = _sass(b, "probe-branch")
    out = []
    for n in BRANCH_LOOPS:
        ins = _k(txt, f"_Z4loopILi{n}EEvPyPf")
        # 32 warps at 512 and 1024 are measured, not replayed: millions of issues per row
        for w in (1, 4) if n >= 512 else (1, 4, 32):
            c = _warps_case(b, f"branch loop{n}_w{w}", "branch", ins, w, BRANCH_PASSES, "cycles")
            if w == 32 and 16 <= n <= 256:
                c.excluded = "measured rate above the issue limit: question 9"
            out.append(c)
    for n in BRANCH_FLATS:
        ins = _k(txt, f"_Z4flatILi{n}EEvPyPf")
        for w in (1, 2, 4, 8, 32):
            c = _warps_case(b, f"branch flat{n}_w{w}", "branch", ins, w, n, "cycles")
            if w >= 8:  # warp 0 reads its clock after a barrier, the others may already be running
                c.excluded = "warp 0's clock window is not the block's at several warps per pipe"
            out.append(c)
    ins = _k(txt, "_Z3fwdjPyPf")
    for name, z in (("fwd_taken", 0xFFFFFFFF), ("fwd_not", 0)):
        for w in (1, 4, 32):
            out.append(
                _warps_case(
                    b,
                    f"branch {name}_w{w}",
                    "branch",
                    ins,
                    w,
                    BRANCH_PASSES,
                    "cycles",
                    {FWD_MASK_ARG: z},
                )
            )
    return out


MMA_PASSES = ((2, 2048), (3, 1024), (4, 1024), (5, 1024), (8, 512), (16, 256), (32, 128), (64, 64))
MMA_OTHER = {"fma": 0, "alu": 1, "lds": 2, "ldsm": 3}


def mma_issue_cases(b: Batch) -> list[Case]:
    """Round 6: mma per pass, other instructions beside it on the same warp and on warp 4 (probe
    mma_issue); cycles per mma on warp 0."""
    txt = _sass(b, "probe-mma_issue")
    out = []
    for na, it in MMA_PASSES:
        ins = _k(txt, f"_Z4passILi{na}ELi{it}EEvjPyPj")
        out.append(_warps_case(b, f"mma_issue n{na}", "mma_issue", ins, 1, na * it, "cycles/mma"))
    for name, k in MMA_OTHER.items():
        for m in (1, 2) if name == "ldsm" else (1, 2, 4, 8, 14):
            ins = _k(txt, f"_Z6besideILi{k}ELi{m}EEvjPyPj")
            out.append(
                _warps_case(b, f"mma_issue m{name}_{m}", "mma_issue", ins, 1, 8 * 256, "cycles/mma")
            )
    for name, k in (("none", "n1"), ("fma", "0"), ("alu", "1"), ("lds", "2")):
        ins = _k(txt, f"_Z3nbrILi{k}EEvjPyPj")
        out.append(
            _warps_case(b, f"mma_issue nbr_{name}", "mma_issue", ins, 5, 16 * 256, "cycles/mma")
        )
    return out


MMA_PIPE_SETS = ((1, 1), (2, 1), (4, 1), (8, 1), (1, 2), (1, 4), (1, 8), (2, 2), (4, 4), (8, 8))


def mma_pipe_cases(b: Batch) -> list[Case]:
    """Round 6: the tensor pipe's rate by compiler, operands and warps per sub-partition (probe
    mma_pipe, built by the host's and by the image's nvcc)."""
    out = []
    for unit, tag in (("probe-mma_pipe", "v13_0"), ("probe-mma_pipe_i", "v13_2")):
        txt = _sass(b, unit)
        for sa, sb in MMA_PIPE_SETS:
            ins = _k(txt, f"_Z1kILi{sa}ELi{sb}EEvjPyPj")
            for w in (1, 8):
                name = f"mma_pipe {tag}_a{sa}_b{sb}_w{w}"
                out.append(
                    _warps_case(
                        b, name, "mma_pipe", ins, w, 256 * 16 * max(1, w // 4), "cycles/mma"
                    )
                )
    return out


def lds_narrow_cases(b: Batch) -> list[Case]:
    """Round 6: LDS and STS by element size and lane stride on 1-32 warps (probe lds_narrow)."""
    txt = _sass(b, "probe-lds_narrow")
    out = []
    names = (("LDS.U8", 1), ("LDS.U16", 2), ("LDS.32", 4), ("STS.U16", 2), ("STS.32", 4))
    for kind, (name, elem) in enumerate(names):
        ins = _k(txt, f"_Z1kILi{kind}EEvjjPyPj")
        for st in (elem, 16, 64):
            for w in (1, 4, 32):
                # the arguments `stride` and `zero` enter the lanes' addresses
                out.append(
                    _warps_case(
                        b,
                        f"lds_narrow {name}_s{st}_w{w}",
                        "lds_narrow",
                        ins,
                        w,
                        256 * 8 * w,
                        "cycles/instr/SM",
                        {(0, 0x380): st, (0, 0x384): 0},
                    )
                )
    return out


LAUNCH_SHORT = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 192, 256)


def launch_short_cases(b: Batch) -> list[Case]:
    """Round 6: the launch term around the kernel span where back-to-back launches leave the
    submission rate (probe launch_short); the launch probe's replay on its kernel."""
    txt = _sass(b, "probe-launch_short")

    def row(ins):
        def build(**kw):
            m = kw.get("machine") or default_machine()
            per_sm = math.ceil(48 / m.sms)
            return SM(ins, 128, per_sm, 1, active_sms=min(48, m.sms), grid_blocks=48, **kw)

        return build

    out = []
    for i in LAUNCH_SHORT:
        ins = _k(txt, f"_Z4workILi{i}EEvPfPy")
        out.append(
            _case(
                b,
                f"launch_short work_i{i}",
                "launch_short",
                1,
                "us",
                row(ins),
                lambda sm, r: _launch_kernel(sm.m, r["cycles"] / (_ghz(b, sm.m) * 1e3)),
            )
        )
        for mode in ("b2b", "graph"):
            out.append(
                _case(
                    b,
                    f"launch_short {mode}_work_i{i}",
                    "launch_short",
                    1,
                    "us",
                    row(ins),
                    lambda sm, r, mode=mode: _launch_seq(
                        sm.m, mode, r["cycles"] / (_ghz(b, sm.m) * 1e3)
                    ),
                )
            )
    return out


L2STAGES_TILE = {"block_M": 128, "block_N": 256, "block_K": 32, "threads": 256}


def l2stages_cases(b: Batch) -> list[Case]:
    """Round 2, P-2's second experiment (probe l2share_stages): the lean cp.async loop by stages
    and blocks, one block per SM, warm L2; the copy pattern of TileLang's 128x256x32 tile."""
    txt = _sass(b, "probe-l2share_stages")
    out = []
    for st in (1, 2, 3, 4):
        ins = _k(txt, f"_Z4loadILi{st}EEvPK6__halfS2_")
        h = {**L2STAGES_TILE, "num_stages": st}
        for blocks in (1, 16, 48):
            out.append(
                _case(
                    b,
                    f"l2share_stages s{st}_b{blocks}",
                    "l2share_stages",
                    4,
                    "us",
                    # the main loop runs KT - (S - 1) = 33 - S times (32 k-tiles)
                    lambda ins=ins, h=h, st=st, blocks=blocks, **kw: SM(
                        ins, 256, 1, 33 - st, active_sms=blocks, copy_lines=copy_lines(h), **kw
                    ),
                    lambda sm, r: r["cycles"] / (_ghz(b, sm.m) * 1e3),
                )
            )
    return out


L2REUSE_N = (2, 8, 16, 20, 22, 24, 27, 32, 43)  # steps of 48 x 1024 x 16 B = 0.75 MiB
L2REUSE_KINDS = ("cg", "ca", "cs")


def l2reuse_cases(b: Batch) -> list[Case]:
    """Round 3 (probe l2reuse): F = 0.75 n MiB read by 48 blocks x 1024 threads, one 16-byte load
    per thread per step; the in-kernel span. Modes: cold (after a read flush: every load a DRAM
    miss), warm (back-to-back), in4 (four passes after a flush). The engine is told the miss
    fraction of each mode by the warm rule under test (`l2_warm_miss`): cold 1, warm the rule's,
    in4 (1 + 3 x the rule's) / 4."""
    txt = _sass(b, "probe-l2reuse")
    out = []
    for n in L2REUSE_N:
        nbytes = n * 48 * 1024 * 16
        for k, kind in enumerate(L2REUSE_KINDS):
            for mode in ("cold", "warm", "in4"):
                passes = 4 if mode == "in4" else 1
                ins = _k(txt, f"_Z2rdILi{n}ELi{k}ELi{passes}EEvPK4int4PyPi")

                def build(ins=ins, nbytes=nbytes, mode=mode, kind=kind, passes=passes, **kw):
                    mm = kw.get("machine") or default_machine()
                    w = l2_warm_miss(nbytes, kind, mm)
                    miss = {"cold": 1.0, "warm": w, "in4": (1 + 3 * w) / 4}[mode]
                    secs = nbytes / (mm.C["dram_read_GBs"] * 1e9)
                    return SM(
                        ins,
                        1024,
                        1,
                        1,
                        active_sms=48,
                        ldg_lines=[(4, 512, False)],
                        miss_frac=miss,
                        dram_clk_per_512B=dram_clk(nbytes, secs, 48, mm),
                        **kw,
                    )

                out.append(
                    _case(
                        b,
                        f"l2reuse n{n}_{kind}_{mode}",
                        "l2reuse",
                        5,
                        "us",
                        build,
                        lambda sm, r: r["cycles"] / (_ghz(b, sm.m) * 1e3),
                        limit=200_000_000,
                    )
                )
    return out


LAUNCH_GRIDS = ((1, 128), (48, 128), (96, 128), (512, 128), (1024, 128), (2048, 128), (4096, 128))
LAUNCH_GRIDS += ((8192, 128), (48, 1024), (128, 1024), (256, 1024), (512, 1024))
LAUNCH_WORK = (0, 64, 128, 256, 512, 1024, 4096, 16384)
# round 4: launches in a sequence (b2b: back-to-back, graph: one CUDA graph)
LAUNCH_SEQ_WORK = (0, 64, 256, 1024, 4096, 16384)
LAUNCH_SEQ_GRIDS = ((1, 128), (512, 128), (4096, 128))


def launch_cases(b: Batch) -> list[Case]:
    """Round 3 (probe launch): event time of one launch with a synchronize after it, as the
    test-set harness times kernels; the engine's row time is its replay of the kernel on one SM
    plus its launch term."""
    txt = _sass(b, "probe-launch")
    empty, tiny = _k(txt, "_Z5emptyv"), _k(txt, "_Z4tinyPi")

    def row(ins, threads, grid):
        def build(**kw):
            m = kw.get("machine") or default_machine()
            per_sm = math.ceil(grid / m.sms)
            return SM(ins, threads, per_sm, 1, active_sms=min(grid, m.sms), grid_blocks=grid, **kw)

        return build

    out = []
    for g, t in LAUNCH_GRIDS:
        for name, ins in (("empty", empty), ("tiny", tiny)):
            out.append(
                _case(
                    b,
                    f"launch {name}_g{g}_t{t}",
                    "launch",
                    1,
                    "us",
                    row(ins, t, g),
                    lambda sm, r: r["cycles"] / (_ghz(b, sm.m) * 1e3) + _launch_event(sm.m),
                )
            )
    for g, t in LAUNCH_SEQ_GRIDS:
        out.append(
            _case(
                b,
                f"launch graph_g{g}_t{t}",
                "launch",
                1,
                "us",
                row(empty, t, g),
                lambda sm, r: _launch_seq(sm.m, "graph", r["cycles"] / (_ghz(b, sm.m) * 1e3)),
            )
        )
    for i in LAUNCH_SEQ_WORK:
        ins = _k(txt, f"_Z4workILi{i}EEvPfPy")
        for mode in ("b2b", "graph"):
            out.append(
                _case(
                    b,
                    f"launch {mode}_work_i{i}",
                    "launch",
                    1,
                    "us",
                    row(ins, 128, 48),
                    lambda sm, r, mode=mode: _launch_seq(
                        sm.m, mode, r["cycles"] / (_ghz(b, sm.m) * 1e3)
                    ),
                )
            )
    for i in LAUNCH_WORK:
        ins = _k(txt, f"_Z4workILi{i}EEvPfPy")
        out.append(
            _case(
                b,
                f"launch work_i{i}",
                "launch",
                1,
                "us",
                row(ins, 128, 48),
                lambda sm, r: r["cycles"] / (_ghz(b, sm.m) * 1e3) + _launch_event(sm.m),
            )
        )
    return out


def _launch_event(m: Machine) -> float:
    """The launch term of a KTA batch row: one launch between two events and a synchronize (probe
    launch, round 3; before it the fitted back-to-back `launch_us`)."""
    return m.param(m.desc.get("launch_event_us", m.desc["launch_us"]))


def _launch_seq(m: Machine, mode: str, span_us: float) -> float:
    """Per-launch time of a kernel in a sequence of launches (round 4, probe launch b2b/graph work
    points): `b2b` back-to-back from one stream, `graph` inside one CUDA graph; the kernel's span
    plus the gap between kernels, at least the submission rate. Round 6 (probe launch_short): a
    back-to-back kernel gone before the next launch is ready leaves a smaller gap; one still
    running makes the next wait for it, with a floor."""
    q = m.desc["launch_seq"][mode]
    if "short_below" in q:
        if span_us < m.param(q["short_below"]):
            return max(m.param(q["submit"]), span_us + m.param(q["short_gap"]))
        return max(m.param(q["floor"]), span_us + m.param(q["gap"]))
    return max(m.param(q["submit"]), span_us + m.param(q["gap"]))


def _launch_kernel(m: Machine, span_us: float) -> float:
    """One launch between two events with a synchronize after it (round 6, probe launch_short):
    a kernel shorter than the launch itself is hidden in it, a longer one pays the event gap."""
    q = m.desc["launch_kernel"]
    if span_us < m.param(q["short_below"]):
        return m.param(q["short"])
    return max(m.param(q["floor"]), span_us + _launch_event(m))


def _launch(b: Batch, unit: str) -> dict[str, Any]:
    return json.loads((b.build / "testset" / unit / "launch.json").read_text())


def kernel_cases(b: Batch) -> list[Case]:
    root = b.build / "testset"
    if not root.exists():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if (p / "k.sass").exists()):
        L = _launch(b, d.name)
        regs = regs_of(b, f"testset/{d.name}", L["entry"])
        if L["family"] == "hgemm":
            # round 4: the rotating variants are their own variant group
            rot = " rotating" if L.get("cache") == "rotating" else ""
            group = "testset " + L["set"] if "set" in L else "testset hgemm2048" + rot
            out.append(_hgemm_case(b, d, L, regs, group))
        elif L["family"] == "fp8gemm":
            # kta:fixture byte-equivalent mapping: the HGEMM front end counts 2-byte elements, so
            # an e4m3 tile of block_K bytes is replayed as a fp16 tile of block_K / 2 (same bytes,
            # lines, DRAM traffic and mma count per k-tile); B's K-major layout is not modelled
            eq = {
                **L,
                "pick": {**L["pick"], "block_K": L["pick"]["block_K"] // 2},
                "shape": {**L["shape"], "K": L["shape"]["K"] // 2},
            }
            out.append(_hgemm_case(b, d, eq, regs, "testset fp8gemm2048"))
        elif L["family"] == "sequence":
            out += [_sequence_case(b, d, L, mode) for mode in ("kernel", "b2b", "graph")]
        else:
            # round 4: the small kernels also as launched in a sequence (b2b, graph), own sets
            out += [_generic_case(b, d, L, regs, mode) for mode in ("kernel", "b2b", "graph")]
    return out


def _hgemm_case(
    b: Batch, d: Path, L: dict[str, Any], regs: int, probe: str = "testset hgemm2048"
) -> Case:
    def build(**kw):
        return _KernelRun(
            lambda sm_hook, machine, trace: simulate_kernel(
                d / "k.sass",
                L["pick"],
                L["shape"],
                regs,
                L["smem"],
                warm=L["cache"] == "warm",
                kernel=L["entry"],
                machine=machine,
                sm_hook=sm_hook,
                trace=trace,
            ),
            **kw,
        )

    return _case(
        b,
        f"{d.name} kernel",
        probe,
        7,
        "us",
        build,
        lambda sm, r: r["us"] - _launch_diff(b, sm) - launch_us() + _launch_event(sm.m),
        limit=0,
    )


def _launch_diff(b: Batch, sm: Any) -> float:
    """simulate_kernel converts with the description's kernel clock; reconvert the SM part with
    the batch's clock."""
    if not b.clock_GHz:
        return 0.0
    m = sm.m
    f0 = m.param(m.desc["clock"]["kernel_GHz"])
    sim_us = sm.cycles / (f0 * 1e3)
    return sim_us - sm.cycles / (b.clock_GHz * 1e3)


_l2_cache: dict[tuple, Any] = {}


def _l2_replay(ins: list[Ins], L: dict[str, Any], trips: int, in_flight: int, m: Machine) -> Any:
    cap = m.C["l2_effective_bytes"]
    key = (L["entry"], json.dumps(L, sort_keys=True), len(ins), trips, in_flight, cap)
    if key not in _l2_cache:
        block = tuple(L.get("block", [L["threads"]]))
        acc = walk(ins, L["threads"], trips, tuple(L["grid"]), block, m)
        _l2_cache[key] = replay_walk(acc, in_flight, cap)
    return _l2_cache[key]


def _generic_sm(d: Path, L: dict[str, Any], regs: int, carried: tuple[str, ...], **kw: Any) -> SM:
    """A kernel without a family front end (FA forward, decode attention, GEMV, RMSNorm, softmax):
    the busiest SM of the grid with block replacement, the chip's shares by blocks in flight, the
    main loop's trip count from TileLang's source loop. The L2 (round 5, l2.py): the loads' own
    addresses over the grid through one cache in launch order; rotating launches read that traffic
    at the DRAM rate shared by the blocks in flight (as the dram_rotate probe), warm launches the
    part the measured curve of the read footprint loses (plain loads; evict-first loads their own).
    LDG lines, sectors and L1 hits from the lanes' addresses (sm.SM lane tracking, round 3).
    `carried`: buffers an earlier kernel of a sequence left in the L2 (round 6): loads from them
    hit."""
    ins = _k((d / "k.sass").read_text(), L["entry"])
    src = (d / L.get("source", "kernel.cu")).read_text()
    # the source loop of the kernel's main loop: `k`; a GEMM's k-loop may be called `ko` (MoE)
    lp = main_loop(ins)
    mma = lp and any(x.base in ("HMMA", "QMMA", "IMMA") for x in ins[lp[0] : lp[1] + 1])
    name = "ko?" if mma else "k"
    loops = [
        int(x) for x in re.findall(rf"for \(int {name} = 0; {name} < (\d+); \+\+{name}\)", src)
    ]
    src_trips = max(loops) if loops else 1
    tiles = math.prod(L["grid"])
    m = kw.get("machine") or default_machine()
    h = {"threads": L["threads"]}
    rmax = occupancy(h, regs, L["smem"], m)
    per_sm = math.ceil(tiles / m.sms)
    eq = min(tiles, m.sms * rmax) / min(rmax, per_sm)
    fb = L["footprint_bytes"]
    rotating = L.get("cache") == "rotating"
    kind = "cs" if all(".EF" in x.op for x in ins if x.base == "LDG") else "ca"
    # round 4: the written buffers (gen.py `write_bytes`) are DRAM traffic beside the reads
    wb = L.get("write_bytes", 0) if rotating else 0
    if m.P["l2_replay"]:
        r = _l2_replay(ins, L, src_trips, m.sms * rmax, m)
        scale = 1.0 if rotating else l2_warm_miss(r.footprint, kind, m)
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
    elif rotating:
        rb, miss = fb - wb, 1.0
    else:  # warm (round 3, probe l2reuse): the re-read misses by the footprint
        rb, miss = fb, l2_warm_miss(fb, kind, m)
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
        src_trips if main_loop(ins) else 1,
        active_sms=eq,
        resident=rmax,
        grid_blocks=tiles,
        grid_dim=tuple(L["grid"]),
        block_dim=tuple(L.get("block", [L["threads"]])),
        l1_bytes=m.C["l1_unified_bytes"] - rmax * L["smem"],
        **kw,
    )


def _generic_case(b: Batch, d: Path, L: dict[str, Any], regs: int, mode: str = "kernel") -> Case:
    def build(**kw):
        return _generic_sm(d, L, regs, (), **kw)

    def read(sm, r):
        span = r["cycles"] / (_ghz(b, sm.m) * 1e3)
        return _launch_kernel(sm.m, span) if mode == "kernel" else _launch_seq(sm.m, mode, span)

    return _case(
        b,
        f"{d.name} {mode}",
        f"testset {L['family']}" + ("" if mode == "kernel" else f" {mode}"),
        7,
        "us",
        build,
        read,
    )


def _carried(L: dict[str, Any], m: Machine) -> list[tuple[str, ...]]:
    """Per kernel of a sequence, the buffers earlier kernels left in the L2: what they read and
    wrote, oldest dropped beyond the L2's effective capacity."""
    held: dict[str, int] = {}
    out = []
    for k in L["kernels"]:
        out.append(tuple(held))
        for n in (*k["reads"], *k["writes"]):
            held.pop(n, None)
            held[n] = L["buffers"][n]
        while sum(held.values()) > m.C["l2_effective_bytes"]:
            held.pop(next(iter(held)))
    return out


def _sequence_case(b: Batch, d: Path, L: dict[str, Any], mode: str) -> Case:
    """Kernels in order, the L2 state carried over (`_carried`), per sequence: a sequence of one
    launch per kernel, the gaps of the launch mode; `kernel` = one sequence between two events."""

    def build(**kw):
        return _SeqRun(
            [
                lambda c=c, k=k, **a: _generic_sm(
                    d,
                    {**k, "cache": L["cache"]},
                    regs_of(b, f"testset/{d.name}", k["entry"]),
                    c,
                    **a,
                )
                for k, c in zip(
                    L["kernels"], _carried(L, kw.get("machine") or default_machine()), strict=True
                )
            ],
            **kw,
        )

    def read(sm, r):
        spans = [c / (_ghz(b, sm.m) * 1e3) for c in sm.spans]
        if mode == "kernel":
            gaps = sum(_launch_seq(sm.m, "b2b", x) for x in spans[:-1])
            return gaps + _launch_kernel(sm.m, spans[-1])
        return sum(_launch_seq(sm.m, mode, x) for x in spans)

    return _case(
        b,
        f"{d.name} {mode}",
        f"testset {L['set']}" + ("" if mode == "kernel" else f" {mode}"),
        7,
        "us",
        build,
        read,
        limit=0,
    )


class _KernelRun:
    """A Case build result for whole-kernel front ends: run() calls the front end and keeps the
    SM it simulated (for attribution) as `sm`; `cycles` and `m` are the SM's."""

    def __init__(self, fn, machine=None, trace: bool = False):
        self.fn = fn
        self.machine, self.trace = machine or default_machine(), trace
        self.sm: SM | None = None

    def run(self, limit: int = 0) -> dict[str, Any]:
        def hook(sm: SM) -> None:
            self.sm = sm

        return self.fn(hook, self.machine, self.trace)

    @property
    def m(self) -> Machine:
        return self.machine

    @property
    def cycles(self) -> float:
        assert self.sm is not None
        return self.sm.cycles


class _SeqRun(_KernelRun):
    """A Case build result for a sequence of kernels: `spans` the cycles of each, the SM kept for
    attribution the longest."""

    def __init__(self, makers, machine=None, trace: bool = False):
        super().__init__(None, machine, trace)
        self.makers = makers
        self.spans: list[float] = []

    def run(self, limit: int = 0) -> dict[str, Any]:
        best = None
        for make in self.makers:
            sm = make(machine=self.machine, trace=self.trace)
            r = sm.run()
            self.spans.append(sm.cycles)
            if best is None or sm.cycles > best[0].cycles:
                best = (sm, r)
        assert best is not None
        self.sm = best[0]
        return best[1]


def all_cases(batch_dir: Path) -> list[Case]:
    b = load(batch_dir)
    out: list[Case] = []
    builders = [
        ("probe-smem_ldsts", smem_cases),
        ("probe-shfl", shfl_cases),
        ("probe-ex2_ffma", ex2_cases),
        ("probe-namedbar", bar_cases),
        ("probe-l1hit", l1_cases),
        ("probe-store_tail", store_cases),
        ("probe-l2share_copy", l2share_cases),
        ("probe-dram_rotate", dram_cases),
        ("probe-dram_rw", dram_rw_cases),
        ("probe-mma_pair", mma_pair_cases),
        ("probe-mma_types", mma_types_cases),
        ("probe-row_reduce", row_reduce_cases),
        ("probe-st256", st256_cases),
        ("probe-branch", branch_cases),
        ("probe-mma_issue", mma_issue_cases),
        ("probe-lds_narrow", lds_narrow_cases),
        ("probe-launch_short", launch_short_cases),
        ("probe-mma_pipe", mma_pipe_cases),
        ("probe-l2share_stages", l2stages_cases),
        ("probe-l2reuse", l2reuse_cases),
        ("probe-launch", launch_cases),
        ("testset", kernel_cases),
    ]
    for unit, f in builders:  # a batch built with `--only` has a subset of the units
        if (b.build / unit).is_dir():
            out += f(b)
    return out


def run_case(case: Case, **kw: Any) -> tuple[float, Any, dict]:
    sm = case.build(**kw)
    r = sm.run(limit=case.limit) if case.limit else sm.run()
    target = sm.sm if isinstance(sm, _KernelRun) else sm
    return case.read(sm, r), target, r
