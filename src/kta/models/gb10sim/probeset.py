"""Replays of KTA's own measurement batches (docs/design/perfmodel/LOOP.md products 5 and 6): the
probes under `perfmodel/probes/` and the test-set kernels under `perfmodel/testset/`, each from the
SASS the batch built (`perfmodel/batches/<batch>/build/<unit>/k.sass`) with the launch the batch
ran, against the medians the batch measured (`measured.json`, empty before the run: the replay
is then the prediction written into the batch's entry blocks).

Cycle-valued probes compare cycles directly. Microsecond-valued rows convert with the clock the
batch measured (`clock.chain_GHz`) when there is one, else with the description's kernel clock.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from .kernel import (
    copy_lines,
    dram_clk,
    dram_mix_secs,
    generic_kernel,
    machine_trips,
    occupancy,
    sass_trips,
)
from .l2 import rotating_miss as l2_rotating_miss
from .l2 import warm_miss as l2_warm_miss
from .machine import Machine, default_machine
from .probes import Case, ClockMarked, marked_cycles
from .sass import Ins, parse
from .sm import SM

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


@cache
def _read(path: Path) -> str:
    return path.read_text()


def _sass(b: Batch, unit: str) -> str:
    return _read(b.build / unit / "k.sass")


def regs_of(b: Batch, unit: str, kernel: str) -> int:
    m = re.search(
        rf"Function {re.escape(kernel)}:\s*REG:(\d+)", _read(b.build / unit / "res-usage.txt")
    )
    if not m:
        raise ValueError(f"{unit}: no register count for {kernel}")
    return int(m.group(1))


def _ghz(b: Batch, m: Machine) -> float:
    return b.clock_GHz or m.param(m.desc["clock"]["kernel_GHz"])


@cache
def _k(ins_text: str, kernel: str) -> list[Ins]:
    ins = parse(ins_text, kernel=kernel)
    if not ins:
        raise ValueError(f"no SASS for {kernel}")
    return ins


# A probe case below is one line per sweep point: `cid` is "<probe> <point>", the probe's build unit
# is probe-<probe>, `kernel` its mangled entry. The helpers differ in how the SM is set up and read.
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


def _unit_ins(b: Batch, cid: str, kernel: str, src: str | None = None) -> tuple[str, list[Ins]]:
    probe = cid.split()[0]
    return probe, _k(_sass(b, src or f"probe-{probe}"), kernel)


def per(div: float):
    return lambda sm, r: marked_cycles(sm, r) / div


def _time(b: Batch, launch: str | None = None):
    """Read a run as microseconds at the batch's clock: the kernel's span, or with a launch term
    (`event`: `_launch_event`; `kernel`: `_launch_kernel`; `b2b`, `graph`: `_launch_seq`)."""

    def read(sm, r):
        span = r["cycles"] / (_ghz(b, sm.m) * 1e3)
        if launch == "event":
            return span + _launch_event(sm.m)
        if launch == "kernel":
            return _launch_kernel(sm.m, span)
        return _launch_seq(sm.m, launch, span) if launch else span

    return read


def _marked(
    b, cid, kernel, threads, div, unit, level=2, src=None, trips=1, excluded=None, **sm
) -> Case:
    """Timed by warp 0's clock reads, `div` operations between them (or a read function); `sm`
    goes to the simulator."""
    probe, ins = _unit_ins(b, cid, kernel, src)
    build = lambda **kw: ClockMarked(ins, threads, 1, trips, **sm, **kw)  # noqa: E731
    read = div if callable(div) else per(div)
    return _case(b, cid, probe, level, unit, build, read, excluded=excluded)


def _us(b, cid, kernel, threads, read, level=4, trips=1, **sm) -> Case:
    """One run of the plain simulator, in microseconds."""
    probe, ins = _unit_ins(b, cid, kernel)
    return _case(
        b, cid, probe, level, "us", lambda **kw: SM(ins, threads, 1, trips, **sm, **kw), read
    )


def _stream(
    b,
    cid,
    kernel,
    threads,
    reads,
    miss: Any = 1.0,
    writes=0,
    blocks=1,
    read=None,
    limit=200_000_000,
):
    """A stream through the DRAM path at the chip's share (48 blocks in flight): `reads` and
    `writes` bytes per launch, `miss` a fraction or a function of the machine."""
    probe, ins = _unit_ins(b, cid, kernel)

    def build(**kw):
        m = kw.get("machine") or default_machine()
        return SM(
            ins,
            threads,
            blocks,
            1,
            active_sms=48,
            ldg_lines=[(4, 512, False)],
            miss_frac=miss if isinstance(miss, float) else miss(m),
            dram_clk_per_512B=dram_clk(reads, dram_mix_secs(reads, writes, m), 48, m),
            **kw,
        )

    return _case(b, cid, probe, 5, "us", build, read or _time(b, "event"), limit=limit)


def _launched(b, cid, ins, threads, grid, launch) -> Case:
    """One SM's replay of a `grid`-block launch at the chip's share, with a launch term."""

    def build(**kw):
        m = kw.get("machine") or default_machine()
        per_sm = math.ceil(grid / m.sms)
        return SM(ins, threads, per_sm, 1, active_sms=min(grid, m.sms), grid_blocks=grid, **kw)

    return _case(b, cid, cid.split()[0], 1, "us", build, _time(b, launch))


def smem_cases(b: Batch) -> list[Case]:
    return [
        _marked(
            b,
            f"smem_ldsts {name}_w{w}",
            f"_Z1kILi{kind}EEvjPyPj",
            32 * w,
            ITERS * 8 * w,
            "cycles/instr/SM",
            1 if w == 1 else 2,
        )
        for kind, name in enumerate(["LDS.32", "LDS.64", "LDS.128", "STS.32", "STS.128"])
        for w in SMEM_WARPS
    ]


def shfl_cases(b: Batch) -> list[Case]:
    tput = "_Z1kILi8EEvPyPj"
    return [
        _marked(b, "shfl latency_w1", "_Z1kILi1EEvPyPj", 32, ITERS * 5, "cycles/hop", 1),
        *(
            _marked(b, f"shfl tput_w{w}", tput, 32 * w, ITERS * 5 * 8 * w, "cycles/instr/SM")
            for w in (1, 4, 8, 16, 32)
        ),
    ]


def ex2_cases(b: Batch) -> list[Case]:
    out = []
    for e, f in ((8, 0), (8, 8), (8, 16), (8, 32), (0, 32), (4, 32)):
        cid = f"ex2_ffma E{e}_F{f}"
        # 8 warps per sub-partition issue at least E + F + 3 (loop) instructions per pass each
        floor = 8 * (e + f + 3)
        meas = b.measured.get(cid, float("nan"))
        excluded = None
        if meas < floor:
            excluded = (
                f"measured {meas:.0f} cycles is below the issue limit {floor} (one warp "
                "instruction per cycle per sub-partition): open question"
            )
        kernel = f"_Z1kILi{e}ELi{f}EEvPyPf"
        out.append(_marked(b, cid, kernel, 1024, ITERS, "cycles/pass/smsp", 3, excluded=excluded))
    return out


def _pair_sum(sm: SM, r: dict) -> float:
    """namedbar arrive: warp 0 reads the clock before and after each bar.arrive."""
    marks = getattr(sm, "marks", [])
    pairs = list(zip(marks[0::2], marks[1::2], strict=False))
    return sum(t1 - t0 for t0, t1 in pairs) / BAR_ITERS


def bar_cases(b: Batch) -> list[Case]:
    return [
        *(
            _marked(b, f"namedbar sync_w{w}", "_Z6sync_kPy", 32 * w, BAR_ITERS, "cycles/barrier", 1)
            for w in (2, 4, 8, 16, 32)
        ),
        # the loop counter starts from an HFMA2 used as a move, which the value tracking does not
        # evaluate: the loop's trip count is given
        *(
            _marked(
                b,
                f"namedbar arrive_L{L}",
                f"_Z8arrive_kILi{L}EEvPyPf",
                256,
                _pair_sum,
                "cycles/arrive",
                3,
                trips=BAR_ITERS,
            )
            for L in (0, 64, 256)
        ),
    ]


def l1_cases(b: Batch) -> list[Case]:
    hit = [(4, 512, True)]
    return [
        _marked(
            b, "l1hit chase", "_Z5chasePKjPy", 32, HOPS, "cycles/hop", 1, ldg_lines=[(1, 4, True)]
        ),
        *(
            _marked(
                b,
                f"l1hit tput_w{w}",
                "_Z4tputPK4int4PyPi",
                32 * w,
                ITERS * 8 * w,
                "cycles/instr/SM",
                ldg_lines=hit,
            )
            for w in (4, 8, 16, 32)
        ),
    ]


def store_cases(b: Batch) -> list[Case]:
    return [
        _us(
            b,
            f"store_tail {name}_b{blocks}",
            f"_Z2stILi{mode}EEvPji",
            256,
            _time(b),
            active_sms=blocks,
            const_bank={(0, 0x388): 8},
        )
        for mode, name in ((0, "tl"), (1, "v16"))
        for blocks in (1, 2, 8, 16, 32, 48)
    ]


def l2share_cases(b: Batch) -> list[Case]:
    ins = _k(_sass(b, "probe-l2share_copy"), "matmul_kernel")
    trips, lines = sass_trips(ins, L2SHARE_PICK, 15), copy_lines(L2SHARE_PICK)
    return [
        _us(
            b,
            f"l2share_copy b{blocks}",
            "matmul_kernel",
            256,
            _time(b),
            trips=trips,
            active_sms=blocks,
            copy_lines=lines,
        )
        for blocks in (1, 16, 32, 48)
    ]


def dram_cases(b: Batch) -> list[Case]:
    return [
        _stream(b, f"dram_rotate t{t}", f"_Z6streamILi{t}EEvPK4int4Pi", t, DRAM_BYTES)
        for t in (32, 64, 128, 256, 512, 1024)
    ]


def dram_burst_cases(b: Batch) -> list[Case]:
    """Round 7: the dram_rotate replay at shorter bursts per launch (probe dram_burst)."""
    return [
        _stream(
            b, f"dram_burst t{t}_n{n}", f"_Z6streamILi{t}ELi{n}EEvPK4int4Pi", t, 48 * 1024 * 16 * n
        )
        for t in (256, 1024)
        for n in (5, 10, 21, 42, 85)
    ]


def rot_depth_cases(b: Batch) -> list[Case]:
    """Round 7: a stream of n x 786 KB per launch whose start walks a window of D MiB."""

    def row(n, mib):
        nbytes = 48 * 1024 * 16 * n
        # the walk restarts at the last whole chunk that fits the window
        window = max(nbytes, mib * 2**20 // nbytes * nbytes)
        return _stream(
            b,
            f"rot_depth n{n}_D{mib}",
            f"_Z6streamILi{n}EEvPK4int4Pi",
            1024,
            nbytes,
            miss=lambda m: l2_rotating_miss(window, "ca", m),
        )

    return [
        row(n, mib) for n in (10, 21) for mib in (1, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384)
    ]


DRAM_RW_UN = 96 * 512 * 40  # dram_rw: int4 per plane
DRAM_RW_MIX = ((2, 0), (2, 1), (1, 1), (1, 2), (0, 2))


def dram_rw_cases(b: Batch) -> list[Case]:
    """Round 4 (probe dram_rw): NR read and NW written planes of 30 MiB, rotating (every read a
    miss, every written line back to DRAM), 96 blocks x 512 threads."""
    return [
        _stream(
            b,
            f"dram_rw r{nr}_w{nw}",
            f"_Z2rwILi{nr}ELi{nw}EEvPK4int4PS0_Pi",
            512,
            nr * DRAM_RW_UN * 16,
            writes=nw * DRAM_RW_UN * 16,
            blocks=2,
            limit=400_000_000,
        )
        for nr, nw in DRAM_RW_MIX
    ]


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


def _span_case(
    b, cid, kernel, threads, grid, smem=0, stack=None, smem_in_l1=True, excluded=None
) -> Case:
    """The busiest SM of a grid, block replacement at the kernel's occupancy; the blocks' span.
    Shared memory takes its bytes from the L1 unless `smem_in_l1` is off."""
    probe, ins = _unit_ins(b, cid, kernel)
    regs = regs_of(b, f"probe-{probe}", kernel)

    def build(**kw):
        m = kw.get("machine") or default_machine()
        rmax = occupancy({"threads": threads}, regs, smem, m)
        per_sm = math.ceil(grid / m.sms)
        return BlockMarked(
            ins,
            threads,
            per_sm,
            1,
            active_sms=min(grid, m.sms * rmax) / min(rmax, per_sm),
            resident=rmax,
            grid_blocks=grid,
            grid_dim=(grid, 1, 1),
            l1_bytes=m.C["l1_unified_bytes"] - rmax * (smem if smem_in_l1 else 0),
            stack_bytes=stack,
            **kw,
        )

    return _case(b, cid, probe, 4, "cycles", build, _span, excluded=excluded)


def mma_pair_cases(b: Batch) -> list[Case]:
    """Round 2: the HGEMM warp tile's mma + ldmatrix loop from shared memory, one block, one per
    SM, two per SM with a start skew (probe mma_pair). The replay is one SM; the grid gives the
    block indices (blocks 48..95 run the skew) and the chip share."""
    out = []

    def row(ins, name, grid, per_sm):
        def build(**kw):
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
            out.append(_case(b, f"mma_pair {name}_{what}", "mma_pair", 3, "cycles", build, read))

    for skew in MMA_SKEWS:
        ins = _k(_sass(b, "probe-mma_pair"), f"_Z1kILi{skew}EEvjPyPf")
        if skew == 0:
            row(ins, "b1", 1, 1)
            row(ins, "b48", 48, 1)
        row(ins, f"b96_s{skew}", 96, 2)
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
    return [
        _marked(
            b,
            f"mma_types {name}_{pt}",
            f"_Z1kILi{MMA_TYPE_INDEX[name]}ELi{na}EEvjPyPj",
            32 * warps,
            256 * na,
            "cycles/instr/subpartition",
            excluded=(
                "s4 k64 has no native mma on sm_121a: the compiler expands it, no test row"
                if name == "s32_s4"
                else None
            ),
        )
        for name in MMA_TYPES
        for pt, warps, na in (("w1", 1, 16), ("w4", 4, 16), ("chain", 1, 1))
    ]


def row_reduce_cases(b: Batch) -> list[Case]:
    """Round 5: the steps of a row kernel's block at 1 and 11 blocks per SM (probe row_reduce);
    the span of the busiest SM, warm (every load a hit), the chip's shares by blocks in flight."""
    return [
        _span_case(
            b,
            f"row_reduce {name}_g{grid}_span",
            f"_Z1kILi{ld}ELi{red}ELi{st}EEvPyPKfPfPj",
            128,
            grid,
            smem=512,
            smem_in_l1=False,
        )
        for name, (ld, red, st) in ROW_POINTS.items()
        for grid in (48, 512)
    ]


ST256_GRIDS = (1, 2, 16, 48, 96, 512)


def st256_cases(b: Batch) -> list[Case]:
    """Round 5: 32-byte-per-lane global stores and loads at grids 1-512 (probe st256), warm."""
    return [
        _span_case(b, f"st256 {name}_g{g}", f"_Z1kILi{ld}ELi{st}EEvPyPKyS0_Pj", 128, g)
        for name, (ld, st) in {"st": (0, 1), "ld": (1, 0), "ld_st": (1, 1)}.items()
        for g in ST256_GRIDS
    ]


L2MIX_POINTS = (
    *((x, 0, 32, 32, 0) for x in (1, 2, 4)),
    *((0, x, 32, 32, 0) for x in (1, 2, 4)),
    *(
        (a, c, 32, 32, p)
        for p in (0, 1)
        for a, c in ((4, 1), (2, 1), (1, 1), (2, 2), (1, 2), (1, 4))
    ),
    *(
        pt
        for w, d in ((16, 16), (16, 64), (32, 64), (4, 4))
        for pt in ((0, 2, w, d, 0), (2, 2, w, d, 0), (2, 2, w, d, 1))
    ),
)


def l2mix_cases(b: Batch) -> list[Case]:
    """Round 7: loads and stores per thread by byte ratio, store shape and phase (probe l2mix)."""
    return [
        _span_case(
            b,
            f"l2mix L{ld}_S{st}_W{w}_D{d}_P{p}_g{g}",
            f"_Z1kILi{ld}ELi{st}ELi{w}ELi{d}ELi{p}EEvPyPKyS0_Pj",
            128,
            g,
        )
        for ld, st, w, d, p in L2MIX_POINTS
        for g in (96, 512)
    ]


def spill_cases(b: Batch) -> list[Case]:
    """Round 7: stack words loaded and stored (probe spill)."""
    return [
        _span_case(
            b, f"spill t{t}_n{n}_g{g}_s{s}", f"_Z1kILi{t}ELi{n}EEvPyPj", t, g, s * 1024, n * 4
        )
        for t in (64, 256)
        for n in (16, 64, 256, 1024)
        for g in (1, 48, 192)
        for s in (0, 96)
    ]


def warp_map_cases(b: Batch) -> list[Case]:
    """Round 8: which sub-partition a block's warps go to (probe warp_map)."""
    out = []
    for name, kern, pts in (
        ("ex2", "x", ((32, 1), (32, 2), (32, 4), (32, 8), (64, 1), (64, 2), (64, 4), (96, 1))),
        ("ex2", "x", ((96, 2), (128, 1), (128, 2))),
        ("mma", "m", ((32, 1), (32, 2), (32, 4), (64, 1), (64, 2), (64, 4), (128, 1), (128, 2))),
    ):
        for t, n in pts:
            cid = f"warp_map {name}_t{t}_n{n}"
            out.append(_span_case(b, cid, f"_Z1{kern}ILi{t}EEvPyPf", t, 48 * n))
    return out


COPY_DRAM_BYTES = 8 * 128 * 16 * 16  # per block: copies, threads, bytes, steps


def copy_dram_cases(b: Batch) -> list[Case]:
    """Round 8: one-stage cp.async from DRAM with arithmetic between the steps (probe copy_dram):
    every read of a `rot` launch misses, every read of a `warm` launch hits."""

    def case(rot, z, g):
        cid = f"copy_dram {'rot' if rot else 'warm'}_z{z}_g{g}"
        probe, ins = _unit_ins(b, cid, f"_Z1kILi{z}EEvPyPKcPf")

        def build(**kw):
            m = kw.get("machine") or default_machine()
            rb = g * COPY_DRAM_BYTES
            return BlockMarked(
                ins,
                128,
                1,
                16,
                active_sms=g,
                copy_lines=[4],
                miss_frac=1.0 if rot else 0.0,
                dram_clk_per_512B=dram_clk(rb, dram_mix_secs(rb, 0, m), g, m),
                burst_bytes=g * COPY_DRAM_BYTES / 16 if rot else 0.0,
                **kw,
            )

        return _case(b, cid, probe, 4, "cycles", build, _span)

    return [case(r, z, g) for r in (0, 1) for g in (1, 16, 48) for z in (0, 64, 256, 1024)]


FIRST_READER_Z = (0, 4, 8, 16, 24, 32, 48, 64)
FIRST_STEPS = 64


def first_reader_cases(b: Batch) -> list[Case]:
    """Round 9: a follower warp reads the lines a lead warp read Z passes earlier (probe
    first_reader): cycles per step of warp 0 between its clock reads."""

    def per_step(sm, r):
        return (
            sum(q - p for p, q in zip(sm.marks[0::2], sm.marks[1::2], strict=False)) / FIRST_STEPS
        )

    def case(rot, kind, z, lead):
        cid = f"first_reader {'rot' if rot else 'warm'}_{kind}_z{z}_{'lead' if lead else 'alone'}"
        k = 0 if kind == "ldg" else 1
        name = f"_Z1kILi{z}ELi{k}ELi{1 if lead else 0}EEvPyPKcPf"
        probe, ins = _unit_ins(b, cid, name)
        miss = {i: 1.0 if rot else 0.0 for i, x in enumerate(ins) if x.base in ("LDG", "LDGSTS")}
        pcs = dict(miss)

        def build(**kw):
            m = kw.get("machine") or default_machine()
            return ClockMarked(
                ins,
                T_FIRST,
                1,
                FIRST_STEPS,
                copy_lines=[4],
                miss_by_pc=pcs,
                dram_clk_per_512B=dram_clk(FIRST_STEPS * 512, dram_mix_secs(512, 0, m), 1, m),
                **kw,
            )

        return _case(b, cid, probe, 4, "cycles", build, per_step)

    return [
        case(r, kind, z, lead)
        for r in (0, 1)
        for z in FIRST_READER_Z
        for kind in ("ldg", "cp")
        for lead in (1, 0)
    ]


T_FIRST = 128
FIRST_SHARE_BLOCK = 8 * 128 * 16 * 16  # copies, threads, bytes, steps


LOAD_Z = (0, 16, 64, 256, 1024)
LOAD_G = (1, 12, 24, 48, 96, 144)
LOAD_STEPS = 32


def load_curve_cases(b: Batch, name: str = "load_curve", gs=LOAD_G, zs=LOAD_Z) -> list[Case]:
    """Round 10: a lone copy's cycles at Z passes of spacing between the other warps' copies, on G
    blocks (probes load_curve, load_sat), warp 0 of the busiest SM's first block."""

    def per_step(sm, r):
        return sum(q - p for p, q in zip(sm.marks[0::2], sm.marks[1::2], strict=False)) / LOAD_STEPS

    def case(g, z):
        cid = f"{name} g{g}_z{z}"
        probe, ins = _unit_ins(b, cid, f"_Z1kILi{z}EEvPyPKcPf")

        def build(**kw):
            m = kw.get("machine") or default_machine()
            per = math.ceil(g / m.sms)
            return ClockMarked(
                ins,
                128,
                per,
                LOAD_STEPS,
                active_sms=min(g, m.sms * per) / per,
                resident=per,
                grid_blocks=g,
                grid_dim=(g, 1, 1),
                copy_lines=[4],
                **kw,
            )

        return _case(b, cid, probe, 4, "cycles", build, per_step)

    return [case(g, z) for g in gs for z in zs]


def load_sat_cases(b: Batch) -> list[Case]:
    return load_curve_cases(b, "load_sat", (12, 24, 48, 96, 144), (0, 64, 256))


def copy_depth_cases(b: Batch) -> list[Case]:
    """Round 10: K copies issued with no wait (probe copy_depth): warp 0's cycles to issue them,
    and to the last landing."""

    def case(rot, t, k, landed):
        cid = f"copy_depth {'rot' if rot else 'warm'}_t{t}_k{k}" + ("_landed" if landed else "")
        probe, ins = _unit_ins(b, cid, f"_Z1kILi{k}ELi{t}EEvPyPKc")
        pcs = {i: 1.0 if rot else 0.0 for i, x in enumerate(ins) if x.base == "LDGSTS"}

        def build(**kw):
            m = kw.get("machine") or default_machine()
            clk = dram_clk(512, dram_mix_secs(512, 0, m), 1, m)
            return ClockMarked(
                ins, t, 1, 1, copy_lines=[4], miss_by_pc=pcs, dram_clk_per_512B=clk, **kw
            )

        def read(sm, r):
            return sm.marks[2 if landed else 1] - sm.marks[0]

        return _case(b, cid, probe, 4, "cycles", build, read)

    return [
        case(r, t, k, x)
        for r in (0, 1)
        for k in (4, 8, 16, 32, 64, 128, 256)
        for t in (32, 128)
        for x in (False, True)
    ]


def first_share_cases(b: Batch) -> list[Case]:
    """Round 9: shared tiles at the chip's scale (probe first_share): the wave of 48 blocks of
    `share`, the waves of `seq` on one block per SM, the steps of `pieces` on one block."""

    def wave(rot, d, m):
        """A wave reading 48 / d tiles: every slice misses when rotating, d blocks read it."""
        rb = 48 / d * FIRST_SHARE_BLOCK
        clk = dram_clk(rb, dram_mix_secs(rb, 0, m), 48, m)
        return ({None: (1.0 if rot else 0.0, d)}, clk, rb / 16 if rot else 0.0)

    def share(rot, d, z):
        cid = f"first_share {'rot' if rot else 'warm'}_share_d{d}_z{z}"
        probe, ins = _unit_ins(b, cid, f"_Z5shareILi{z}ELi{d}ELi0EEvPyPKcPf")

        def build(**kw):
            m = kw.get("machine") or default_machine()
            return BlockMarked(
                ins,
                128,
                1,
                16,
                active_sms=48,
                copy_lines=[4],
                wave_miss=[wave(rot, d, m)] if m.P["miss_by_wave"] else None,
                miss_frac=(1.0 / d if rot else 0.0),
                dram_clk_per_512B=wave(rot, d, m)[1],
                **kw,
            )

        return _case(b, cid, probe, 4, "cycles", build, _span)

    def seq(rot, w, part, d=4):
        tag = "d2" if d == 2 else ""
        cid = f"first_share {'rot' if rot else 'warm'}_seq{tag}_w{w}_{part}"
        probe, ins = _unit_ins(b, cid, f"_Z5shareILi0ELi{d}ELi96EEvPyPKcPf")

        def build(**kw):
            m = kw.get("machine") or default_machine()
            first = wave(rot, d, m)
            # 98304 B of shared memory: one block per SM, waves of 48; tile by block modulo 96:
            # the first two waves read new tiles, the others what those left in the L2
            later = ({None: (0.0, d)}, *first[1:])
            f = 1.0 / (d * w) if rot else 0.0
            return BlockMarked(
                ins,
                128,
                2 * w,
                16,
                active_sms=48,
                copy_lines=[4],
                resident=1,
                grid_blocks=96 * w,
                grid_dim=(96 * w, 1, 1),
                wave_miss=[first, first, *[later] * (2 * w - 2)] if m.P["miss_by_wave"] else None,
                miss_frac=f,
                dram_clk_per_512B=first[1],
                **kw,
            )

        def read(sm, r):
            k = 0 if part == "first" else 2 * w - 1
            v = sm.bmarks.get(k, [])
            return v[-1] - v[0] if len(v) > 1 else float("nan")

        return _case(b, cid, probe, 4, "cycles", build, read)

    def pieces(rot, s):
        cid = f"first_share {'rot' if rot else 'warm'}_pieces_s{s}"
        probe, ins = _unit_ins(b, cid, f"_Z6piecesILi{s}EEvPyPKcPf")

        def build(**kw):
            m = kw.get("machine") or default_machine()
            rb = 128 * 64 * 16
            return BlockMarked(
                ins,
                128,
                1,
                64,
                copy_lines=[1],
                miss_frac=1.0 if rot else 0.0,
                dram_clk_per_512B=dram_clk(rb, dram_mix_secs(rb, 0, m), 1, m),
                burst_bytes=128 * 16 if rot else 0.0,
                **kw,
            )

        return _case(b, cid, probe, 4, "cycles", build, lambda sm, r: _span(sm, r) / 64)

    # r9 stopped before `seq`, r9c added the 2-block point
    text = _sass(b, "probe-first_share")
    seqs = "_Z5shareILi0ELi4ELi96EEvPyPKcPf" in text
    d2 = "_Z5shareILi0ELi2ELi96EEvPyPKcPf" in text
    return [
        *(share(r, d, 0) for r in (0, 1) for d in (1, 2, 4, 8, 16, 48)),
        *(share(r, d, 256) for r in (0, 1) for d in (1, 4, 16, 48)),
        *(seq(r, w, p) for r in (0, 1) for w in (2, 4) for p in ("first", "last") if seqs),
        *(seq(r, 2, p, 2) for r in (0, 1) for p in ("first", "last") if d2),
        *(pieces(r, s) for r in (0, 1) for s in (16, 32, 64, 128)),
    ]


STORE_SPREAD_POINTS = (
    *((4, r, 16) for r in (128, 256, 512, 1024, 2048, 4096, 8192)),
    (4, 4096, 32),
    (4, 4096, 8),
    (2, 4096, 8),
    (8, 4096, 32),
    (16, 4096, 64),
    (32, 4096, 128),
    (8, 512, 32),
    (16, 512, 64),
    (32, 512, 128),
)


def store_spread_cases(b: Batch) -> list[Case]:
    """Round 8: 4-byte stores, lanes in rows (probe store_spread)."""
    return [
        _span_case(
            b, f"store_spread L{ln}_R{r}_S{st}_g{g}", f"_Z1kILi{ln}ELi{r}ELi{st}EEvPyPjS1_", 128, g
        )
        for ln, r, st in STORE_SPREAD_POINTS
        for g in (256, 512)
    ]


ARITH_POINTS = (
    *((2, 0, a) for a in (0, 32, 128, 512)),
    *((2, 2, a) for a in (0, 32, 128, 512)),
    *((0, 2, a) for a in (32, 128, 512)),
)


def arith_beside_cases(b: Batch) -> list[Case]:
    """Round 8: FFMA between loads and stores of a row kernel's shape (probe arith_beside)."""
    return [
        _span_case(
            b,
            f"arith_beside L{ld}_S{st}_A{a}_g{g}",
            f"_Z1kILi{ld}ELi{st}ELi{a}EEvPyPKyS0_Pj",
            128,
            g,
        )
        for ld, st, a in ARITH_POINTS
        for g in (96, 512)
    ]


def mma_ldsm_cases(b: Batch) -> list[Case]:
    """Round 8: one tile's ldmatrix and mma per pass over 64-512 threads (probe mma_ldsm)."""
    pts = ((64, 128), (128, 64), (256, 32), (512, 16))
    return [
        _marked(
            b,
            f"mma_ldsm t{t}_L{ld}_M{mm}",
            f"_Z1kILi{t}ELi{ld}ELi{mm}EEvPyPf",
            t,
            32,
            "cycles/pass",
        )
        for k in (0, 8, 32)
        for t, mm in pts
        for ld in (mm * k // 64,)
    ]


SPILL_RW_POINTS = (
    *((n, ld, st, 0) for n in (64, 256) for ld, st in ((1, 0), (0, 1), (1, 1), (4, 1), (1, 4))),
    *((1024, ld, st, 0) for ld, st in ((1, 0), (0, 1), (1, 1), (4, 1))),
    *((1024, ld, 1, h) for ld in (1, 4) for h in (32, 128)),
)


def spill_rw_cases(b: Batch) -> list[Case]:
    """Round 8: stack loads and stores apart (probe spill_rw)."""
    return [
        _span_case(
            b,
            f"spill_rw n{n}_L{ld}_S{st}_H{h}_g{g}",
            f"_Z1kILi256ELi{n}ELi{ld}ELi{st}ELi{h}EEvPyPjjj",
            256,
            g,
            0,
            n * 4,
            excluded="hot set by a register index: the replay walks the stack in turn"
            if h
            else None,
        )
        for n, ld, st, h in SPILL_RW_POINTS
        for g in (1, 48)
    ]


STORE_LINE_POINTS = (
    *((x, 32, 32) for x in (1, 2, 3, 4, 6, 8)),
    *((x, 16, 16) for x in (2, 4, 8)),
    (2, 16, 32),
    (4, 16, 32),
    *((x, 16, 64) for x in (1, 2, 4)),
    *((x, 32, 64) for x in (1, 2, 4)),
    (4, 4, 4),
    (8, 4, 4),
    (2, 4, 32),
    (4, 4, 32),
    (2, 4, 64),
    (1, 32, 128),
    (2, 32, 128),
    (1, 4, 128),
    (2, 4, 128),
)


def store_lines_cases(b: Batch) -> list[Case]:
    """Round 7: stores by bytes and extent (probe store_lines)."""
    return [
        _span_case(
            b, f"store_lines S{st}_W{w}_D{d}_g{g}", f"_Z1kILi{st}ELi{w}ELi{d}EEvPyS0_Pj", 128, g
        )
        for st, w, d in STORE_LINE_POINTS
        for g in (128, 256, 512)
    ]


BRANCH_LOOPS = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
BRANCH_FLATS = (64, 256, 1024, 4096)
BRANCH_PASSES = 512
FWD_MASK_ARG = (0, 0x380)  # the `z` argument of the branch probe's fwd kernel


def branch_cases(b: Batch) -> list[Case]:
    """Round 6: taken-branch cost and instruction fetch (probe branch), FFMA loops and straight
    lines on 1-32 warps, a forward branch taken and not."""
    out = []
    for n in BRANCH_LOOPS:
        # 32 warps at 512 and 1024 are measured, not replayed: millions of issues per row
        for w in (1, 4) if n >= 512 else (1, 4, 32):
            note = "measured rate above the issue limit: question 9"
            excluded = note if w == 32 and 16 <= n <= 256 else None
            cid, kernel = f"branch loop{n}_w{w}", f"_Z4loopILi{n}EEvPyPf"
            out.append(_marked(b, cid, kernel, 32 * w, BRANCH_PASSES, "cycles", excluded=excluded))
    for n in BRANCH_FLATS:
        for w in (1, 2, 4, 8, 32):
            # warp 0 reads its clock after a barrier, the others may already be running
            note = "warp 0's clock window is not the block's at several warps per pipe"
            excluded = note if w >= 8 else None
            cid, kernel = f"branch flat{n}_w{w}", f"_Z4flatILi{n}EEvPyPf"
            out.append(_marked(b, cid, kernel, 32 * w, n, "cycles", excluded=excluded))
    for name, z in (("fwd_taken", 0xFFFFFFFF), ("fwd_not", 0)):
        for w in (1, 4, 32):
            const = {FWD_MASK_ARG: z}
            cid = f"branch {name}_w{w}"
            out.append(
                _marked(b, cid, "_Z3fwdjPyPf", 32 * w, BRANCH_PASSES, "cycles", const_bank=const)
            )
    return out


MMA_PASSES = ((2, 2048), (3, 1024), (4, 1024), (5, 1024), (8, 512), (16, 256), (32, 128), (64, 64))
MMA_OTHER = {"fma": 0, "alu": 1, "lds": 2, "ldsm": 3}


def mma_issue_cases(b: Batch) -> list[Case]:
    """Round 6: mma per pass, other instructions beside it on the same warp and on warp 4 (probe
    mma_issue); cycles per mma on warp 0."""
    unit = "cycles/mma"
    return [
        *(
            _marked(b, f"mma_issue n{na}", f"_Z4passILi{na}ELi{it}EEvjPyPj", 32, na * it, unit)
            for na, it in MMA_PASSES
        ),
        *(
            _marked(
                b, f"mma_issue m{name}_{m}", f"_Z6besideILi{k}ELi{m}EEvjPyPj", 32, 8 * 256, unit
            )
            for name, k in MMA_OTHER.items()
            for m in ((1, 2) if name == "ldsm" else (1, 2, 4, 8, 14))
        ),
        *(
            _marked(b, f"mma_issue nbr_{name}", f"_Z3nbrILi{k}EEvjPyPj", 160, 16 * 256, unit)
            for name, k in (("none", "n1"), ("fma", "0"), ("alu", "1"), ("lds", "2"))
        ),
    ]


MMA_PIPE_SETS = ((1, 1), (2, 1), (4, 1), (8, 1), (1, 2), (1, 4), (1, 8), (2, 2), (4, 4), (8, 8))


def mma_pipe_cases(b: Batch) -> list[Case]:
    """Round 6: the tensor pipe's rate by compiler, operands and warps per sub-partition (probe
    mma_pipe, built by the host's and by the image's nvcc)."""
    return [
        _marked(
            b,
            f"mma_pipe {tag}_a{sa}_b{sb}_w{w}",
            f"_Z1kILi{sa}ELi{sb}EEvjPyPj",
            32 * w,
            256 * 16 * max(1, w // 4),
            "cycles/mma",
            src=unit,
        )
        for unit, tag in (("probe-mma_pipe", "v13_0"), ("probe-mma_pipe_i", "v13_2"))
        for sa, sb in MMA_PIPE_SETS
        for w in (1, 8)
    ]


def lds_narrow_cases(b: Batch) -> list[Case]:
    """Round 6: LDS and STS by element size and lane stride on 1-32 warps (probe lds_narrow);
    the arguments `stride` and `zero` enter the lanes' addresses."""
    names = (("LDS.U8", 1), ("LDS.U16", 2), ("LDS.32", 4), ("STS.U16", 2), ("STS.32", 4))
    return [
        _marked(
            b,
            f"lds_narrow {name}_s{st}_w{w}",
            f"_Z1kILi{kind}EEvjjPyPj",
            32 * w,
            256 * 8 * w,
            "cycles/instr/SM",
            const_bank={(0, 0x380): st, (0, 0x384): 0},
        )
        for kind, (name, elem) in enumerate(names)
        for st in (elem, 16, 64)
        for w in (1, 4, 32)
    ]


LAUNCH_SHORT = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 192, 256)


def launch_short_cases(b: Batch) -> list[Case]:
    """Round 6: the launch term around the kernel span where back-to-back launches leave the
    submission rate (probe launch_short); the launch probe's replay on its kernel."""
    txt = _sass(b, "probe-launch_short")
    out = []
    for i in LAUNCH_SHORT:
        ins = _k(txt, f"_Z4workILi{i}EEvPfPy")
        out.append(_launched(b, f"launch_short work_i{i}", ins, 128, 48, "kernel"))
        for mode in ("b2b", "graph"):
            out.append(_launched(b, f"launch_short {mode}_work_i{i}", ins, 128, 48, mode))
    return out


L2STAGES_TILE = {"block_M": 128, "block_N": 256, "block_K": 32, "threads": 256}


def l2stages_cases(b: Batch) -> list[Case]:
    """Round 2, P-2's second experiment (probe l2share_stages): the lean cp.async loop by stages
    and blocks, one block per SM, warm L2; the copy pattern of TileLang's 128x256x32 tile; the main
    loop runs KT - (S - 1) = 33 - S times (32 k-tiles)."""
    return [
        _us(
            b,
            f"l2share_stages s{st}_b{blocks}",
            f"_Z4loadILi{st}EEvPK6__halfS2_",
            256,
            _time(b),
            trips=33 - st,
            active_sms=blocks,
            copy_lines=copy_lines({**L2STAGES_TILE, "num_stages": st}),
        )
        for st in (1, 2, 3, 4)
        for blocks in (1, 16, 48)
    ]


L2REUSE_N = (2, 8, 16, 20, 22, 24, 27, 32, 43)  # steps of 48 x 1024 x 16 B = 0.75 MiB
L2REUSE_KINDS = ("cg", "ca", "cs")


def l2reuse_cases(b: Batch) -> list[Case]:
    """Round 3 (probe l2reuse): F = 0.75 n MiB read by 48 blocks x 1024 threads, one 16-byte load
    per thread per step; the in-kernel span. Modes: cold (after a read flush: every load a DRAM
    miss), warm (back-to-back), in4 (four passes after a flush). The engine is told the miss
    fraction of each mode by the warm rule under test (`l2_warm_miss`): cold 1, warm the rule's,
    in4 (1 + 3 x the rule's) / 4."""

    def row(n, k, kind, mode):
        nbytes = n * 48 * 1024 * 16

        def miss(m):
            w = l2_warm_miss(nbytes, kind, m)
            return {"cold": 1.0, "warm": w, "in4": (1 + 3 * w) / 4}[mode]

        return _stream(
            b,
            f"l2reuse n{n}_{kind}_{mode}",
            f"_Z2rdILi{n}ELi{k}ELi{4 if mode == 'in4' else 1}EEvPK4int4PyPi",
            1024,
            nbytes,
            miss=miss,
            read=_time(b),
        )

    return [
        row(n, k, kind, mode)
        for n in L2REUSE_N
        for k, kind in enumerate(L2REUSE_KINDS)
        for mode in ("cold", "warm", "in4")
    ]


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
    work = lambda i: _k(txt, f"_Z4workILi{i}EEvPfPy")  # noqa: E731
    return [
        *(
            _launched(b, f"launch {name}_g{g}_t{t}", ins, t, g, "event")
            for g, t in LAUNCH_GRIDS
            for name, ins in (("empty", empty), ("tiny", tiny))
        ),
        *(
            _launched(b, f"launch graph_g{g}_t{t}", empty, t, g, "graph")
            for g, t in LAUNCH_SEQ_GRIDS
        ),
        *(
            _launched(b, f"launch {mode}_work_i{i}", work(i), 128, 48, mode)
            for i in LAUNCH_SEQ_WORK
            for mode in ("b2b", "graph")
        ),
        *(_launched(b, f"launch work_i{i}", work(i), 128, 48, "event") for i in LAUNCH_WORK),
    ]


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
        if L["family"] in ("hgemm", "fp8gemm"):
            # round 4: the rotating variants are their own variant group
            rot = " rotating" if L.get("cache") == "rotating" else ""
            group = "testset " + L["set"] if "set" in L else "testset hgemm2048" + rot
            group = "testset fp8gemm2048" if L["family"] == "fp8gemm" else group
            out.append(_generic_case(b, d, L, regs, "kernel", group))
        elif L["family"] == "sequence":
            out += [_sequence_case(b, d, L, mode) for mode in ("kernel", "b2b", "graph")]
        else:
            # round 4: the small kernels also as launched in a sequence (b2b, graph), own sets
            out += [_generic_case(b, d, L, regs, mode) for mode in ("kernel", "b2b", "graph")]
    return out


def _generic_sm(d: Path, L: dict[str, Any], regs: int, carried: tuple[str, ...], **kw: Any) -> SM:
    """A test-set kernel on the one front end (kernel.generic_kernel); the main loop's trip count
    from TileLang's source loop `k`; a GEMM's k-loop may be called `ko` (MoE, HGEMM)."""
    ins = _k((d / "k.sass").read_text(), L["entry"])
    src = (d / L.get("source", "kernel.cu")).read_text()
    name = "ko?"
    loops = [
        int(x) for x in re.findall(rf"for \(int {name} = 0; {name} < (\d+); \+\+{name}\)", src)
    ]
    trips = machine_trips(ins) or (max(loops) if loops else 1)
    return generic_kernel(ins, L, regs, trips, carried, **kw)


def _generic_case(
    b: Batch, d: Path, L: dict[str, Any], regs: int, mode: str = "kernel", group: str | None = None
) -> Case:
    def build(**kw):
        return _generic_sm(d, L, regs, (), **kw)

    return _case(
        b,
        f"{d.name} {mode}",
        group or f"testset {L.get('set', L['family'])}" + ("" if mode == "kernel" else f" {mode}"),
        7,
        "us",
        build,
        _time(b, mode),
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
                    {**k, "cache": L["cache"], "rot": L["rot"]},
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


class _SeqRun:
    """A Case build result for a sequence of kernels: `spans` the cycles of each, the SM kept for
    attribution the longest."""

    def __init__(self, makers, machine=None, trace: bool = False):
        self.machine, self.trace = machine or default_machine(), trace
        self.makers = makers
        self.spans: list[float] = []
        self.sm: SM | None = None

    @property
    def m(self) -> Machine:
        return self.machine

    @property
    def cycles(self) -> float:
        assert self.sm is not None
        return self.sm.cycles

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


# unit of a batch's build directory -> its cases; a batch built with `--only` has a subset of units
BUILDERS = (
    ("probe-smem_ldsts", smem_cases),
    ("probe-shfl", shfl_cases),
    ("probe-ex2_ffma", ex2_cases),
    ("probe-namedbar", bar_cases),
    ("probe-l1hit", l1_cases),
    ("probe-store_tail", store_cases),
    ("probe-l2share_copy", l2share_cases),
    ("probe-dram_rotate", dram_cases),
    ("probe-dram_rw", dram_rw_cases),
    ("probe-dram_burst", dram_burst_cases),
    ("probe-rot_depth", rot_depth_cases),
    ("probe-spill", spill_cases),
    ("probe-store_lines", store_lines_cases),
    ("probe-warp_map", warp_map_cases),
    ("probe-copy_dram", copy_dram_cases),
    ("probe-store_spread", store_spread_cases),
    ("probe-arith_beside", arith_beside_cases),
    ("probe-first_reader", first_reader_cases),
    ("probe-first_share", first_share_cases),
    ("probe-load_curve", load_curve_cases),
    ("probe-load_sat", load_sat_cases),
    ("probe-copy_depth", copy_depth_cases),
    ("probe-mma_ldsm", mma_ldsm_cases),
    ("probe-spill_rw", spill_rw_cases),
    ("probe-l2mix", l2mix_cases),
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
)


def cases_by_builder(batch_dir: Path) -> list[tuple[str, list[Case]]]:
    b = load(batch_dir)
    return [(f.__name__, f(b)) for unit, f in BUILDERS if (b.build / unit).is_dir()]


def all_cases(batch_dir: Path) -> list[Case]:
    return [c for _, cases in cases_by_builder(batch_dir) for c in cases]


@cache
def _top_level() -> dict[str, ast.stmt]:
    out: dict[str, ast.stmt] = {}
    for n in ast.parse(Path(__file__).read_text()).body:
        if isinstance(n, ast.FunctionDef | ast.ClassDef):
            out[n.name] = n
        elif isinstance(n, ast.Assign):
            out.update((t.id, n) for t in n.targets if isinstance(t, ast.Name))
    return out


@cache
def code_key(builder: str) -> str:
    """Digest of the definitions of this file a builder's rows reach: the builder and every
    top-level name it reaches. Adding a probe here leaves the other rows' results valid."""
    defs = _top_level()
    seen: set[str] = set()
    todo = [builder, "run_cases"]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        todo += [n.id for n in ast.walk(defs[name]) if isinstance(n, ast.Name) and n.id in defs]
    text = "\n".join(ast.unparse(defs[n]) for n in sorted(seen))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def run_case(case: Case, **kw: Any) -> tuple[float, Any, dict]:
    (value,), target, r = run_cases([case], **kw)
    return value, target, r


def run_cases(cases: list[Case], **kw: Any) -> tuple[list[float], Any, dict]:
    """One replay read by several cases that share it: the kernel, b2b and graph rows of one test
    kernel differ only in the launch arithmetic of `read`."""
    case = cases[0]
    sm = case.build(**kw)
    r = sm.run(limit=case.limit) if case.limit else sm.run()
    target = sm.sm if isinstance(sm, _SeqRun) else sm
    return [c.read(sm, r) for c in cases], target, r
