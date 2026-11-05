"""The coverage matrix (LOOP §3a, D874): operation × target × pattern × sharing, each cell measured
with a rule (M), measured without a rule (m) or empty (.), with its probe; what each operation and
target decides; each blind miss placed in a cell. `python -m kta_perfmodel.matrix` rewrites the
section of docs/design/29-mechanism-inventory.md between its markers and prints the counts.

The state is read from the probes and the inventory, by hand: a sweep that lands changes `CELLS`
(appended in `ROUND16`), and the counts before and after come from dropping or keeping that list.
"""

from __future__ import annotations

import argparse
import itertools
import re
from collections import Counter
from typing import NamedTuple

from . import upstream

DOC = upstream.ROOT / "docs" / "design" / "29-mechanism-inventory.md"
OPS = ("read", "write", "rmw", "atomic", "copy", "tload", "compute", "barrier")
TARGETS = ("shared", "stack", "L1", "L2", "DRAM")
PATTERNS = ("order", "stride", "row", "hashed", "cap")
SHARING = ("w", "b", "s", "c")
OP_NAME = {
    "read": "read (LDG, LDS, LDL)",
    "write": "write (STG, STS, STL)",
    "rmw": "read-modify-write of the same words",
    "atomic": "atomic (ATOM, RED, ATOMS)",
    "copy": "copy (cp.async)",
    "tload": "tensor load (ldmatrix from shared, TMA from global)",
    "compute": "compute beside traffic to the target",
    "barrier": "barrier (BAR, mbarrier, fence)",
}


# LOOP §3a, the axes checked (round 19, D904): modules of 29 with no cell, each with its state.
# M: measured with a rule; P: partial; A: assumed; X: not modelled.
BESIDE: tuple[tuple[str, str, str], ...] = (
    ("Scheduler, stall counts, dependency counters", "M", "issue slots, not traffic"),
    ("Instruction fetch", "M", "`branch`; 9 clk per 8 past the buffer"),
    ("Divergence and reconvergence", "P", "D811; lanes and guards per lane D903"),
    ("Value-dependent control", "P", "loops from loaded data: `hand_mandel` 3.27×, `stl_l_n1024`"),
    (
        "Register file, operand collector",
        "P",
        "not measurable from user space; mma operands frozen",
    ),
    ("Uniform datapath, default latencies", "A", "`default_var_latency_clk`, `rbar_release_clk`"),
    (
        "Compute pipes: tensor, FMA, ALU, XU",
        "M",
        "one `compute` row beside traffic; pipes alone here",
    ),
    ("Pipes mixed (LOP3 with FFMA, FP64, popcount)", "X", "issue slot only"),
    ("Shared-memory atomics, CAS loops", "X", "no kernel on the scorecard"),
    ("Constant loads `LDC`", "X", "default latency"),
    ("Memory fences, flag polling", "X", "`MEMBAR` default; polling has no primitive"),
    ("TMA box and engine", "A", "rates fitted on the TMA ablation; `tload` cells hold the path"),
    ("mbarrier hand-off", "M", "134 clk; arrive after a store X"),
    ("L2 slices, set hash, NoC", "X", "not seen in any sweep so far"),
    ("DRAM channels, banks", "X", "D879: not reached"),
    ("TLB", "M", "D879: no cost to 1 GiB"),
    ("CPU contention for the DRAM", "F", "an input, 0"),
    ("Residency, dispatch, block replacement", "M", "one SM stands for the rest (A)"),
    ("Launch, kernel sequences", "M", "D769, D779, D883"),
    ("Clock under load, power, thermal", "P", "calibrated per batch; states X"),
)


def dropped(op: str, t: str, p: str, s: str) -> str | None:
    """Why a cell cannot occur, else None."""
    if op in ("copy", "tload", "atomic") and t == "stack":
        return "no such path"
    if op in ("copy", "atomic") and t in ("L1", "shared") and not (op, t) == ("atomic", "shared"):
        return "bypasses the L1, or the target is the copy's source"
    if op == "tload" and t == "L1":
        return "bypasses the L1"
    if op in ("write", "rmw") and t == "L1":
        return "the L1 writes through: the target is the L2"
    if t == "shared" and s == "c":
        return "shared memory is per SM: the same cell as `s`"
    if t == "shared" and p == "cap":
        return "shared memory has no eviction"
    if op == "compute" and p != "order":
        return "compute has no pattern"
    if op == "barrier" and (p != "order" or t not in ("shared", "L2")):
        return "a barrier has no pattern; target only shared (BAR) or L2 (mbarrier, fence)"
    return None


class Mark(NamedTuple):
    op: str
    target: str
    patterns: str
    sharing: str
    state: str
    probe: str


# later marks override earlier ones; a space-separated field names several values
CELLS: tuple[Mark, ...] = (
    Mark("read", "shared", "order stride", "w b", "M", "smem_ldsts, lds_narrow (D784)"),
    Mark("read", "shared", "row", "b", "M", "lds_narrow (D784)"),
    Mark("read", "shared", "order", "s", "M", "upstream smem_bw, shfl"),
    Mark("read", "stack", "order", "b s c", "M", "spill_rw (D800)"),
    Mark("read", "stack", "hashed", "c", "M", "stl_* (D862)"),
    Mark("read", "stack", "cap", "c", "m", "stl_u_* (question 44)"),
    Mark("read", "L1", "order", "w b", "M", "l1hit, first_reader (D809)"),
    Mark("read", "L1", "row", "b c", "M", "row_reduce (D777)"),
    Mark("read", "L2", "order", "w", "M", "l1hit (342 clk miss)"),
    Mark("read", "L2", "order", "b", "M", "first_reader"),
    Mark("read", "L2", "order stride", "c", "M", "l2_cap (D835)"),
    Mark("read", "L2", "row", "c", "m", "row_reduce, dl_*_warm; threads D870"),
    Mark("read", "L2", "cap", "c", "M", "l2reuse (D766), rot_depth"),
    Mark("read", "DRAM", "order", "w", "M", "dram_spread (latency)"),
    Mark("read", "DRAM", "order", "c", "m", "dram_rotate, dram_burst, dl_* (D870): half the rate"),
    Mark("read", "DRAM", "row", "c", "m", "gd_*, rd_* (D870)"),
    Mark("read", "DRAM", "hashed", "w", "M", "dram_spread (chase)"),
    Mark("write", "shared", "order stride", "w", "M", "smem_ldsts, lds_narrow"),
    Mark("write", "shared", "order stride", "b", "m", "smem_ldsts: 2-6 warps unidentified"),
    Mark("write", "stack", "order", "b s c", "M", "spill_rw (D800)"),
    Mark("write", "stack", "hashed", "c", "M", "stk_* (D852)"),
    Mark("write", "stack", "cap", "c", "M", "spill_rw (133 B/clk past 192 KB)"),
    Mark("write", "L2", "order", "b s", "M", "store_tail (partner rule)"),
    Mark("write", "L2", "order stride row", "c", "M", "store_lines, store_spread, st256 (D838)"),
    Mark("write", "L2", "hashed", "c", "m", "hs_*, hb_* (D851): past 4 MiB not separated"),
    Mark("write", "L2", "cap", "c", "m", "wr_*, wl_* (D802, D873): 6.5 to 7.6 MiB no step"),
    Mark("write", "DRAM", "order", "c", "m", "dram_rw 0:2 −85 %: no pacing without reads"),
    Mark("write", "DRAM", "hashed", "c", "m", "hs_* (D851)"),
    Mark("rmw", "stack", "order", "c", "M", "spill (load then store)"),
    Mark("rmw", "stack", "cap", "c", "m", "stu_* (question 44)"),
    Mark("rmw", "L2", "order", "c", "M", "rw_*, wr_rmw (D833)"),
    Mark("rmw", "L2", "row", "c", "m", "rmw_*, question 42"),
    Mark("rmw", "L2", "hashed", "c", "m", "bl_mixrwa"),
    Mark("rmw", "DRAM", "order", "c", "m", "rmw_*"),
    Mark("atomic", "L2", "order hashed row", "c", "M", "atom_*, atb_*, atr_* (D834, D850)"),
    Mark("copy", "L2", "order", "w b s", "M", "copy_two, copy_depth (D869)"),
    Mark("copy", "L2", "order stride", "c", "M", "l2_cap, loopsteps (D835)"),
    Mark("copy", "L2", "row", "c", "m", "fad_* (D871): loss by D mod 32"),
    Mark("copy", "L2", "cap", "c", "M", "first_share (D807)"),
    Mark("copy", "DRAM", "order", "b", "M", "first_reader (D810)"),
    Mark("copy", "DRAM", "order", "c", "m", "copy_dram, first_share; rate D869 rows +18 %"),
    Mark("tload", "shared", "order stride", "w b s", "M", "mma_ldsm, upstream instlat"),
    Mark("tload", "L2", "order", "b", "M", "upstream dramlat (440 clk)"),
    Mark("tload", "L2", "order", "c", "m", "fitted rates (TMA ablation)"),
    Mark("tload", "DRAM", "order", "b", "M", "upstream dramlat (1170 clk)"),
    Mark("compute", "shared", "order", "w b", "M", "mma_issue, mma_ldsm"),
    Mark("compute", "shared", "order", "s", "M", "mma_pair, tr_* (D849)"),
    Mark("compute", "L1", "order", "c", "M", "arith_beside (D809)"),
    Mark("compute", "L2", "order", "c", "m", "wq_*: questions 32, 36"),
    Mark("barrier", "shared", "order", "w b", "M", "namedbar"),
    Mark("barrier", "shared", "order", "s", "M", "mma_pair"),
)
ROUND16: tuple[Mark, ...] = (
    Mark("read", "DRAM", "order stride row hashed", "c", "M", "mr_* (D877): 64-byte bursts"),
    Mark("read", "DRAM", "order", "b s", "M", "mr_g* (D877)"),
    Mark("read", "DRAM", "cap", "c", "M", "mh_*, md_* (D879): no TLB cost to 1 GiB"),
    Mark("read", "L2", "cap", "c", "M", "mk_*, md_*, mr_s* (D878): 128-byte lines"),
    Mark("read", "L2", "hashed", "c", "M", "mh_* (D878, D877)"),
    Mark("read", "L1", "cap", "b s", "M", "ml_*, ml2_* (D880): LRU, 128 KiB per SM"),
    Mark("read", "L1", "hashed", "b", "m", "ml_h_*: +140 %, not pursued"),
    Mark("read", "L1", "order", "s", "M", "ml2_*"),
    Mark("read", "shared", "row", "s", "m", "ms_s_*: D 64, 96, 128 slower, short rows"),
    Mark("write", "shared", "row", "s", "m", "ms_s_*: D 64, 96, 128 slower, short rows"),
    Mark("write", "DRAM", "order", "c", "M", "mw_o_* (D881): no limit past the store's own"),
    Mark("write", "DRAM", "hashed", "c", "M", "mw_h_f64 (D881): a burst per lane"),
    Mark("write", "L2", "hashed", "c", "m", "mw_h_f16 −73 %: inside the L2, not pursued"),
    Mark("write", "L2", "row", "c", "m", "mt_w_*: lanes in part of a sector −22 to +44 %"),
    Mark("rmw", "L2", "order", "c", "m", "mx_*: +73 to +104 % at first score, no rule"),
    Mark("copy", "DRAM", "order", "s c", "M", "copy_cold (D882): 10 clk per copy, cold as warm"),
    Mark("copy", "DRAM", "row", "c", "m", "fa rotating under bursts, no sweep of its own"),
    Mark("copy", "L2", "row", "b", "m", "ms_c_*: short rows, flat by D"),
)

ROUND17: tuple[Mark, ...] = (
    Mark("write", "L2", "row", "c", "M", "pw_*, mt_w_* (D887): 0.063 clk a sector, 0.105 a line"),
    Mark("rmw", "L2", "order", "c", "M", "px_*, mx_* (D887): the same rate; `mx_*` +30 % left"),
    Mark("read", "DRAM", "order", "c", "M", "tw_*, gr_* (D886): no window; edges of a run 0.6"),
)

DECIDES = {
    ("read", "shared"): "layout, split of loads and arithmetic",
    ("read", "stack"): "tile size at the register limit",
    ("read", "L1"): "threads, row layout (GEMV, decode, rows)",
    ("read", "L2"): "threads, split, tile",
    ("read", "DRAM"): "threads, split, rotation; every rotating group",
    ("write", "shared"): "layout",
    ("write", "stack"): "tile size at the register limit",
    ("write", "L2"): "store path, width, split, batch size at the dirty limit",
    ("write", "DRAM"): "store path past the dirty limit",
    ("rmw", "stack"): "tile size at the register limit",
    ("rmw", "L2"): "in-place or out-of-place rows",
    ("rmw", "DRAM"): "in-place or out-of-place rows",
    ("atomic", "shared"): "reduction form",
    ("atomic", "L2"): "reduction form, bins",
    ("atomic", "DRAM"): "reduction form past the L2",
    ("copy", "L2"): "stages, tile, threads",
    ("copy", "DRAM"): "stages, tile, swizzle",
    ("tload", "shared"): "tile, layout, ways",
    ("tload", "L2"): "copy path: TMA or cp.async",
    ("tload", "DRAM"): "copy path: TMA or cp.async",
    ("compute", "shared"): "warp specialisation",
    ("compute", "stack"): "tile size",
    ("compute", "L1"): "split of arithmetic",
    ("compute", "L2"): "stages, threads",
    ("compute", "DRAM"): "stages",
    ("barrier", "shared"): "threads, stages",
    ("barrier", "L2"): "store path, sequence",
}

# Round 15's 51 blind rows off by more than 10 %: row -> (op, target, pattern, sharing, class)
# class: d defect, k known rule wrong on the row, n no rule covers
MISSES: dict[str, tuple[str, str, str, str, str]] = {
    "bl15_chunked_w6": ("write", "L2", "cap", "c", "k"),
    "bl15_clampread": ("read", "L2", "stride", "c", "k"),
    "bl15_divrun": ("write", "L2", "hashed", "c", "d"),
    "bl15_divseven": ("read", "L2", "order", "c", "k"),
    "bl15_divseven_t128": ("read", "L2", "order", "c", "k"),
    "bl15_floatindex": ("read", "L1", "stride", "c", "d"),
    "bl15_modring": ("write", "L2", "stride", "c", "k"),
    "bl15_morton": ("read", "L2", "hashed", "c", "k"),
    "bl15_popindex": ("read", "L1", "hashed", "c", "k"),
    "bl15_reverse": ("write", "L2", "order", "c", "d"),
    "bl15_reverse_m4": ("write", "L2", "order", "c", "k"),
    "bl15_triangle": ("read", "DRAM", "row", "c", "k"),
    "bl15_twobuf": ("read", "L2", "stride", "c", "d"),
    "bl15_wideindex": ("read", "L2", "hashed", "c", "k"),
    "bl15_wrapstencil": ("read", "L2", "row", "c", "k"),
    **{
        f"dec15_{v}_warm": ("read", "L2", "row", "c", "k") for v in ("b", "s1", "split4", "split16")
    },
    **{
        f"dec15_{v}_rotating": ("read", "DRAM", "row", "c", "n")
        for v in ("b", "s1", "split4", "split16")
    },
    **{
        f"fa15_{v}_warm": ("copy", "L2", "row", "c", "n")
        for v in ("c32x64_s2_t64", "c64x64_s1_t128", "c64x64_s2_t128", "c64x64_s2_t64")
    },
    **{
        f"fa15_{v}_rotating": ("copy", "DRAM", "row", "c", "n")
        for v in ("c32x64_s2_t64", "c64x64_s1_t128", "c64x64_s2_t128", "c64x64_s2_t64")
    },
    **{f"gemv15_{v}_warm": ("read", "L2", "row", "c", "k") for v in ("b", "np2_r64", "np2", "r64")},
    **{
        f"gemv15_{v}_rotating": ("read", "DRAM", "order", "c", "n")
        for v in ("mk16", "np2_r64", "np2")
    },
    "hgemm_c128x128x32_s3_t128_sw8": ("copy", "L2", "order", "s", "k"),
    "hgemm_c128x256x32_s3_t256": ("copy", "L2", "order", "s", "d"),
    "hgemm_c128x256x32_s3_t256_rotating": ("copy", "DRAM", "order", "s", "d"),
    **{f"rms15_t{t}_warm": ("read", "L2", "row", "c", "n") for t in (16, 32, 64, 128, 256)},
    **{f"sm15_t{t}_warm": ("write", "L2", "cap", "c", "k") for t in (16, 32, 64, 128, 256)},
}


# Round 16's blind rows above 10 %, by family: pattern of id -> (cell, class). Read after the sweep.
MISSES16: dict[str, tuple[str, str, str, str, str, int]] = {
    "gatherfour": ("read", "L2", "hashed", "c", "k", 2),
    "ninepoint": ("read", "L2", "row", "c", "k", 1),
    "packrows": ("read", "L2", "row", "c", "k", 1),
    "revrows": ("read", "L2", "row", "c", "k", 1),
    "recordthree": ("write", "L2", "stride", "c", "k", 1),
    "stackhist": ("rmw", "stack", "hashed", "c", "n", 1),
    "stackwin": ("read", "stack", "hashed", "c", "k", 1),
    "stridewalk": ("read", "L2", "stride", "c", "k", 2),
    "twolevel": ("read", "L2", "hashed", "c", "k", 1),
    "xorput": ("write", "L2", "hashed", "c", "k", 2),
    "decode warm": ("read", "L2", "row", "c", "k", 6),
    "decode rotating": ("read", "DRAM", "row", "c", "n", 7),
    "fa warm": ("copy", "L2", "row", "c", "k", 2),
    "fa rotating": ("copy", "DRAM", "row", "c", "n", 2),
    "gemv warm": ("read", "L2", "row", "c", "k", 7),
    "gemv rotating": ("read", "DRAM", "order", "c", "n", 5),
    "hgemm warm": ("copy", "L2", "order", "s", "k", 1),
    "hgemm rotating": ("copy", "DRAM", "order", "c", "k", 1),
    "rmsnorm warm": ("read", "L2", "row", "c", "n", 6),
    "softmax warm": ("write", "L2", "row", "c", "k", 6),
}


# Round 17's blind rows above 10 %, placed the same way
MISSES17: dict[str, tuple[str, str, str, str, str, int]] = {
    "addput": ("write", "L2", "hashed", "c", "k", 2),
    "atomrange": ("atomic", "L2", "row", "c", "k", 1),
    "blockmax": ("atomic", "L2", "hashed", "c", "k", 1),
    "primefold": ("write", "L2", "stride", "c", "k", 2),
    "recordfive": ("write", "L2", "stride", "c", "k", 1),
    "stackmed": ("rmw", "stack", "hashed", "c", "n", 1),
    "stackrun": ("rmw", "stack", "cap", "c", "k", 1),
    "threelevel": ("read", "L2", "hashed", "c", "k", 1),
    "unpackrows": ("write", "L2", "row", "c", "k", 1),
    "decode warm": ("read", "L2", "row", "c", "k", 5),
    "decode rotating": ("read", "DRAM", "row", "c", "n", 4),
    "fa warm": ("copy", "L2", "row", "c", "k", 4),
    "fa rotating": ("copy", "DRAM", "row", "c", "n", 4),
    "gemv warm": ("read", "L2", "row", "c", "k", 6),
    "gemv rotating": ("read", "DRAM", "order", "c", "n", 5),
    "hgemm warm": ("copy", "L2", "order", "s", "k", 1),
    "hgemm rotating": ("copy", "DRAM", "order", "c", "k", 2),
    "rmsnorm warm": ("read", "L2", "row", "c", "n", 6),
    "softmax warm": ("write", "L2", "row", "c", "k", 6),
}


ROUND18: tuple[Mark, ...] = (
    Mark("write", "stack", "order", "c", "M", "sp_* (D897): frames count at the dirty limit"),
    Mark("copy", "DRAM", "row", "c", "m", "dr_* (D899): rows off 64 bytes slow BN 64, no rule"),
    Mark("read", "DRAM", "row", "c", "M", "gq_* (D898): warps per row +5 % a doubling, no rule"),
)
ROUND19: tuple[Mark, ...] = (
    Mark("write", "L2", "cap", "c", "M", "rw_*, sw_*, wb_* (D911): a miss curve with stores"),
    Mark("copy", "L2", "row", "c", "M", "cb_*, fas_*, dw_* (D910): runs of lanes in a line"),
    Mark("copy", "DRAM", "row", "c", "m", "dec20 BN 64 rotating after D910 (D912): no rule"),
)
ALL_MARKS = CELLS + ROUND16 + ROUND17 + ROUND18 + ROUND19
# Placed after replay's check (`review coverage`): decode's time is its copies and its compute
MISSES19: dict[str, tuple[str, str, str, str, str, int]] = {
    "gemv warm": ("read", "L2", "row", "c", "k", 8),
    "decode rotating": ("copy", "DRAM", "row", "c", "n", 6),
    "decode warm": ("compute", "shared", "order", "s", "k", 6),
    "rmsnorm warm": ("write", "L2", "cap", "c", "n", 5),
    "softmax warm": ("write", "L2", "cap", "c", "n", 5),
    "fa rotating": ("copy", "DRAM", "row", "c", "n", 3),
    "fa warm": ("copy", "L2", "row", "c", "k", 3),
    "rmsnorm rotating": ("read", "DRAM", "row", "c", "k", 3),
    "byteload": ("read", "L2", "order", "c", "k", 2),
    "atomtwo": ("atomic", "L2", "hashed", "c", "k", 1),
    "halfstore": ("write", "L2", "order", "c", "k", 1),
    "loadmax": ("read", "L2", "stride", "c", "k", 1),
    "maskload": ("read", "L2", "order", "c", "k", 1),
    "pairsum": ("read", "L2", "order", "c", "k", 1),
    "stridestore": ("write", "L2", "stride", "c", "k", 1),
    "triplet": ("read", "L2", "order", "c", "k", 1),
    "twodim": ("read", "L2", "order", "c", "k", 1),
    "gemv rotating": ("read", "DRAM", "row", "c", "k", 1),
    "hgemm rotating": ("copy", "DRAM", "order", "c", "k", 1),
}


# Round 18's blind rows above 10 %, placed the same way; class d: a defect found by this round
MISSES18: dict[str, tuple[str, str, str, str, str, int]] = {
    "decode rotating": ("read", "DRAM", "row", "c", "n", 7),
    "gemv rotating": ("read", "DRAM", "order", "c", "d", 7),
    "decode warm": ("read", "L2", "row", "c", "k", 6),
    "rmsnorm warm": ("read", "L2", "row", "c", "n", 6),
    "softmax warm": ("write", "L2", "row", "c", "k", 6),
    "gemv warm": ("read", "L2", "row", "c", "k", 5),
    "fa rotating": ("copy", "DRAM", "row", "c", "n", 3),
    "fa warm": ("copy", "L2", "row", "c", "k", 3),
    "stackfold": ("rmw", "stack", "cap", "c", "k", 2),
    "rmsnorm rotating": ("read", "DRAM", "row", "c", "d", 2),
    "histbytes": ("atomic", "L2", "hashed", "c", "k", 1),
    "prefixstack": ("rmw", "stack", "hashed", "c", "k", 1),
}


def cells(marks: tuple[Mark, ...]) -> dict[tuple[str, str, str, str], Mark | None]:
    out: dict[tuple[str, str, str, str], Mark | None] = {}
    for key in itertools.product(OPS, TARGETS, PATTERNS, SHARING):
        if not dropped(*key):
            out[key] = None
    for m in marks:
        for p in m.patterns.split():
            for s in m.sharing.split():
                key = (m.op, m.target, p, s)
                if key not in out:
                    raise SystemExit(f"mark on a dropped cell: {key}")
                out[key] = m
    return out


def count(table: dict) -> Counter:
    return Counter(m.state if m else "." for m in table.values())


def misses(table: dict) -> Counter:
    c: Counter = Counter()
    for key in MISSES.values():
        m = table[key[:4]]
        c[m.state if m else "."] += 1
    return c


def render(marks: tuple[Mark, ...]) -> str:
    table = cells(marks)
    before = cells(CELLS)
    placed: Counter = Counter(v[:4] for v in MISSES.values())
    n = count(table)
    b = count(before)
    r16 = count(cells(CELLS + ROUND16))
    lines = [
        f"Cells that can occur: {len(table)} of "
        f"{len(OPS) * len(TARGETS) * len(PATTERNS) * len(SHARING)}, the rest dropped "
        "(`tools/kta_perfmodel/matrix.py`, `dropped`, says why for each).",
        "",
        "| State | Before round 16 | Before round 17 | Now |",
        "| --- | ---: | ---: | ---: |",
        f"| M measured with a rule | {b['M']} | {r16['M']} | {n['M']} |",
        f"| m measured without a rule | {b['m']} | {r16['m']} | {n['m']} |",
        f"| · empty | {b['.']} | {r16['.']} | {n['.']} |",
        "",
    ]
    bm, nm = misses(before), misses(table)
    c17: Counter = Counter()
    k17: Counter = Counter()
    for v in MISSES17.values():
        m = table[v[:4]]
        c17[m.state if m else "."] += v[5]
        k17[v[4]] += v[5]
    c18: Counter = Counter()
    k18: Counter = Counter()
    for v in MISSES18.values():
        m = table[v[:4]]
        c18[m.state if m else "."] += v[5]
        k18[v[4]] += v[5]
    c19: Counter = Counter()
    k19: Counter = Counter()
    for v in MISSES19.values():
        m = table[v[:4]]
        c19[m.state if m else "."] += v[5]
        k19[v[4]] += v[5]
    lines += [
        f"Round 19's {sum(c19.values())} blind rows above 10 %, placed after the replay's check: "
        f"in M cells {c19['M']}, in m cells {c19['m']}, in empty cells {c19['.']}. By class at "
        f"first score: defect {k19['d']}, known rule {k19['k']}, no rule {k19['n']}.",
        "",
    ]
    lines += [
        f"Round 18's {sum(c18.values())} blind rows above 10 %, placed after the rules: in M cells "
        f"{c18['M']}, in m cells {c18['m']}, in empty cells {c18['.']}. By class at first score: "
        f"defect {k18['d']}, known rule {k18['k']}, no rule {k18['n']}.",
        "",
        f"Round 17's {sum(c17.values())} blind rows above 10 %, placed after the rules: in M cells "
        f"{c17['M']}, in m cells {c17['m']}, in empty cells {c17['.']}. By class at first score: "
        f"defect {k17['d']}, known rule {k17['k']}, no rule {k17['n']}.",
        "",
    ]
    c16: Counter = Counter()
    for v in MISSES16.values():
        m = table[v[:4]]
        c16[m.state if m else "."] += v[5]
    k16 = Counter()
    for v in MISSES16.values():
        k16[v[4]] += v[5]
    lines += [
        f"Round 16's {sum(c16.values())} blind rows above 10 %, placed after the sweep: in M cells "
        f"{c16['M']}, in m cells {c16['m']}, in empty cells {c16['.']}. By class at first score: "
        f"defect {k16['d']}, known rule {k16['k']}, no rule {k16['n']}.",
        "",
    ]
    lines += [
        f"Round 15's {len(MISSES)} blind rows above 10 %: in M cells {bm['M']}, in m cells "
        f"{bm['m']}, in empty cells {bm['.']} before round 16; {nm['M']}, {nm['m']}, {nm['.']} "
        "now. By class: defect "
        f"{sum(v[4] == 'd' for v in MISSES.values())}, known rule "
        f"{sum(v[4] == 'k' for v in MISSES.values())}, no rule "
        f"{sum(v[4] == 'n' for v in MISSES.values())}.",
        "",
        "Patterns: order (contiguous lanes walking on), stride (lanes or warps a constant apart), "
        "row (rows a pitch apart), hashed (no order in the lanes), cap (footprint across the "
        "target's capacity; for DRAM, across the TLB's reach). Sharing: w one warp, b one block, "
        "s blocks on an SM, c every SM. A cell reads `state` and, in brackets, the blind misses "
        "of round 15 in it.",
        "",
    ]
    for op, t in itertools.product(OPS, TARGETS):
        rows = [p for p in PATTERNS if any((op, t, p, s) in table for s in SHARING)]
        if not rows:
            continue
        lines += [f"**{OP_NAME[op]} × {t}**: decides {DECIDES.get((op, t), '—')}.", ""]
        lines += ["| Pattern | w | b | s | c | Probes |", "| --- | --- | --- | --- | --- | --- |"]
        for p in rows:
            cols, probes = [], []
            for s in SHARING:
                key = (op, t, p, s)
                if key not in table:
                    cols.append("")
                    continue
                m = table[key]
                st = m.state if m else "·"
                if placed.get(key):
                    st += f" ({placed[key]})"
                cols.append(st)
                if m and m.probe not in probes:
                    probes.append(m.probe)
            lines.append(f"| {p} | {' | '.join(cols)} | {'; '.join(probes)} |")
        lines.append("")
    lines += ["Modules of 29 with no cell (LOOP §3a, the axes checked):", ""]
    lines += ["| Module | State | Note |", "| --- | --- | --- |"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in BESIDE]
    lines.append("")
    return "\n".join(lines)


BEGIN, END = "<!-- matrix: generated -->", "<!-- matrix end -->"


def main() -> None:
    assert sum(v[5] for v in MISSES16.values()) == 56
    assert sum(v[5] for v in MISSES17.values()) == 54
    assert sum(v[5] for v in MISSES18.values()) == 49
    assert sum(v[5] for v in MISSES19.values()) == 51
    ap = argparse.ArgumentParser(prog="python -m kta_perfmodel.matrix")
    ap.parse_args()
    marks = ALL_MARKS
    body = render(marks)
    text = DOC.read_text()
    section = f"{BEGIN}\n{body}\n{END}"
    if BEGIN in text:
        text = re.sub(f"{BEGIN}.*{END}", lambda _: section, text, flags=re.S)
    else:
        head = "## 8. Counts"
        text = text.replace(head, f"## 7a. Coverage matrix (LOOP §3a)\n\n{section}\n\n{head}")
    DOC.write_text(text)
    t = cells(marks)
    print(len(t), dict(count(t)), "misses", dict(misses(t)), "before", dict(misses(cells(CELLS))))
    by: Counter = Counter()
    for key, m in t.items():
        if not m:
            by[key[:2]] += 1
    print(by.most_common(12))


if __name__ == "__main__":
    main()
