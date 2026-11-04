"""The L2 for every front end (KTA PERFMODEL round 5): reuse inside a launch from what the blocks
actually load, residency across launches from the measured warm-miss curve.

A front end hands `replay` the launch as waves of blocks in flight; the blocks of a wave walk their
steps in lockstep and the replay passes each step's 128-byte lines through one cache of the
measured effective capacity (two lists, probation and protected: probe l2policy, upstream). The
result is what one launch reads from DRAM with none of its data in the L2 (rotating inputs). A warm
launch re-reads that traffic by the curve of its read footprint (`warm_miss`, probe l2reuse): the
cold reads scaled, not a second cache model.

`walk` supplies the steps of any kernel from its own SASS: a few grid blocks are replayed with lane
tracking on (two trips of the main loop), and each global load's address is extended over the trips
and over the grid by the stride seen. An axis that is not affine (a division of a block index) is
walked at every index. A load whose address the tracking does not know reuses nothing.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .machine import Machine, default_machine
from .sass import Ins, main_loop
from .sm import SM

LINE_BITS = 7
LINE = 1 << LINE_BITS


class Cache:
    """Two LRU lists over `cap` bytes: a hit promotes to protected, eviction takes the oldest
    probation entry first."""

    def __init__(self, cap: float) -> None:
        self.cap, self.used = cap, 0.0
        self.prob: OrderedDict = OrderedDict()
        self.prot: OrderedDict = OrderedDict()

    def touch(self, key: Any, size: float) -> bool:
        """True on a miss (the entry is then filled)."""
        if key in self.prot:
            self.prot.move_to_end(key)
            return False
        if key in self.prob:
            self.prot[key] = self.prob.pop(key)
            return False
        self.prob[key] = size
        self.used += size
        while self.used > self.cap:
            _, v = (self.prob if self.prob else self.prot).popitem(last=False)
            self.used -= v
        return True


def warm_miss(footprint_bytes: float, kind: str = "ca", machine: Machine | None = None) -> float:
    """Fraction of a warm re-read of `footprint_bytes` that misses the L2 (probe l2reuse): plain
    and .cg loads by `l2_warm_miss_by_MiB`, evict-first (`cs`) loads by their own curve."""
    C = (machine or default_machine()).C
    pts = C["l2_warm_miss_evict_first_by_MiB" if kind == "cs" else "l2_warm_miss_by_MiB"]
    x = footprint_bytes / 2**20
    if x >= pts[-1][0]:
        return float(pts[-1][1])
    for (x0, y0), (x1, y1) in itertools.pairwise(pts):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return 0.0


@dataclass
class Load:
    """One warp's global load in the walked block: `step` -1 before the main loop, -2 after it,
    0 inside it (then `stride` bytes per trip)."""

    wid: int
    pc: int
    occ: int
    step: int
    width: int
    base: tuple | None
    offs: tuple | None
    stride: int | None = 0
    # per grid axis: byte shift by axis index
    shift: list[dict[int, int | None]] = field(default_factory=list)


@dataclass
class Access:
    trips: int
    grid: tuple[int, int, int]
    loads: list[Load]
    unknown: int = 0


def _entries(log: list, lp: tuple[int, int] | None) -> list[Load]:
    out: list[Load] = []
    seen: dict[tuple, Load] = {}
    occ: dict[tuple, int] = {}
    for wid, loops, pc, width, addr in log:
        inside = lp is not None and lp[0] <= pc <= lp[1]
        step = 0 if inside else (-1 if lp is None or pc < lp[0] else -2)
        k = (wid, pc, loops if inside else 0)
        i = occ[k] = occ.get(k, -1) + 1
        base, offs = addr if addr is not None else (None, None)
        if inside and loops > 0:  # the second trip gives the stride of the first trip's load
            first = seen.get((wid, pc, i))
            if first is not None and first.offs is not None and offs is not None:
                d = {(o - p) & 0xFFFFFFFF for o, p in zip(offs, first.offs, strict=True)}
                first.stride = d.pop() if len(d) == 1 and base == first.base else None
            elif first is not None:
                first.stride = None
            continue
        e = Load(wid, pc, i, step, width, base, offs)
        seen[(wid, pc, i)] = e
        out.append(e)
    return out


def _shifts(a: list[Load], b: list[Load]) -> list[int | None]:
    """Per load the byte shift from block a to block b; None when the two do not line up."""
    if len(a) != len(b):
        return [None] * len(a)
    out: list[int | None] = []
    for x, y in zip(a, b, strict=True):
        if (x.wid, x.pc, x.occ) != (y.wid, y.pc, y.occ) or x.offs is None or y.offs is None:
            out.append(None)
            continue
        d = {(q - p) & 0xFFFFFFFF for p, q in zip(x.offs, y.offs, strict=True)}
        out.append(d.pop() if len(d) == 1 and x.base == y.base else None)
    return out


def walk(
    ins: list[Ins],
    threads: int,
    trips: int,
    grid: tuple,
    block_dim: tuple | None = None,
    machine: Machine | None = None,
) -> Access:
    """The kernel's global loads over the whole grid, from its own SASS (module docstring)."""
    lp = main_loop(ins)
    g = (*grid, 1, 1)[:3]

    stride_seen = [False]

    def one(b: int) -> list[Load]:
        log: list = []
        SM(
            ins,
            threads,
            1,
            min(trips, 2) if lp else 1,
            grid_dim=g,
            block_dim=block_dim,
            block_ids=[b],
            addr_log=log,
            machine=machine,
        ).run()
        stride_seen[0] |= any(e[1] > 0 and lp and lp[0] <= e[2] <= lp[1] for e in log)
        return _entries(log, lp)

    loads = one(0)
    # a loop the replay resolves from its own values runs in full inside `one`, nothing to extend
    extended = stride_seen[0]
    unit = [1, g[0], g[0] * g[1]]
    for ax, n in enumerate(g):
        table: dict[int, list[int | None]] = {0: [0] * len(loads)}
        if n > 1:
            probes = sorted({1, min(2, n - 1), n - 1})
            for v in probes:
                table[v] = _shifts(loads, one(v * unit[ax]))
            affine = all(
                e.offs is None
                or (
                    table[1][i] is not None
                    and all(
                        table[v][i] == (table[1][i] * v) & 0xFFFFFFFF  # type: ignore[operator]
                        for v in probes
                    )
                )
                for i, e in enumerate(loads)
            )
            if affine:
                table = {
                    v: [None if d is None else (d * v) & 0xFFFFFFFF for d in table[1]]
                    for v in range(n)
                }
            else:
                for v in range(1, n):
                    if v not in table:
                        table[v] = _shifts(loads, one(v * unit[ax]))
        for i, e in enumerate(loads):
            e.shift.append({v: t[i] for v, t in table.items()})
    grid3 = (g[0], g[1], g[2])
    return Access(trips if extended else 1, grid3, loads, sum(e.offs is None for e in loads))


@dataclass
class Replay:
    reads: float  # bytes one launch reads from DRAM with none of its data in the L2
    footprint: float  # distinct bytes read
    loaded: float  # bytes the blocks load (each block's lines per step)
    miss_by_pc: dict[int, float]  # per load instruction, the fraction of its lines that miss
    waves: list[float]
    base_by_pc: dict[int, Any] = field(default_factory=dict)  # per load, its address's base


class _Group:
    """The loads of one part of the block (before, inside, after the main loop) as arrays."""

    def __init__(self, loads: list[Load], bases: dict) -> None:
        self.unknown = [e for e in loads if e.offs is None]
        self.known = known = [e for e in loads if e.offs is not None]
        self.pc = np.array([e.pc for e in known], dtype=np.int64)
        self.offs = np.array([e.offs for e in known], dtype=np.int64).reshape(len(known), 32)
        self.width = np.array([e.width for e in known], dtype=np.int64).reshape(-1, 1)
        ids = [bases.setdefault(e.base, len(bases)) for e in known]
        self.tag = (np.array(ids, dtype=np.int64) << 36).reshape(-1, 1)
        self.stride = np.array([e.stride or 0 for e in known], dtype=np.int64)
        self.flat = np.array([e.stride is None for e in known], dtype=bool)
        self.shift: list[dict[int, tuple]] = [{}, {}, {}]
        for ax in range(3):
            for v in known[0].shift[ax] if known else ():
                d = [e.shift[ax].get(v) for e in known]
                self.shift[ax][v] = (
                    np.array([x or 0 for x in d], dtype=np.int64),
                    np.array([x is None for x in d], dtype=bool),
                )

    def keys(self, block: tuple[int, int, int], trip: int) -> tuple[Any, Any]:
        """(line keys per load as one array, the load index of each key); loads whose address is
        not known at this block or trip are left out of the keys and returned in the mask."""
        bz, by, bx = block
        d = trip * self.stride
        bad = (self.flat if trip else np.zeros(len(self.known), dtype=bool)).copy()
        for ax, v in ((0, bx), (1, by), (2, bz)):
            s, b = self.shift[ax][v]
            d = d + s
            bad |= b
        lo = (self.offs + d.reshape(-1, 1)) & 0xFFFFFFFF
        hi = lo + self.width - 1
        k = np.concatenate((lo >> LINE_BITS, hi >> LINE_BITS), axis=1) + self.tag
        return k, bad


def replay_walk(acc: Access, in_flight: int, cap: float) -> Replay:
    """`acc`'s blocks in launch order, `in_flight` at a time, through one `Cache`."""
    gx, gy, gz = acc.grid
    cache = Cache(cap)
    blocks = list(itertools.product(range(gz), range(gy), range(gx)))
    bases: dict = {}
    groups = [_Group([e for e in acc.loads if e.step == s], bases) for s in (-1, 0, -2)]
    seen: set = set()
    reads = loaded = 0.0
    r_pc: dict[int, float] = {}
    l_pc: dict[int, float] = {}
    waves: list[float] = []

    def touch(g: _Group, wave: list[tuple], trip: int) -> None:
        nonlocal reads, loaded
        for e in g.unknown:  # unknown address: 32 lanes' lines, reused by nothing
            b = max(LINE, 32 * e.width) * len(wave)
            reads += b
            loaded += b
            r_pc[e.pc] = r_pc.get(e.pc, 0.0) + b
            l_pc[e.pc] = l_pc.get(e.pc, 0.0) + b
        if not g.known:
            return
        lists, owners = [], []
        for block in wave:
            k, bad = g.keys(block, trip)
            idx = np.repeat(np.arange(len(g.known)), k.shape[1])
            flat = k.ravel()
            if bad.any():
                bb = np.where(bad)[0]
                n = len(bb) * LINE * len(wave)
                keep = ~bad[idx]
                flat, idx = flat[keep], idx[keep]
                reads += n / len(wave)
                loaded += n / len(wave)
            u, first = np.unique(flat, return_index=True)  # the block's distinct lines
            own = g.pc[idx[first]]
            pcs, n = np.unique(own, return_counts=True)
            for pc, c in zip(pcs.tolist(), n.tolist(), strict=True):
                l_pc[pc] = l_pc.get(pc, 0.0) + c * LINE
            loaded += len(u) * LINE
            lists.append(u)
            owners.append(own)
        u, first = np.unique(np.concatenate(lists), return_index=True)
        own = np.concatenate(owners)[first]
        for key, pc in zip(u.tolist(), own.tolist(), strict=True):
            seen.add(key)
            if cache.touch(key, LINE):
                reads += LINE
                r_pc[pc] = r_pc.get(pc, 0.0) + LINE

    for w in range(0, len(blocks), in_flight):
        before = reads
        wave = blocks[w : w + in_flight]
        touch(groups[0], wave, 0)
        for t in range(acc.trips):
            touch(groups[1], wave, t)
        touch(groups[2], wave, 0)
        waves.append(reads - before)
    miss = {pc: min(1.0, r_pc.get(pc, 0.0) / v) for pc, v in l_pc.items() if v}
    base = {e.pc: e.base for e in acc.loads}
    return Replay(reads, len(seen) * LINE, loaded, miss, waves, base)


def rotating_miss(window_bytes: float, m: Machine) -> float:
    """Part of a rotating launch's reads that miss, by the bytes the rotation touches. Round 6: all
    of them; the probe rot_depth replays this rule before it is measured."""
    return warm_miss(window_bytes, "ca", m) if window_bytes <= 24 * 2**20 else 1.0
