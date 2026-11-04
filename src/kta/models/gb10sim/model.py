"""The GB10's parameters (gb10.json) as plain values and the traffic a tile grid sends to the L2 and
DRAM: the launch order, the k-slice LRU over the L2, the bank conflicts of a shared-memory layout.
What is left of upstream's analytic model (agentic/gb10/model.py); the whole-kernel simulation is
kernel.py, whose DRAM traffic this feeds."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .l2 import Cache

PARAMS = Path(__file__).with_name("gb10.json")  # KTA: the copied calibration/spark-6951/gb10.json
C = {k: v["value"] for k, v in json.loads(PARAMS.read_text()).items() if not k.startswith("_")}
SMS = C["sms"]


def launch_order(tiles_m: int, tiles_n: int, panel: int):
    """(row, col) of each block in launch order: TileLang's rasterization2DRow (serpentine over
    panels of `panel` tile
    rows, src/tl_templates/cuda/threadblock_swizzle.h) or, without use_swizzle, row-major with
    columns fastest."""
    n = tiles_m * tiles_n
    if not panel:
        for i in range(n):
            yield i // tiles_n, i % tiles_n
        return
    psize = panel * tiles_n
    total = -(-n // psize)
    for i in range(n):
        off, pidx = i % psize, i // psize
        stride = panel if pidx + 1 < total else (n - pidx * psize) // tiles_n
        col = tiles_n - 1 - off // stride if pidx & 1 else off // stride
        yield off % stride + pidx * panel, col


_kslice_cache = {}


def dram_split_kslice(h: dict, shape: dict, resident_blocks: int, cap: float) -> dict:
    """DRAM traffic at k-slice granularity: the blocks of a wave (launch order) walk k in lockstep;
    at each k-step each
    block reads its A slice (block_M x block_K) and B slice (block_K x block_N). Slices pass through
    an LRU of the L2's
    measured effective capacity; a slice already there is a hit (neighbours in a wave share A rows
    and B columns). Between
    waves only what is still resident carries over. Checked against ncu's L2 read misses
    (agentic/gb10, Oct 20)."""
    M, N, K = shape["M"], shape["N"], shape["K"]
    bm, bn, bk = h["block_M"], h["block_N"], h["block_K"]
    panel = int(h.get("swizzle", 0) or 0)
    key = (M, N, K, bm, bn, bk, panel, resident_blocks, cap)
    if key in _kslice_cache:
        return _kslice_cache[key]
    # per wave: A and B read misses, and the rows and columns of tiles in flight (neighbouring
    # column tiles read neighbouring pieces of the same B rows at the same k-step, so DRAM sees one
    # long run)
    tiles_m, tiles_n, KT = math.ceil(M / bm), math.ceil(N / bn), math.ceil(K / bk)
    a_sz, b_sz = min(bm, M) * bk * 2, bk * min(bn, N) * 2
    # the L2 protects reused data from single-use data (probe l2policy.cu: a re-read 4-8 MB set
    # survives a 64 MB stream): two LRU lists, slices seen once (probation) and slices hit at least
    # once (protected); a hit promotes; eviction takes the oldest probation slice first, then the
    # oldest protected one
    cache = Cache(cap)
    per_wave = []
    order = list(launch_order(tiles_m, tiles_n, panel))
    for w in range(0, len(order), resident_blocks):
        wave = order[w : w + resident_blocks]
        rows = sorted({r for r, _ in wave})
        cols = sorted({c for _, c in wave})
        reads = {"A": 0.0, "B": 0.0}
        for k in range(KT):
            for kind, ids, sz in (("A", rows, a_sz), ("B", cols, b_sz)):
                for i in ids:
                    if cache.touch((kind, i, k), sz):
                        reads[kind] += sz
        per_wave.append(
            {
                "A": reads["A"],
                "B": reads["B"],
                "rows": len(rows),
                "cols": len(cols),
                "blocks": len(wave),
            }
        )
    out = {"waves": per_wave, "c_over": max(0.0, M * N * 2 - cap)}
    _kslice_cache[key] = out
    return out


def bank_conflict_ways(row_stride_bytes: int) -> int:
    """ldmatrix reads 8 rows of 16 bytes per 8x8 matrix. Shared memory has 32 banks of 4 bytes, so a
    16-byte row covers
    a quarter of them; rows whose addresses fall on the same quarter serialize. Ways = 8 / distinct
    quarters touched."""
    quarters = {(r * row_stride_bytes // 16) % 8 for r in range(8)}
    return 8 // len(quarters)
