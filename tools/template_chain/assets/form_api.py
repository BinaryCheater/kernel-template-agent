"""The decision-level form every family writes against. Harness code; do not edit."""

import math
from dataclasses import dataclass

SMS = 48


def occupancy(threads: int, regs: int, smem: int) -> int:
    """Blocks per SM: the smallest of the register, shared-memory and thread walls."""
    warps = math.ceil(threads / 32)
    per_warp = math.ceil(max(regs, 16) * 32 / 256) * 256
    return max(1, min(65536 // (warps * per_warp), 102400 // (smem + 1024), 1536 // threads, 24))


def spill_share(need: float, threads: int) -> float:
    """Registers past the per-thread limit, which falls to 128 above 256 threads a block."""
    limit = 255 if threads <= 256 else 128
    return max(0.0, need - limit) / need


@dataclass(frozen=True)
class Amounts:
    """What one launch of one kernel asks of the machine."""

    tensor_gflop: float  # tensor-core products, 1e9 element multiplies
    scalar_mop: float  # per-element scalar operations (exp, normalisation, decode), 1e6
    copy_mb: float  # bytes copied from global to shared memory, 1e6, counting repeats
    dram_mb: float  # bytes that miss the L2 per call, 1e6; 0 when inputs are warm
    store_mb: float  # bytes written back to global memory, 1e6
    blocks: int  # blocks in the grid
    trips: int  # pipeline iterations of one block
    stages: int  # pipeline stages
    warps: int  # threads // 32
    occ: int  # blocks per SM, from occupancy()
    spill: float  # share of accumulators past the register limit, from spill_share()
    rotating: bool  # inputs are not warm in the L2 at each call
