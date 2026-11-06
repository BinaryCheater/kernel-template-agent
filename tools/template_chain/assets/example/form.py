"""Example family: plain FP16 GEMM, C = A @ B.T. Not related to your task."""

import math

from form_api import Amounts, occupancy, spill_share

FACTORS = {
    "BM": (64, 96, 128, 192, 256),
    "BN": (64, 96, 128, 192, 256),
    "BK": (16, 32, 48, 64, 96),
    "s": (1, 2, 3, 4),
    "T": (64, 128, 256),
}
STRUCTURE = ("BM", "BN", "T")
ANCHOR = {"BM": 128, "BN": 128, "BK": 32, "s": 3, "T": 128}


def smem(x: dict) -> int:
    return x["s"] * (x["BM"] + x["BN"]) * x["BK"] * 2


def legal(shape: dict, x: dict) -> bool:
    return smem(x) + 1024 <= 102400 and shape["K"] % x["BK"] == 0


def amounts(shape: dict, x: dict, records) -> Amounts:
    M, N, K = shape["M"], shape["N"], shape["K"]
    BM, BN, BK, st, T = x["BM"], x["BN"], x["BK"], x["s"], x["T"]
    regs = records[0]["regs"] if records else BM * BN / T + 62  # accumulators plus addressing
    size = records[0]["smem"] if records else smem(x)
    blocks = math.ceil(M / BM) * math.ceil(N / BN)
    trips = math.ceil(K / BK)
    return Amounts(
        tensor_gflop=M * N * K / 1e9,
        scalar_mop=0.0,
        copy_mb=blocks * trips * (BM + BN) * BK * 2 / 1e6,
        dram_mb=0.0,
        store_mb=M * N * 2 / 1e6,
        blocks=blocks,
        trips=trips,
        stages=st,
        warps=T // 32,
        occ=occupancy(T, regs, size),
        spill=spill_share(regs, T),
        rotating=False,
    )
