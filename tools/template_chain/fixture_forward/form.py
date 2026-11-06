"""# kta:fixture hand-written family that exercises the harness; never scored as an arm."""

import math

from form_api import Amounts, occupancy, spill_share

FACTORS = {
    "BM": (32, 48, 64, 96, 128),
    "BN": (32, 48, 64, 96, 128),
    "s": (1, 2, 3),
    "T": (64, 128, 256),
}
STRUCTURE = ("BM", "BN", "T")
ANCHOR = {"BM": 64, "BN": 64, "s": 2, "T": 128}


def smem(x: dict, d: int) -> int:
    return (2 * x["BM"] + 2 * x["s"] * x["BN"]) * d * 2


def legal(shape: dict, x: dict) -> bool:
    return smem(x, shape["D"]) + 1024 <= 102400


def amounts(shape: dict, x: dict, records) -> Amounts:
    S, H, D = shape["S"], shape["H"], shape["D"]
    BM, BN, st, T = x["BM"], x["BN"], x["s"], x["T"]
    r = 4.09 * BM * BN / T + 1.87 * BM * D / T - 61.5 + 87.5 * (st >= 2) + 45.9 * (st >= 3)
    regs = records[0]["regs"] if records else max(r, 16)
    size = records[0]["smem"] if records else smem(x, D)
    return Amounts(
        tensor_gflop=S * S * H * D / 1e9,  # causal: half of the two products
        scalar_mop=S * S * H / 2e6,
        copy_mb=(S * S * H * D * 2 / BM + S * H * D * 2) / 1e6,
        dram_mb=0.0,
        store_mb=S * H * D * 2 / 1e6,
        blocks=math.ceil(S / BM) * H,
        trips=math.ceil(S / BN) // 2 + 1,
        stages=st,
        warps=T // 32,
        occ=occupancy(T, regs, size),
        spill=spill_share(regs, T),
        rotating=False,
    )
