"""Decision-level time form: amounts from decisions, one bottleneck, regimes as small tables.

T = launch + stores + max(compute, l2_copies, dram) + turnover + stage cost + barrier cost

Every amount is a closed expression of the family's decision variables and the shape; every rate
is one constant, or one entry of a table indexed by a regime that is itself a function of the
decisions. That restriction is what lets `mip.py` export the form without approximation.
"""

import math
from dataclasses import dataclass

CLOCK_GHZ = 2.5
SMS = 48


def occupancy(threads: int, regs: int, smem: int) -> int:
    """Blocks per SM: the walls of `kta.models.gb10sim.kernel.occupancy`."""
    warps = math.ceil(threads / 32)
    per_warp = math.ceil(regs * 32 / 256) * 256
    return max(1, min(65536 // (warps * per_warp), 102400 // (smem + 1024), 1536 // threads, 24))


@dataclass(frozen=True)
class Amounts:
    """What one assignment asks of the machine, in units fixed by the term table."""

    tensor_gflop: float  # tensor products, 1e9
    scalar_mop: float  # per-element scalar work (softmax), 1e6
    copy_mb: float  # bytes copied global to shared
    dram_mb: float  # bytes that must come from DRAM on a rotating input
    store_mb: float  # bytes written back
    blocks: int
    trips: int  # pipeline iterations of a block
    stages: int
    warps: int
    occ: int
    spill: bool
    rotating: bool

    @property
    def wps(self) -> float:
        return self.warps * self.occ / 4

    @property
    def use(self) -> float:
        return min(1.0, self.wps)

    @property
    def waves(self) -> float:
        return self.blocks / min(self.blocks, SMS * self.occ)


def regime(a: Amounts) -> str:
    """Tensor issue regime: warps sharing one tensor pipe."""
    return "w1" if a.wps <= 1 else "w15" if a.wps < 2 else "w2"


def hgemm(s: dict, x: dict, regs: int, smem: int, cache: str) -> Amounts:
    M, N, K = s["M"], s["N"], s["K"]
    BM, BN, BK, st, T = x["BM"], x["BN"], x["BK"], x["s"], x["T"]
    nb = math.ceil(M / BM) * math.ceil(N / BN)
    kt = math.ceil(K / BK)
    return Amounts(
        tensor_gflop=M * N * K / 1e9,
        scalar_mop=0.0,
        copy_mb=nb * kt * (BM + BN) * BK * 2 / 1e6,
        dram_mb=(M * K + K * N) * 2 / 1e6,
        store_mb=M * N * 2 / 1e6,
        blocks=nb,
        trips=kt,
        stages=st,
        warps=T // 32,
        occ=occupancy(T, regs, smem),
        spill=regs >= 250,
        rotating=cache == "rotating",
    )


def fa(s: dict, x: dict, regs: int, smem: int, cache: str) -> Amounts:
    S, H, D = s["S"], s["H"], s["D"]
    BM, BN, st, T = x["BM"], x["BN"], x["s"], x["T"]
    nb = math.ceil(S / BM) * H
    kt = math.ceil(S / BN)
    return Amounts(
        tensor_gflop=2 * S * S * H * D / 1e9,
        scalar_mop=S * S * H / 1e6,
        copy_mb=(S * S * H * D * 4 / BM + S * H * D * 2) / 1e6,
        dram_mb=3 * S * H * D * 2 / 1e6,
        store_mb=S * H * D * 2 / 1e6,
        blocks=nb,
        trips=kt,
        stages=st,
        warps=T // 32,
        occ=occupancy(T, regs, smem),
        spill=regs >= 250,
        rotating=cache == "rotating",
    )


def decisions(r: dict) -> dict:
    p = r["pick"]
    if r["family"] == "hgemm":
        return {
            "BM": p["block_M"],
            "BN": p["block_N"],
            "BK": p["block_K"],
            "s": p["num_stages"],
            "T": p["threads"],
        }
    return {"BM": p["BM"], "BN": p["BN"], "s": p["stages"], "T": p["threads"]}


def amounts(r: dict) -> Amounts:
    c = r.get("counters") or {}
    regs = c.get("launch__registers_per_thread") or r["res"].get("regs") or 200
    build = hgemm if r["family"] == "hgemm" else fa
    return build(r["shape"], decisions(r), regs, r["smem"], r.get("cache") or "warm")


def terms(a: Amounts) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """The form as data: additive terms and the branches of the max, linear in constants."""
    reg = regime(a)
    compute = {
        f"tensor_{reg}": a.tensor_gflop / a.use,
        f"scalar_{reg}": a.scalar_mop / a.use,
    }
    if a.spill:
        compute["spill"] = a.tensor_gflop / a.use
    branches = {"compute": compute, "copy": {"copy": a.copy_mb}}
    if a.rotating:
        branches["dram"] = {"dram": a.dram_mb}
    additive = {
        "launch": 1.0,
        "store": a.store_mb,
        "stage": float(a.stages),
        "wave": a.waves,
        "barrier": max(0, a.warps - 4) * a.waves * a.trips / 10,
    }
    return additive, branches


def branch_us(terms_: dict[str, float], c: dict[str, float]) -> float:
    return sum(c.get(k, 0.0) * v for k, v in terms_.items())


def time_us(a: Amounts, c: dict[str, float]) -> float:
    additive, branches = terms(a)
    return branch_us(additive, c) + max(branch_us(b, c) for b in branches.values())


def active_branch(a: Amounts, c: dict[str, float]) -> str:
    _, branches = terms(a)
    return max(branches, key=lambda k: branch_us(branches[k], c))


# Zero fit: every value from gb10.json and the ISA, none from a kernel of these families.
DERIVED = {
    "launch": 2.8,  # launch_event_us
    "tensor_w1": 2e6 / (SMS * 1024 * CLOCK_GHZ * 0.833),  # tensor_flop, tensor_eff_gemm_loop[1]
    "tensor_w15": 2e6 / (SMS * 1024 * CLOCK_GHZ * 0.895),
    "tensor_w2": 2e6 / (SMS * 1024 * CLOCK_GHZ * 0.895),
    "copy": 1e3 / 2250,  # cpasync_l2_aggregate_GBs
    "dram": 1e3 / 237,  # dram_read_GBs
    "store": 1e3 / (244 * CLOCK_GHZ),  # sim.stg_chip_wide_B_per_clk
}
