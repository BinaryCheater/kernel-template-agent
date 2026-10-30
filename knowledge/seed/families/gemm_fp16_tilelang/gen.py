"""Generator of the gemm_fp16_tilelang family: one TileLang prim_func per assignment and shape.

``generate(assignment, case)`` returns a GeneratedKernel whose ``kernel.py`` defines the
prim_func ``gemm(A, B, C)`` for the shape in ``case`` (M, N, K are compile-time constants),
with a ``kta:step <marker> begin/end`` region for every chain item. The launch spec gives
the grid, the block and the dynamic shared memory the pipelined A/B stages need.

TileLang pass configs (decision D281): warp specialization and TMA lowering are disabled,
because the harness launches cubins directly and cannot build the host-side tensor maps
TMA needs; copies stay cp.async, as the chain's async_transfer step states.
"""

from __future__ import annotations

from typing import Any

from kta.backends.base import GeneratedKernel, LaunchSpec, Param

DEFAULT_SHAPE = {"M": 256, "N": 256, "K": 256}
PASS_CONFIGS = {"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}
MARKERS = [
    "tiles",
    "tiles_store",
    "k_split",
    "ab_smem",
    "ab_async",
    "ab_stages",
    "ab_swizzle",
    "mma",
]

KERNEL = """\
import tilelang.language as T

M, N, K = {M}, {N}, {K}
BM, BN, BK = {BM}, {BN}, {BK}
STAGES = {S}
THREADS = {threads}
SPLIT = {split}
K_PER = {K_PER}  # reduction indices per CTA along K (K when SPLIT == 1)
PASS_CONFIGS = {PASS_CONFIGS!r}


@T.prim_func
def gemm(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    # kta:step tiles begin  (one BM x BN output tile per CTA; SPLIT CTAs per tile along K)
    with T.Kernel({grid_expr}, threads=THREADS) as ({grid_vars}):
        # kta:step tiles end
        # kta:step ab_smem begin  (dedicated shared-memory buffers for the A and B k-tiles)
        A_shared = T.alloc_shared((BM, BK), "float16")
        B_shared = T.alloc_shared((BK, BN), "float16")
        # kta:step ab_smem end
        # kta:step ab_swizzle begin  (bank map: TileLang's layout inference gives A_shared and
        # B_shared the XOR-swizzled layout that T.gemm's ldmatrix reads need; not overridden)
        # kta:step ab_swizzle end
        C_local = T.alloc_fragment((BM, BN), "float32")
        T.clear(C_local)
        k0 = {k0_expr}
        # kta:step k_split begin  (groups of BK reduction indices, summed in sequence)
        # kta:step ab_stages begin  (STAGES rotating slots, loads issued STAGES - 1 k-tiles ahead)
        for ko in T.Pipelined(T.ceildiv(K_PER, BK), num_stages=STAGES):
            # kta:step ab_stages end
            # kta:step ab_async begin  (cp.async global -> shared; the pipeline awaits completion)
            T.copy(A[by * BM, k0 + ko * BK], A_shared)
            T.copy(B[k0 + ko * BK, bx * BN], B_shared)
            # kta:step ab_async end
            # kta:step mma begin  (mma.sync m16n8k16, FP16 operands, FP32 accumulator)
            T.gemm(A_shared, B_shared, C_local)
            # kta:step mma end
        # kta:step k_split end
        # kta:step tiles_store begin  (residue tiles are clipped at the M and N edges)
        T.copy(C_local, C[by * BM, bx * BN])
        # kta:step tiles_store end
"""


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def generate(assignment: dict[str, Any], case: dict[str, Any]) -> GeneratedKernel:
    shape = {k: int(case.get(k, DEFAULT_SHAPE[k])) for k in ("M", "N", "K")}
    a = {k: int(assignment[k]) for k in ("BM", "BN", "BK", "S", "threads", "split")}
    split = a["split"]
    k_per = _ceil_div(_ceil_div(shape["K"], split), a["BK"]) * a["BK"] if split > 1 else shape["K"]
    grid = [_ceil_div(shape["N"], a["BN"]), _ceil_div(shape["M"], a["BM"])]
    if split > 1:
        grid.append(split)
        grid_expr = "T.ceildiv(N, BN), T.ceildiv(M, BM), SPLIT"
        grid_vars = "bx, by, bz"
        k0_expr = "bz * K_PER"
    else:
        grid_expr = "T.ceildiv(N, BN), T.ceildiv(M, BM)"
        grid_vars = "bx, by"
        k0_expr = "0"
    source = KERNEL.format(
        **shape,
        **a,
        K_PER=k_per,
        PASS_CONFIGS=PASS_CONFIGS,
        grid_expr=grid_expr,
        grid_vars=grid_vars,
        k0_expr=k0_expr,
    )
    smem = a["S"] * (a["BM"] + a["BN"]) * a["BK"] * 2
    params = [Param("A", "f16*"), Param("B", "f16*"), Param("C", "f16*")]
    launch = LaunchSpec(grid, [a["threads"]], smem, params)
    return GeneratedKernel("tilelang", {"kernel.py": source}, "gemm", launch, list(MARKERS))
