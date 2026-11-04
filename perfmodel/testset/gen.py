"""Generate the test-set kernels (28 §5, first cut) with TileLang and write one CUDA harness each.

Runs inside the worker image (`kta-worker:tilelang-0.1.15`, TileLang 0.1.15) on spark:

    python perfmodel/testset/gen.py <out_dir>

For every kernel it writes `<out>/<name>/kernel.cu` (TileLang's CUDA source), `harness.cu` (the
launch with event timing), `launch.json` (grid, block, dynamic shared memory, shape, tile holes).
Compilation and SASS dumps are done by perfmodel/remote/build.sh.

HGEMM: the KTA family `knowledge/seed/families/gemm_fp16_tilelang` (its kernel template, copied
here because the image has no KTA install; same pass configs, D281: cp.async, no warp
specialisation) at one shape, 2048^3, varying tile, stages, threads and the raster order
(`T.use_swizzle`, TileLang's panelled launch order). FA forward: TileLang's flash-attention forward
pattern (online softmax, exp2), B=1, S=2048, H=8, D=128, non-causal, the same pass configs.

Round 2 (28 §5 test-set table): decode attention, GQA split-KV, as TileLang's flash-decoding
pattern split into its two kernels (partial attention per KV split, then the combine over splits);
a w4a16 GEMV (TileLang's dequantize-GEMV pattern: int4 weights packed two per byte, fp16 group
scales of 128, fp32 accumulation, a thread all-reduce); RMSNorm and row softmax (one row per block,
fp16 in and out, fp32 reductions). The DRAM-bound ones also run `rotating`: `rot` input sets used
in turn, rot x footprint above the L2 (gb10-measure).

Round 6: a blind set (HGEMM at two other shapes, FA, RMSNorm and softmax at other sizes) and the
split-KV decode as one sequence of two kernels (`_decode_seq`: launch.json family `sequence`).

Round 7: one-change variants (`_whatif`: launch.json `whatif` names the base unit and the axis; the
table scores predicted delta against measured delta) and larger blind groups (`_blind7`).
`ONLY` limits generation to the matching names; `SHARD=i/n` generates one share of them.
"""

import json
import math
import os
import re
import sys
import traceback
from pathlib import Path

import tilelang
import tilelang.language as T

PASS_CONFIGS = {"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}
SHAPE = {"M": 2048, "N": 2048, "K": 2048}
# name: (BM, BN, BK, stages, threads, raster panel or 0)
HGEMM = {
    "c128x128x32_s3_t128": (128, 128, 32, 3, 128, 0),
    "c128x128x32_s4_t128": (128, 128, 32, 4, 128, 0),
    "c128x128x64_s2_t128": (128, 128, 64, 2, 128, 0),
    "c128x128x64_s3_t256": (128, 128, 64, 3, 256, 0),
    "c128x256x32_s3_t256": (128, 256, 32, 3, 256, 0),
    "c256x128x32_s3_t256": (256, 128, 32, 3, 256, 0),
    "c64x128x32_s4_t128": (64, 128, 32, 4, 128, 0),
    "c128x256x64_s2_t256": (128, 256, 64, 2, 256, 0),
    "c128x128x32_s3_t128_sw8": (128, 128, 32, 3, 128, 8),
    "c128x256x32_s3_t256_sw4": (128, 256, 32, 3, 256, 4),
}
FA = {"B": 1, "S": 2048, "H": 8, "D": 128, "BM": 64, "BN": 64, "stages": 2, "threads": 128}
DECODE = {"B": 1, "H": 32, "G": 8, "S": 4096, "D": 128, "BN": 64, "BH": 64, "split": 8, "stages": 2}
# round 3: FP8 GEMM (e4m3 inputs, B stored N x K, fp32 accumulation, fp16 out) at 2048^3
FP8 = {
    "c128x128x64_s3_t128": (128, 128, 64, 3, 128),
    "c128x256x64_s3_t256": (128, 256, 64, 3, 256),
}
# round 4: MoE grouped GEMM, E experts with an equal share of the tokens each (balanced routing),
# tokens sorted by expert: C[t] = A[t] W[e(t)]
# w4a16 B dequantised in the main loop (round 5; sm_121a's ptxas refuses the NVFP4 block-scaled mma)
W4GEMM = {
    "c128x128x32_s3_t128": (128, 128, 32, 3, 128),
    "c64x128x32_s4_t128": (64, 128, 32, 4, 128),
}
W4GEMM_GROUP = 32
MOE = {"E": 8, "T": 2048, "N": 1024, "K": 2048}
MOE_TILES = {
    "c128x128x32_s3_t128": (128, 128, 32, 3, 128),
    "c64x128x32_s4_t128": (64, 128, 32, 4, 128),
}
# round 6, blind rows: scored by the engine as it stood before they were measured
HGEMM_BLIND = {
    (4096, 1024, 2048): {
        "c128x128x32_s3_t128": (128, 128, 32, 3, 128, 0),
        "c256x128x32_s3_t256": (256, 128, 32, 3, 256, 0),
        "c128x256x64_s2_t256": (128, 256, 64, 2, 256, 0),
    },
    (1024, 1024, 1024): {
        "c128x128x32_s4_t128": (128, 128, 32, 4, 128, 0),
        "c64x128x32_s4_t128": (64, 128, 32, 4, 128, 0),
        "c128x128x64_s3_t256": (128, 128, 64, 3, 256, 0),
    },
}
# round 7: one blind shape, 14 variants (warm and rotating)
HGEMM_BLIND7_SHAPE = {"M": 3072, "N": 1536, "K": 2048}
HGEMM_BLIND7 = {
    "c128x128x32_s3_t128": (128, 128, 32, 3, 128, 0),
    "c128x128x32_s2_t128": (128, 128, 32, 2, 128, 0),
    "c128x128x32_s4_t128": (128, 128, 32, 4, 128, 0),
    "c128x128x64_s2_t128": (128, 128, 64, 2, 128, 0),
    "c128x128x64_s3_t256": (128, 128, 64, 3, 256, 0),
    "c128x256x32_s3_t256": (128, 256, 32, 3, 256, 0),
    "c256x128x32_s3_t256": (256, 128, 32, 3, 256, 0),
    "c64x128x32_s4_t128": (64, 128, 32, 4, 128, 0),
    "c128x64x32_s4_t128": (128, 64, 32, 4, 128, 0),
    "c128x256x64_s2_t256": (128, 256, 64, 2, 256, 0),
    "c256x256x32_s2_t256": (256, 256, 32, 2, 256, 0),
    "c64x64x32_s4_t64": (64, 64, 32, 4, 64, 0),
    "c128x128x32_s3_t128_sw8": (128, 128, 32, 3, 128, 8),
    "c128x256x32_s3_t256_sw4": (128, 256, 32, 3, 256, 4),
}
FA_BLIND7 = {"B": 1, "S": 1536, "H": 12, "D": 96}
# BM, BN, stages, threads
FA_BLIND7_V = {
    "c64x64_s2_t128": (64, 64, 2, 128),
    "c64x64_s3_t128": (64, 64, 3, 128),
    "c64x64_s1_t128": (64, 64, 1, 128),
    "c64x32_s3_t128": (64, 32, 3, 128),
    "c64x128_s2_t128": (64, 128, 2, 128),
    "c128x64_s2_t128": (128, 64, 2, 128),
    "c128x64_s2_t256": (128, 64, 2, 256),
    "c128x128_s2_t256": (128, 128, 2, 256),
    "c64x64_s2_t64": (64, 64, 2, 64),
    "c32x64_s2_t64": (32, 64, 2, 64),
}
# round 7, one change at a time from a base: name: (BM, BN, BK, stages, threads, panel, variant
# options, axis); `base` names the row it is compared with (default the first row of its table)
WI_HGEMM = {
    "b": (128, 128, 32, 3, 128, 0, {}, "base"),
    "s1": (128, 128, 32, 1, 128, 0, {}, "stages"),
    "s2": (128, 128, 32, 2, 128, 0, {}, "stages"),
    "s4": (128, 128, 32, 4, 128, 0, {}, "stages"),
    "s6": (128, 128, 32, 6, 128, 0, {}, "stages"),
    "direct": (128, 128, 32, 0, 128, 0, {}, "cp.async vs direct"),
    "bk16": (128, 128, 16, 3, 128, 0, {}, "tile"),
    "bk64": (128, 128, 64, 3, 128, 0, {}, "tile"),
    "bm64": (64, 128, 32, 3, 128, 0, {}, "tile"),
    "bn64": (128, 64, 32, 3, 128, 0, {}, "tile"),
    "bm256": (256, 128, 32, 3, 256, 0, {}, "tile+threads"),
    "bn256": (128, 256, 32, 3, 256, 0, {}, "tile+threads"),
    "t64": (128, 128, 32, 3, 64, 0, {}, "threads"),
    "t256": (128, 128, 32, 3, 256, 0, {}, "threads"),
    "sw4": (128, 128, 32, 3, 128, 4, {}, "raster"),
    "sw8": (128, 128, 32, 3, 128, 8, {}, "raster"),
    "nocompute": (128, 128, 32, 3, 128, 0, {"compute": False}, "compute removed"),
    "noload": (128, 128, 32, 0, 128, 0, {"load": False}, "loads removed"),
    "nostore": (128, 128, 32, 3, 128, 0, {"store": "shared"}, "stores removed"),
}
# the overlap question at other load/compute balances
WI_HGEMM2 = {
    "L_s3": (128, 256, 64, 3, 256, 0, {}, "base"),
    "L_s1": (128, 256, 64, 1, 256, 0, {}, "stages"),
    "L_s2": (128, 256, 64, 2, 256, 0, {}, "stages"),
}
WI_HGEMM3 = {
    "S_s4": (64, 128, 32, 4, 128, 0, {}, "base"),
    "S_s1": (64, 128, 32, 1, 128, 0, {}, "stages"),
    "S_s2": (64, 128, 32, 2, 128, 0, {}, "stages"),
}
WI_FA = {
    "b": ({}, "base"),
    "s1": ({"stages": 1}, "stages"),
    "s3": ({"stages": 3}, "stages"),
    "s4": ({"stages": 4}, "stages"),
    "direct": ({"stages": 0}, "cp.async vs direct"),
    "bn32": ({"BN": 32}, "tile"),
    "bn128": ({"BN": 128}, "tile"),
    "bm128": ({"BM": 128}, "tile"),
    "t64": ({"threads": 64}, "threads"),
    "bm128_t256": ({"BM": 128, "threads": 256, "base": "bm128"}, "threads"),
    "nostore": ({"store": "shared"}, "stores removed"),
    "nov": ({"second": False}, "compute removed"),
    "noexp": ({"exp": False}, "compute removed"),
}
WI_GEMV = {
    "b": ({}, "base"),
    "np2": ({"n_part": 2}, "threads"),
    "np8": ({"n_part": 8}, "threads"),
    "r16": ({"reduce": 16}, "threads"),
    "r64": ({"reduce": 64}, "threads"),
    "mk4": ({"mk": 4}, "bytes per lane"),
    "mk16": ({"mk": 16}, "bytes per lane"),
    "nocompute": ({"compute": False}, "compute removed"),
}
WI_RMS = {
    "b": ({}, "base"),
    "t64": ({"threads": 64}, "threads"),
    "t256": ({"threads": 256}, "threads"),
    "fp32": ({"dtype": "float"}, "bytes per lane"),
    "nocompute": ({"compute": False}, "compute removed"),
    "nostore": ({"store": "shared"}, "stores removed"),
}
WI_SOFTMAX = {
    "b": ({}, "base"),
    "t64": ({"threads": 64}, "threads"),
    "t256": ({"threads": 256}, "threads"),
    "nostore": ({"store": "shared"}, "stores removed"),
}
WI_DECODE = {
    "b": ({}, "base"),
    "s1": ({"stages": 1}, "stages"),
    "s3": ({"stages": 3}, "stages"),
    "split4": ({"split": 4}, "blocks"),
    "split16": ({"split": 16}, "blocks"),
    "bn32": ({"BN": 32}, "tile"),
    "S2048": ({"S": 2048}, "shape"),
    "S8192": ({"S": 8192}, "shape"),
}
# rotation depth: input sets used in turn, times the base depth (the replay takes every read cold)
WI_ROT = (2, 4)
WI_W4 = {
    "b": ({}, "base"),
    "nocompute": ({"compute": False}, "compute removed"),
    "nodequant": ({"dequant": False}, "dequantisation removed"),
}
FA_BLIND = {"B": 1, "S": 1024, "H": 16, "D": 64, "BM": 64, "BN": 64, "stages": 2, "threads": 128}
RMS_BLIND = {"M": 2048, "N": 1024}
SOFTMAX_BLIND = {"M": 1024, "N": 2048}
GEMV = {"N": 4096, "K": 4096, "group": 128, "n_part": 4, "reduce": 32}
ROWS = {"M": 512, "N": 4096}


def hgemm(
    BM: int,
    BN: int,
    BK: int,
    S: int,
    threads: int,
    panel: int,
    shape: dict[str, int] = SHAPE,
    load: bool = True,
    compute: bool = True,
    store: str = "global",
):
    """Round 7, one-change variants: `S` 0 a synchronous copy (no cp.async), `load` False keeps the
    tile in shared memory over the k-loop (no global reads), `compute` False drops the mma,
    `store` "shared" writes the C tile to shared memory (no global stores)."""
    M, N, K = shape["M"], shape["N"], shape["K"]

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((BM, BK), "float16")
            B_shared = T.alloc_shared((BK, BN), "float16")
            C_local = T.alloc_fragment((BM, BN), "float32")
            O_shared = T.alloc_shared((BM, BN), "float16") if store == "shared" else None
            t = T.alloc_local((1,), "float")
            if panel:
                T.use_swizzle(panel_size=panel)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=S):
                if load:
                    T.copy(A[by * BM, ko * BK], A_shared)
                    T.copy(B[ko * BK, bx * BN], B_shared)
                if compute:
                    T.gemm(A_shared, B_shared, C_local)
                else:  # the copies stay live: one element of each tile
                    t[0] = T.Cast("float", A_shared[0, 0]) + T.Cast("float", B_shared[0, 0])
            if not compute:  # the C tile keeps its stores, spread over the threads as with the mma
                for i, j in T.Parallel(BM, BN):
                    C_local[i, j] = t[0]
            if store == "shared":
                T.copy(C_local, O_shared)
            else:
                T.copy(C_local, C[by * BM, bx * BN])

    return gemm


def fp8gemm(BM: int, BN: int, BK: int, S: int, threads: int):
    """TileLang's FP8 GEMM pattern (examples/gemm_fp8): C = A B^T, A M x K and B N x K in e4m3, fp32
    accumulation, fp16 out; the same pass configs as the HGEMMs."""
    M, N, K = SHAPE["M"], SHAPE["N"], SHAPE["K"]
    f8 = "float8_e4m3"

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), f8),
        B: T.Tensor((N, K), f8),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((BM, BK), f8)
            B_shared = T.alloc_shared((BN, BK), f8)
            C_local = T.alloc_fragment((BM, BN), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=S):
                T.copy(A[by * BM, ko * BK], A_shared)
                T.copy(B[bx * BN, ko * BK], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * BM, bx * BN])

    return gemm


def moe_gemm(E: int, T_: int, N: int, K: int, BM: int, BN: int, BK: int, S: int, threads: int):
    """MoE expert GEMM over grouped tokens (the HGEMM template with a per-block expert): tokens
    sorted by expert, T_/E per expert (a multiple of BM), expert weights E x K x N in fp16; block
    row `by` belongs to expert by // (T_/E/BM)."""
    per = T_ // E // BM

    @T.prim_func
    def gemm(
        A: T.Tensor((T_, K), "float16"),
        W: T.Tensor((E, K, N), "float16"),
        C: T.Tensor((T_, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(T_, BM), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((BM, BK), "float16")
            B_shared = T.alloc_shared((BK, BN), "float16")
            C_local = T.alloc_fragment((BM, BN), "float32")
            e = by // per
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=S):
                T.copy(A[by * BM, ko * BK], A_shared)
                T.copy(W[e, ko * BK, bx * BN], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * BM, bx * BN])

    return gemm


def w4gemm(
    BM: int,
    BN: int,
    BK: int,
    S: int,
    threads: int,
    group: int,
    compute: bool = True,
    dequant: bool = True,
):
    """GEMM with a w4a16 B dequantised in the main loop (TileLang's dequantize-GEMM pattern fused
    into the k-loop): each k-step copies A, the packed int4 B tile and its fp16 group scales to
    shared memory, the block's threads expand B to fp16 there, then the mma. Two int4 per int8."""
    M, N, K = SHAPE["M"], SHAPE["N"], SHAPE["K"]
    SC = BK // group if BK >= group else 1

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float16"),
        Bq: T.Tensor((N, K // 2), "int8"),
        Sc: T.Tensor((N, K // group), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((BM, BK), "float16")
            Bq_shared = T.alloc_shared((BN, BK // 2), "int8")
            Sc_shared = T.alloc_shared((BN, SC), "float16")
            B_shared = T.alloc_shared((BN, BK), "float16")
            C_local = T.alloc_fragment((BM, BN), "float32")
            t = T.alloc_local((1,), "float")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=S):
                T.copy(A[by * BM, ko * BK], A_shared)
                T.copy(Bq[bx * BN, ko * (BK // 2)], Bq_shared)
                T.copy(Sc[bx * BN, ko * BK // group], Sc_shared)
                if dequant:
                    for i, j in T.Parallel(BN, BK):
                        B_shared[i, j] = (
                            T.Cast(
                                "float16",
                                (T.Cast("int32", Bq_shared[i, j // 2]) >> ((j % 2) * 4)) & 15,
                            )
                            - T.float16(8)
                        ) * Sc_shared[i, 0]
                else:  # the copies stay live
                    B_shared[0, 0] = T.Cast("float16", Bq_shared[0, 0]) * Sc_shared[0, 0]
                if compute:
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                else:
                    t[0] = T.Cast("float", A_shared[0, 0]) + T.Cast("float", B_shared[0, 0])
            if not compute:
                for i, j in T.Parallel(BM, BN):
                    C_local[i, j] = t[0]
            T.copy(C_local, C[by * BM, bx * BN])

    return gemm


def fa_forward(
    B: int,
    S: int,
    H: int,
    D: int,
    BM: int,
    BN: int,
    stages: int,
    threads: int,
    store: str = "global",
    second: bool = True,
    exp: bool = True,
):
    """Round 7 variants: `store` "shared" ends in shared memory, `second` False drops V's load and
    the second mma, `exp` False the exp2 of the online softmax."""
    shape = [B, S, H, D]
    scale = (1.0 / D) ** 0.5 * 1.44269504  # log2(e)
    dtype, accum = "float16", "float"

    @T.prim_func
    def fa(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Out: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(S, BM), H, B, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([BM, D], dtype)
            K_shared = T.alloc_shared([BN, D], dtype)
            V_shared = T.alloc_shared([BN, D], dtype)
            O_shared = T.alloc_shared([BM, D], dtype)
            acc_s = T.alloc_fragment([BM, BN], accum)
            acc_s_cast = T.alloc_fragment([BM, BN], dtype)
            acc_o = T.alloc_fragment([BM, D], accum)
            m = T.alloc_fragment([BM], accum)
            m_prev = T.alloc_fragment([BM], accum)
            sc = T.alloc_fragment([BM], accum)
            ssum = T.alloc_fragment([BM], accum)
            lsum = T.alloc_fragment([BM], accum)
            T.copy(Q[bz, bx * BM : (bx + 1) * BM, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(lsum, 0)
            T.fill(m, -T.infinity(accum))
            for k in T.Pipelined(T.ceildiv(S, BN), num_stages=stages):
                T.copy(K[bz, k * BN : (k + 1) * BN, by, :], K_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(m, m_prev)
                T.fill(m, -T.infinity(accum))
                T.reduce_max(acc_s, m, dim=1, clear=False)
                for i in T.Parallel(BM):
                    sc[i] = (T.exp2 if exp else _same)(m_prev[i] * scale - m[i] * scale)
                for i, j in T.Parallel(BM, D):
                    acc_o[i, j] *= sc[i]
                for i, j in T.Parallel(BM, BN):
                    acc_s[i, j] = (T.exp2 if exp else _same)(acc_s[i, j] * scale - m[i] * scale)
                T.copy(acc_s, acc_s_cast)
                if second:
                    T.copy(V[bz, k * BN : (k + 1) * BN, by, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, ssum, dim=1)
                for i in T.Parallel(BM):
                    lsum[i] = lsum[i] * sc[i] + ssum[i]
            for i, j in T.Parallel(BM, D):
                acc_o[i, j] /= lsum[i]
            T.copy(acc_o, O_shared)
            if store == "global":
                T.copy(O_shared, Out[bz, bx * BM : (bx + 1) * BM, by, :])

    return fa


def _same(x):
    return x


def decode_partial(
    B: int, H: int, G: int, S: int, D: int, BN: int, BH: int, split: int, stages: int
):
    """TileLang's GQA flash-decoding split kernel (examples/flash_decoding, without the mask)."""
    scale = (1.0 / D) ** 0.5 * 1.44269504
    dtype, accum = "float16", "float"
    vh = min(BH, H // G)
    seg = S // split
    vn = min(BN, seg)

    @T.prim_func
    def part(
        Q: T.Tensor([B, H, D], dtype),
        K: T.Tensor([B, S, G, D], dtype),
        V: T.Tensor([B, S, G, D], dtype),
        glse: T.Tensor([B, H, split], dtype),
        Opart: T.Tensor([B, H, split, D], dtype),
    ):
        with T.Kernel(B, H // vh, split, threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared([BH, D], dtype)
            K_shared = T.alloc_shared([BN, D], dtype)
            V_shared = T.alloc_shared([BN, D], dtype)
            O_shared = T.alloc_shared([vh, D], dtype)
            acc_s = T.alloc_fragment([BH, BN], accum)
            acc_s_cast = T.alloc_fragment([BH, BN], dtype)
            acc_o = T.alloc_fragment([BH, D], accum)
            m = T.alloc_fragment([BH], accum)
            m_prev = T.alloc_fragment([BH], accum)
            sc = T.alloc_fragment([BH], accum)
            ssum = T.alloc_fragment([BH], accum)
            lsum = T.alloc_fragment([BH], accum)
            kvh = by // ((H // G) // vh)
            T.copy(Q[bx, by * vh : by * vh + BH, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(lsum, 0)
            T.fill(m, -T.infinity(accum))
            for k in T.Pipelined(T.ceildiv(seg, BN), num_stages=stages):
                T.copy(K[bx, seg * bz + k * vn : seg * bz + (k + 1) * vn, kvh, :], K_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(m, m_prev)
                T.fill(m, -T.infinity(accum))
                T.reduce_max(acc_s, m, dim=1, clear=False)
                for i in T.Parallel(BH):
                    m[i] = T.max(m[i], m_prev[i])
                for i in T.Parallel(BH):
                    sc[i] = T.exp2(m_prev[i] * scale - m[i] * scale)
                for i, j in T.Parallel(BH, BN):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - m[i] * scale)
                T.reduce_sum(acc_s, ssum, dim=1)
                for i in T.Parallel(BH):
                    lsum[i] = lsum[i] * sc[i] + ssum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(BH, D):
                    acc_o[i, j] *= sc[i]
                T.copy(V[bx, seg * bz + k * vn : seg * bz + (k + 1) * vn, kvh, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(BH, D):
                acc_o[i, j] /= lsum[i]
            for i in T.Parallel(BH):
                lsum[i] = T.log2(lsum[i]) + m[i] * scale
            for i in T.Parallel(BH):
                if i < vh:
                    glse[bx, by * vh + i, bz] = lsum[i]
            T.copy(acc_o[:vh, :], O_shared)
            T.copy(O_shared, Opart[bx, by * vh : (by + 1) * vh, bz, :])

    return part


def decode_combine(B: int, H: int, D: int, split: int, **_):
    """The combine kernel of split-KV decoding: per head, the log-sum-exp weighted sum of the
    splits' partial outputs; one thread per output element."""

    @T.prim_func
    def comb(
        glse: T.Tensor([B, H, split], "float16"),
        Opart: T.Tensor([B, H, split, D], "float16"),
        Out: T.Tensor([B, H, D], "float16"),
    ):
        with T.Kernel(H, B, threads=D) as (by, bz):
            tx = T.get_thread_binding()
            mx = T.alloc_local([1], "float")
            sm = T.alloc_local([1], "float")
            acc = T.alloc_local([1], "float")
            mx[0] = -T.infinity("float")
            for k in T.serial(split):
                mx[0] = T.max(mx[0], T.Cast("float", glse[bz, by, k]))
            sm[0] = 0.0
            acc[0] = 0.0
            for k in T.serial(split):
                sm[0] += T.exp2(T.Cast("float", glse[bz, by, k]) - mx[0])
                acc[0] += T.exp2(T.Cast("float", glse[bz, by, k]) - mx[0]) * T.Cast(
                    "float", Opart[bz, by, k, tx]
                )
            Out[bz, by, tx] = T.Cast("float16", acc[0] / sm[0])

    return comb


def gemv_w4a16(
    N: int, K: int, group: int, n_part: int, reduce: int, mk: int = 8, compute: bool = True
):
    """TileLang's dequantize-GEMV pattern (examples/dequantize_gemm/example_dequant_gemv_fp16xint4):
    y = W x with W int4 (unsigned, offset 8) packed two per int8 along K, fp16 scale per group of
    `group` K elements; each warp one output row, 8 fp16 / 4 packed bytes per lane per step."""
    # mk: fp16 per load (8: a 16-byte load)
    mkc = mk // 2
    bk = reduce * mk

    @T.prim_func
    def gemv(
        A: T.Tensor((1, K), "float16"),
        Bq: T.Tensor((N, K // 2), "int8"),
        Sc: T.Tensor((N, K // group), "float16"),
        C: T.Tensor((1, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, n_part), 1, threads=(reduce, n_part)) as (bx, by):
            a = T.alloc_local((mk,), "float16")
            bq = T.alloc_local((mkc,), "int8")
            bd = T.alloc_local((mk,), "float16")
            acc = T.alloc_local((1,), "float")
            red = T.alloc_local((1,), "float")
            kr = T.thread_binding(0, reduce, thread="threadIdx.x")
            ni = T.thread_binding(0, n_part, thread="threadIdx.y")
            T.clear(acc)
            for ko in T.serial(T.ceildiv(K, bk)):
                for v in T.vectorized(mk):
                    a[v] = A[by, ko * bk + kr * mk + v]
                for v in T.vectorized(mkc):
                    bq[v] = Bq[bx * n_part + ni, ko * (reduce * mkc) + kr * mkc + v]
                if compute:
                    for ki in T.serial(mk):
                        bd[ki] = (
                            T.Cast("float16", (T.Cast("int32", bq[ki // 2]) >> ((ki % 2) * 4)) & 15)
                            - T.float16(8)
                        ) * Sc[bx * n_part + ni, (ko * bk + kr * mk) // group]
                for ki in T.serial(mk):
                    if compute:
                        acc[0] += T.Cast("float", a[ki]) * T.Cast("float", bd[ki])
                    else:
                        acc[0] += T.Cast("float", a[ki]) + T.Cast("float", bq[0])
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(T.uint32(1), acc[0], True, red[0], kr, dtype="handle")
                )
            if kr == 0:
                C[by, bx * n_part + ni] = T.Cast("float16", red[0])

    return gemv


def rmsnorm(
    M: int,
    N: int,
    threads: int = 128,
    compute: bool = True,
    store: str = "global",
    dtype: str = "float16",
):
    """RMSNorm with weight (TileLang's examples/norm/rms_norm.py, fp16 in and out): one row per
    block of 128 threads, the row in registers, an fp32 sum of squares."""

    @T.prim_func
    def rms(
        X: T.Tensor((M, N), dtype),
        W: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(M, threads=threads) as bx:
            x = T.alloc_fragment((1, N), dtype)
            sq = T.alloc_fragment((1, N), "float")
            ss = T.alloc_fragment((1,), "float")
            o_shared = T.alloc_shared((1, N), dtype) if store == "shared" else None
            T.copy(X[bx : bx + 1, :], x)
            if compute:
                for i, j in T.Parallel(1, N):
                    sq[i, j] = T.Cast("float", x[i, j]) * T.Cast("float", x[i, j])
                T.reduce_sum(sq, ss, dim=1)
                for i in T.Parallel(1):
                    ss[i] = T.rsqrt(ss[i] / N + 1e-6)
                for i, j in T.Parallel(1, N):
                    x[i, j] = T.Cast(
                        dtype, T.Cast("float", x[i, j]) * ss[i] * T.Cast("float", W[j])
                    )
            if store == "shared":
                T.copy(x, o_shared)
            else:
                T.copy(x, Y[bx : bx + 1, :])

    return rms


def softmax(M: int, N: int, threads: int = 128, store: str = "global"):
    """Row softmax (fp16 in and out, exp2 with the log2(e) scale, fp32 max and sum): one row per
    block of 128 threads."""

    @T.prim_func
    def smax(X: T.Tensor((M, N), "float16"), Y: T.Tensor((M, N), "float16")):
        with T.Kernel(M, threads=threads) as bx:
            x = T.alloc_fragment((1, N), "float16")
            s = T.alloc_fragment((1, N), "float")
            o_shared = T.alloc_shared((1, N), "float16") if store == "shared" else None
            mx = T.alloc_fragment((1,), "float")
            sm = T.alloc_fragment((1,), "float")
            T.copy(X[bx : bx + 1, :], x)
            for i, j in T.Parallel(1, N):
                s[i, j] = T.Cast("float", x[i, j]) * 1.44269504
            T.reduce_max(s, mx, dim=1)
            for i, j in T.Parallel(1, N):
                s[i, j] = T.exp2(s[i, j] - mx[i])
            T.reduce_sum(s, sm, dim=1)
            for i, j in T.Parallel(1, N):
                x[i, j] = T.Cast("float16", s[i, j] / sm[i])
            if store == "shared":
                T.copy(x, o_shared)
            else:
                T.copy(x, Y[bx : bx + 1, :])

    return smax


HARNESS = r"""// Test-set harness (perfmodel/testset/gen.py): TileLang's kernel, launched as TileLang launches it,
// on random inputs (fp16 values in [-1, 1) written over every buffer's bytes); WARM untimed launches,
// then REPS launches timed one by one with CUDA events (no profiler). ROT input sets are used in
// turn (ROT 1: warm; ROT x footprint above the L2: rotating). Usage: harness [reps] (reps 1 = one
// launch after warm-up, for the profiler).
#include <cuda.h>
#include "../../../probes/common.cuh"
%(includes)s
__global__ void fill(half_t* p, size_t n, unsigned seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned x = (unsigned)i * 2654435761u ^ seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    p[i] = half_t((float)(x & 0xffff) / 32768.0f - 1.0f);
  }
}
int main(int argc, char** argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 30;
  const int ROT = %(rot)d;
  const size_t bytes[] = {%(sizes)s};
  void* buf[ROT][%(nbuf)d];
  for (int r = 0; r < ROT; r++)
    for (int i = 0; i < %(nbuf)d; i++) {
      CK(cudaMalloc(&buf[r][i], bytes[i]));
      fill<<<1024, 256>>>((half_t*)buf[r][i], bytes[i] / 2, 0x9E3779B9u * (i + 1) + 77u * r);
    }
  CK(cudaDeviceSynchronize());
%(setup)s
  int next = 0;
  auto launch = [&](cudaStream_t st = 0) {
    void** b = buf[next++ %% ROT];
%(launches)s
  };
  for (int i = 0; i < 10; i++) launch();
  CK(cudaDeviceSynchronize());
  if (reps == 1) { launch(); CK(cudaDeviceSynchronize()); printf("RESULT %(name)s once\n"); return 0; }
  std::vector<double> v;
  for (int r = 0; r < reps; r++) v.push_back(event_us([&] { launch(); }));
  Stat s = stat_of(v);
  printf("RESULT %(name)s kernel median=%%.4f iqr=%%.4f unit=us n=%%d samples=", s.median, s.iqr, s.n);
  for (double x : v) printf("%%.3f,", x);
  printf("\n");
  // round 4: the same kernel as inference runs it: B2B launches back-to-back between two events
  // (per launch), and the same B2B launches captured in one CUDA graph (per launch)
  const int B2B = 50;
  std::vector<double> vb, vg;
  for (int r = 0; r < 3 + reps / 2; r++) {
    double x = event_us([&] { for (int i = 0; i < B2B; i++) launch(); }) / B2B;
    if (r >= 3) vb.push_back(x);
  }
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < B2B; i++) launch(st);
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < 3 + reps / 2; r++) {
    CK(cudaEventRecord(e0, st));
    CK(cudaGraphLaunch(ge, st));
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (r >= 3) vg.push_back(ms * 1e3 / B2B);
  }
  for (auto [tag, w] : {std::pair{"b2b", &vb}, std::pair{"graph", &vg}}) {
    Stat t = stat_of(*w);
    printf("RESULT %(name)s %%s median=%%.4f iqr=%%.4f unit=us n=%%d per_launch_of=%%d\n", tag, t.median, t.iqr, t.n, B2B);
  }
  CK(cudaDeviceSynchronize());
  return 0;
}
"""


def _params(src: str, entry: str) -> list[tuple[str, str]]:
    """(type, name) of each parameter of the kernel entry in TileLang's source, in TileLang's order
    (sorted by name, not the prim_func's order)."""
    m = re.search(rf"__global__ void(?: __launch_bounds__\([^)]*\))? {entry}\(([^)]*)\)", src)
    assert m, f"no parameter list for {entry}"
    out = []
    for p in m.group(1).split(","):
        words = p.replace("__restrict__", "").split()
        out.append((" ".join(words[:-1]), words[-1]))
    return out


def _compiled(func, buffers: list[str] | None = None) -> dict:
    """TileLang's source of `func`: entry, dynamic shared memory and the launch arguments as
    indices into `buffers` (default the prim_func's own parameters in order)."""
    k = tilelang.compile(func, target="cuda", pass_configs=PASS_CONFIGS)
    src = k.get_kernel_source()
    m = re.search(r"__global__ void(?: __launch_bounds__\([^)]*\))? (\w+)\(", src)
    assert m, "no kernel entry in TileLang's source"
    entry = m.group(1)
    # TileLang's host stub packs the launch as grid, block, dynamic shared memory: the last int64
    # it stores is the dynamic shared memory (checked against 0.1.15's FA and GEMM stubs); a kernel
    # without dynamic shared memory packs none
    packed = re.findall(r"v_int64\) = \(\(int64_t\)(\d+)\)", k.get_host_source() or "")
    smem = int(packed[-1]) if packed and "extern __shared__" in src else 0
    # buffers by the prim_func's parameter names; the kernel takes them in its own order
    names = [func.buffer_map[p].name for p in func.params]
    params = _params(src, entry)
    order = buffers or names
    args = ", ".join(f"({t})b[{order.index(n)}]" for t, n in params)
    return {
        "src": src,
        "entry": entry,
        "smem": smem,
        "args": args,
        "params": [n for _, n in params],
    }


def _setup(entry: str, smem: int, tag: str = "") -> str:
    return (
        f"  const int smem{tag} = {smem};\n"
        f"  CK(cudaFuncSetAttribute({entry}, cudaFuncAttributeMaxDynamicSharedMemorySize, "
        f"smem{tag}));"
    )


def _wanted(name: str) -> bool:
    if not re.search(os.environ.get("ONLY", "."), name):
        return False
    i, n = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
    return sum(name.encode()) % n == i


def write(
    out: Path,
    name: str,
    func,
    sizes: list[int],
    grid: list[int],
    threads: int | tuple[int, int],
    info: dict,
    elem_bytes: list[int] | None = None,
    rot: int = 1,
    writes: tuple[int, ...] = (-1,),
):
    """`sizes` in elements of each buffer, `elem_bytes` per buffer (default 2, fp16); `writes` the
    indices of the buffers the kernel writes (default the last)."""
    if not _wanted(name):
        return
    c = _compiled(func)
    eb = elem_bytes or [2] * len(sizes)
    nbytes = [n * b for n, b in zip(sizes, eb, strict=True)]
    block = list(threads) if isinstance(threads, tuple) else [threads]
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "kernel.cu").write_text(c["src"])
    g = ", ".join(str(x) for x in grid)
    b = ", ".join(str(x) for x in block)
    (d / "harness.cu").write_text(
        HARNESS
        % {
            "includes": '#include "kernel.cu"',
            "sizes": ", ".join(str(s) for s in nbytes),
            "nbuf": len(sizes),
            "rot": rot,
            "setup": _setup(c["entry"], c["smem"]) + f"\n  dim3 grid({g}), block({b});",
            "launches": f"    {c['entry']}<<<grid, block, smem, st>>>({c['args']});",
            "name": name,
        }
    )
    nthreads = math.prod(block)
    launch = {
        "entry": c["entry"],
        "grid": grid,
        "threads": nthreads,
        "block": block,
        "smem": c["smem"],
        "rot": rot,
        "footprint_bytes": sum(nbytes),
        "write_bytes": sum(nbytes[i] for i in writes),
        **info,
    }
    (d / "launch.json").write_text(json.dumps(launch, indent=1) + "\n")
    print(
        f"generated {name}: entry {c['entry']}, grid {grid}, block {block}, smem {c['smem']}, rot {rot}"
    )


def write_sequence(out: Path, name: str, members: list[dict], buffers: dict[str, int], info: dict):
    """Several kernels launched in order, one input set per sequence, as a generic program form:
    `members` (func, grid, threads, reads, writes: buffer names), `buffers` name -> bytes. The
    harness times the whole sequence (kernel, b2b, graph as `write`; per sequence)."""
    if not _wanted(name):
        return
    order = list(buffers)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    fp = sum(buffers.values())
    rot = 1 if info["cache"] == "warm" else _rot(fp)
    includes, setup, launches, kernels = [], [], [], []
    for i, mem in enumerate(members):
        c = _compiled(mem["func"], order)
        (d / f"kernel{i}.cu").write_text(c["src"])
        includes.append(f'#include "kernel{i}.cu"')
        g, b = (", ".join(str(x) for x in v) for v in (mem["grid"], [mem["threads"]]))
        setup.append(_setup(c["entry"], c["smem"], str(i)))
        launches.append(f"    {c['entry']}<<<dim3({g}), dim3({b}), smem{i}, st>>>({c['args']});")
        kernels.append(
            {
                "source": f"kernel{i}.cu",
                "entry": c["entry"],
                "grid": mem["grid"],
                "threads": mem["threads"],
                "block": [mem["threads"]],
                "smem": c["smem"],
                "params": c["params"],
                "reads": mem["reads"],
                "writes": mem["writes"],
                "footprint_bytes": sum(buffers[n] for n in c["params"]),
                "write_bytes": sum(buffers[n] for n in mem["writes"]),
            }
        )
    (d / "harness.cu").write_text(
        HARNESS
        % {
            "includes": "\n".join(includes),
            "sizes": ", ".join(str(buffers[n]) for n in order),
            "nbuf": len(order),
            "rot": rot,
            "setup": "\n".join(setup),
            "launches": "\n".join(launches),
            "name": name,
        }
    )
    launch = {
        "entry": kernels[0]["entry"],
        "kernels": kernels,
        "buffers": buffers,
        "rot": rot,
        "footprint_bytes": fp,
        **info,
    }
    (d / "launch.json").write_text(json.dumps(launch, indent=1) + "\n")
    print(f"generated sequence {name}: {[k['entry'] for k in kernels]}, rot {rot}")


def main(out: Path) -> None:
    M, N, K = SHAPE["M"], SHAPE["N"], SHAPE["K"]
    for name, (bm, bn, bk, s, th, panel) in HGEMM.items():
        h = {
            "block_M": bm,
            "block_N": bn,
            "block_K": bk,
            "num_stages": s,
            "threads": th,
            "swizzle": panel,
        }
        grid = [-(-N // bn), -(-M // bm), 1]
        # round 4: each variant also rotating (cold operands; the replay's DRAM rule without the
        # warm L2 question)
        for cache in ("warm", "rotating"):
            info = {
                "family": "hgemm",
                "shape": SHAPE,
                "pick": h,
                "smem_estimate": s * (bm + bn) * bk * 2,
                "cache": cache,
            }
            write(
                out,
                f"hgemm2048_{name}" + ("" if cache == "warm" else "_rotating"),
                hgemm(bm, bn, bk, s, th, panel),
                [M * K, K * N, M * N],
                grid,
                th,
                info,
                rot=1 if cache == "warm" else _rot((M * K + K * N + M * N) * 2),
            )
    f = FA
    n = f["B"] * f["S"] * f["H"] * f["D"]
    info = {
        "family": "fa_forward",
        "shape": {k: f[k] for k in ("B", "S", "H", "D")},
        "pick": {k: f[k] for k in ("BM", "BN", "stages", "threads")},
        "smem_estimate": (2 * f["BM"] + 2 * f["BN"]) * f["D"] * 2,
        "cache": "warm",
    }
    grid = [-(-f["S"] // f["BM"]), f["H"], f["B"]]
    write(out, "fa_fwd_s2048_h8_d128", fa_forward(**f), [n, n, n, n], grid, f["threads"], info)
    for name, gen in (
        ("decode", _decode),
        ("gemv", _gemv),
        ("rows", _rows),
        ("fp8", _fp8),
        ("moe", _moe),
        ("w4gemm", _w4gemm),
        ("decode_seq", _decode_seq),
        ("blind", _blind),
        ("whatif", _whatif),
        ("blind7", _blind7),
    ):
        try:  # a kernel that fails to generate is reported; the others stand
            gen(out)
        except Exception:  # noqa: BLE001 - reported, the batch keeps the other kernels
            print(f"FAILED {name}")
            traceback.print_exc()


def _rot(footprint: int) -> int:
    """Input sets for the rotating case: at least 64 MB in turn (rot x footprint > the L2)."""
    return max(2, math.ceil(64 * 2**20 / footprint))


def _decode(out: Path) -> None:
    c = DECODE
    B, H, G, S, D, sp = c["B"], c["H"], c["G"], c["S"], c["D"], c["split"]
    kv = B * S * G * D
    sizes = [B * H * D, kv, kv, B * H * sp, B * H * sp * D]
    vh = min(c["BH"], H // G)
    shape = {k: c[k] for k in ("B", "H", "G", "S", "D")}
    pick = {k: c[k] for k in ("BN", "BH", "split", "stages")}
    for cache in ("warm", "rotating"):
        rot = 1 if cache == "warm" else _rot(sum(sizes) * 2)
        info = {"family": "decode_gqa", "shape": shape, "pick": pick, "cache": cache}
        write(
            out,
            f"decode_gqa_part_s{S}_{cache}",
            decode_partial(**c),
            sizes,
            [B, H // vh, sp],
            128,
            info,
            rot=rot,
            writes=(3, 4),
        )
    info = {"family": "decode_gqa_combine", "shape": shape, "pick": pick, "cache": "warm"}
    write(
        out,
        f"decode_gqa_comb_s{S}_warm",
        decode_combine(**c),
        [B * H * sp, B * H * sp * D, B * H * D],
        [H, B],
        D,
        info,
    )


def _decode_seq(out: Path) -> None:
    """Split-KV decode as it runs: the partial kernel, then the combine over its splits."""
    c = DECODE
    B, H, G, S, D, sp = c["B"], c["H"], c["G"], c["S"], c["D"], c["split"]
    vh = min(c["BH"], H // G)
    kv = B * S * G * D * 2
    buffers = {
        "Q": B * H * D * 2,
        "K": kv,
        "V": kv,
        "glse": B * H * sp * 2,
        "Opart": B * H * sp * D * 2,
        "Out": B * H * D * 2,
    }
    for cache in ("warm", "rotating"):
        members = [
            {
                "func": decode_partial(**c),
                "grid": [B, H // vh, sp],
                "threads": 128,
                "reads": ["Q", "K", "V"],
                "writes": ["glse", "Opart"],
            },
            {
                "func": decode_combine(**c),
                "grid": [H, B],
                "threads": D,
                "reads": ["glse", "Opart"],
                "writes": ["Out"],
            },
        ]
        info = {
            "family": "sequence",
            "set": "decode_gqa_seq",
            "shape": {k: c[k] for k in ("B", "H", "G", "S", "D")},
            "cache": cache,
        }
        write_sequence(out, f"decode_seq_s{S}_{cache}", members, buffers, info)


def _blind(out: Path) -> None:
    """Round 6: rows no engine change has seen."""
    for (M, N, K), variants in HGEMM_BLIND.items():
        shape = {"M": M, "N": N, "K": K}
        for name, (bm, bn, bk, s, th, panel) in variants.items():
            h = {
                "block_M": bm,
                "block_N": bn,
                "block_K": bk,
                "num_stages": s,
                "threads": th,
                "swizzle": panel,
            }
            for cache in ("warm", "rotating"):
                info = {
                    "family": "hgemm",
                    "shape": shape,
                    "pick": h,
                    "smem_estimate": s * (bm + bn) * bk * 2,
                    "cache": cache,
                    "set": f"hgemm {M}x{N}x{K}" + (" rotating" if cache == "rotating" else ""),
                }
                fp = (M * K + K * N + M * N) * 2
                write(
                    out,
                    f"hgemm_m{M}n{N}k{K}_{name}" + ("" if cache == "warm" else "_rotating"),
                    hgemm(bm, bn, bk, s, th, panel, shape),
                    [M * K, K * N, M * N],
                    [-(-N // bn), -(-M // bm), 1],
                    th,
                    info,
                    rot=1 if cache == "warm" else _rot(fp),
                )
    f = FA_BLIND
    n = f["B"] * f["S"] * f["H"] * f["D"]
    info = {
        "family": "fa_forward",
        "shape": {k: f[k] for k in ("B", "S", "H", "D")},
        "pick": {k: f[k] for k in ("BM", "BN", "stages", "threads")},
        "smem_estimate": (2 * f["BM"] + 2 * f["BN"]) * f["D"] * 2,
        "cache": "warm",
    }
    write(
        out,
        f"fa_fwd_s{f['S']}_h{f['H']}_d{f['D']}",
        fa_forward(**f),
        [n, n, n, n],
        [-(-f["S"] // f["BM"]), f["H"], f["B"]],
        f["threads"],
        info,
    )
    M, N = RMS_BLIND["M"], RMS_BLIND["N"]
    for cache in ("warm", "rotating"):
        rot = 1 if cache == "warm" else _rot((2 * M * N + N) * 2)
        info = {"family": "rmsnorm", "shape": RMS_BLIND, "pick": {"threads": 128}, "cache": cache}
        write(
            out,
            f"rmsnorm_m{M}_n{N}_{cache}",
            rmsnorm(M, N),
            [M * N, N, M * N],
            [M],
            128,
            info,
            rot=rot,
        )
    M, N = SOFTMAX_BLIND["M"], SOFTMAX_BLIND["N"]
    info = {"family": "softmax", "shape": SOFTMAX_BLIND, "pick": {"threads": 128}, "cache": "warm"}
    write(out, f"softmax_m{M}_n{N}_warm", softmax(M, N), [M * N, M * N], [M], 128, info)


def _wi(name: str, base: str, axis: str) -> dict:
    return {} if axis == "base" else {"whatif": {"base": base, "axis": axis}}


def _whatif(out: Path) -> None:
    """Round 7: one change at a time from a base kernel, each measured and replayed from its own
    compiled code; the base is in the same batch."""
    for table, tag in ((WI_HGEMM, "b"), (WI_HGEMM2, "L_s3"), (WI_HGEMM3, "S_s4")):
        for cache in ("warm", "rotating"):
            for v, (bm, bn, bk, s, th, panel, opt, axis) in table.items():
                sh = SHAPE
                h = {
                    "block_M": bm,
                    "block_N": bn,
                    "block_K": bk,
                    "num_stages": s,
                    "threads": th,
                    "swizzle": panel,
                }
                info = {
                    "family": "hgemm",
                    "shape": sh,
                    "pick": h,
                    "smem_estimate": s * (bm + bn) * bk * 2,
                    "cache": cache,
                    "set": f"whatif hgemm {cache}",
                    **_wi(v, f"wi_hg_{tag}_{cache}", axis),
                }
                M, N, K = sh["M"], sh["N"], sh["K"]
                fp = (M * K + K * N + M * N) * 2
                write(
                    out,
                    f"wi_hg_{v}_{cache}",
                    hgemm(bm, bn, bk, s, th, panel, sh, **opt),
                    [M * K, K * N, M * N],
                    [-(-N // bn), -(-M // bm), 1],
                    th,
                    info,
                    rot=1 if cache == "warm" else _rot(fp),
                )
    for v, (kw, axis) in WI_FA.items():
        kw = dict(kw)
        base = kw.pop("base", "b")
        f = {**FA, **kw}
        n = f["B"] * f["S"] * f["H"] * f["D"]
        for cache in ("warm", "rotating"):
            if cache == "rotating" and v not in ("b", "s1", "s3", "nostore"):
                continue
            info = {
                "family": "fa_forward",
                "shape": {k: f[k] for k in ("B", "S", "H", "D")},
                "pick": {k: f[k] for k in ("BM", "BN", "stages", "threads")},
                "cache": cache,
                "set": f"whatif fa {cache}",
                **_wi(v, f"wi_fa_{base}_{cache}", axis),
            }
            write(
                out,
                f"wi_fa_{v}_{cache}",
                fa_forward(**f),
                [n, n, n, n],
                [-(-f["S"] // f["BM"]), f["H"], f["B"]],
                f["threads"],
                info,
                rot=1 if cache == "warm" else _rot(4 * n * 2),
                writes=() if kw.get("store") else (-1,),
            )
    for v, (kw, axis) in WI_GEMV.items():
        c = {**GEMV, **kw}
        N, K, g = c["N"], c["K"], c["group"]
        sizes, eb = [K, N * K // 2, N * K // g, N], [2, 1, 2, 2]
        fp = sum(s * b for s, b in zip(sizes, eb, strict=True))
        for cache in ("warm", "rotating"):
            info = {
                "family": "gemv_w4a16",
                "shape": {"N": N, "K": K},
                "pick": c,
                "cache": cache,
                "set": f"whatif gemv {cache}",
                **_wi(v, f"wi_gemv_b_{cache}", axis),
            }
            write(
                out,
                f"wi_gemv_{v}_{cache}",
                gemv_w4a16(**c),
                sizes,
                [-(-N // c["n_part"]), 1],
                (c["reduce"], c["n_part"]),
                info,
                elem_bytes=eb,
                rot=1 if cache == "warm" else _rot(fp),
            )
    M, N = ROWS["M"], ROWS["N"]
    for v, (kw, axis) in WI_RMS.items():
        eb = 4 if kw.get("dtype") == "float" else 2
        for cache in ("warm", "rotating"):
            info = {
                "family": "rmsnorm",
                "shape": ROWS,
                "pick": {"threads": kw.get("threads", 128)},
                "cache": cache,
                "set": f"whatif rmsnorm {cache}",
                **_wi(v, f"wi_rms_b_{cache}", axis),
            }
            write(
                out,
                f"wi_rms_{v}_{cache}",
                rmsnorm(M, N, **kw),
                [M * N, N, M * N],
                [M],
                kw.get("threads", 128),
                info,
                elem_bytes=[eb, eb, eb],
                rot=1 if cache == "warm" else _rot((2 * M * N + N) * eb),
                writes=() if kw.get("store") else (-1,),
            )
    for v, (kw, axis) in WI_SOFTMAX.items():
        info = {
            "family": "softmax",
            "shape": ROWS,
            "pick": {"threads": kw.get("threads", 128)},
            "cache": "warm",
            "set": "whatif softmax warm",
            **_wi(v, "wi_softmax_b_warm", axis),
        }
        write(
            out,
            f"wi_softmax_{v}_warm",
            softmax(M, N, **kw),
            [M * N, M * N],
            [M],
            kw.get("threads", 128),
            info,
        )
    for v, (kw, axis) in WI_DECODE.items():
        c = {**DECODE, **kw}
        B, H, G, S, D, sp = c["B"], c["H"], c["G"], c["S"], c["D"], c["split"]
        kv = B * S * G * D
        sizes = [B * H * D, kv, kv, B * H * sp, B * H * sp * D]
        vh = min(c["BH"], H // G)
        for cache in ("warm", "rotating"):
            info = {
                "family": "decode_gqa",
                "shape": {k: c[k] for k in ("B", "H", "G", "S", "D")},
                "pick": {k: c[k] for k in ("BN", "BH", "split", "stages")},
                "cache": cache,
                "set": f"whatif decode {cache}",
                **_wi(v, f"wi_dec_b_{cache}", axis),
            }
            write(
                out,
                f"wi_dec_{v}_{cache}",
                decode_partial(**c),
                sizes,
                [B, H // vh, sp],
                128,
                info,
                rot=1 if cache == "warm" else _rot(sum(sizes) * 2),
                writes=(3, 4),
            )
    c = DECODE
    B, H, G, S, D, sp = c["B"], c["H"], c["G"], c["S"], c["D"], c["split"]
    kv = B * S * G * D
    sizes = [B * H * D, kv, kv, B * H * sp, B * H * sp * D]
    r = ROWS["M"], ROWS["N"]
    for x in WI_ROT:
        info = {
            "family": "decode_gqa",
            "shape": {k: c[k] for k in ("B", "H", "G", "S", "D")},
            "pick": {k: c[k] for k in ("BN", "BH", "split", "stages")},
            "cache": "rotating",
            "set": "whatif decode rotating",
            **_wi("", "wi_dec_b_rotating", "rotation depth"),
        }
        write(
            out,
            f"wi_dec_rot{x}_rotating",
            decode_partial(**c),
            sizes,
            [B, H // min(c["BH"], H // G), sp],
            128,
            info,
            rot=x * _rot(sum(sizes) * 2),
            writes=(3, 4),
        )
        info = {
            "family": "rmsnorm",
            "shape": ROWS,
            "pick": {"threads": 128},
            "cache": "rotating",
            "set": "whatif rmsnorm rotating",
            **_wi("", "wi_rms_b_rotating", "rotation depth"),
        }
        write(
            out,
            f"wi_rms_rot{x}_rotating",
            rmsnorm(*r),
            [r[0] * r[1], r[1], r[0] * r[1]],
            [r[0]],
            128,
            info,
            elem_bytes=[2, 2, 2],
            rot=x * _rot((2 * r[0] * r[1] + r[1]) * 2),
        )
    MM, NN, KK = SHAPE["M"], SHAPE["N"], SHAPE["K"]
    bm, bn, bk, s, th = W4GEMM["c128x128x32_s3_t128"]
    for v, (kw, axis) in WI_W4.items():
        h = {"block_M": bm, "block_N": bn, "block_K": bk, "num_stages": s, "threads": th}
        info = {
            "family": "w4gemm",
            "shape": SHAPE,
            "pick": h,
            "cache": "warm",
            "set": "whatif w4gemm warm",
            **_wi(v, "wi_w4_b_warm", axis),
        }
        write(
            out,
            f"wi_w4_{v}_warm",
            w4gemm(bm, bn, bk, s, th, W4GEMM_GROUP, **kw),
            [MM * KK, NN * KK // 2, NN * KK // W4GEMM_GROUP, MM * NN],
            [-(-NN // bn), -(-MM // bm), 1],
            th,
            info,
            elem_bytes=[2, 1, 2, 2],
        )


def _blind7(out: Path) -> None:
    """Round 7: larger variant groups at a shape no engine has seen, for a blind Kendall tau."""
    sh = HGEMM_BLIND7_SHAPE
    M, N, K = sh["M"], sh["N"], sh["K"]
    fp = (M * K + K * N + M * N) * 2
    for name, (bm, bn, bk, s, th, panel) in HGEMM_BLIND7.items():
        h = {
            "block_M": bm,
            "block_N": bn,
            "block_K": bk,
            "num_stages": s,
            "threads": th,
            "swizzle": panel,
        }
        for cache in ("warm", "rotating"):
            info = {
                "family": "hgemm",
                "shape": sh,
                "pick": h,
                "smem_estimate": s * (bm + bn) * bk * 2,
                "cache": cache,
                "set": f"hgemm {M}x{N}x{K}" + (" rotating" if cache == "rotating" else ""),
            }
            write(
                out,
                f"hgemm_m{M}n{N}k{K}_{name}" + ("" if cache == "warm" else "_rotating"),
                hgemm(bm, bn, bk, s, th, panel, sh),
                [M * K, K * N, M * N],
                [-(-N // bn), -(-M // bm), 1],
                th,
                info,
                rot=1 if cache == "warm" else _rot(fp),
            )
    b = FA_BLIND7
    n = b["B"] * b["S"] * b["H"] * b["D"]
    for name, (bm, bn, s, th) in FA_BLIND7_V.items():
        f = {**b, "BM": bm, "BN": bn, "stages": s, "threads": th}
        info = {
            "family": "fa_forward",
            "shape": b,
            "pick": {k: f[k] for k in ("BM", "BN", "stages", "threads")},
            "cache": "warm",
            "set": f"fa {b['S']}x{b['H']}x{b['D']}",
        }
        try:
            write(
                out,
                f"fa_s{b['S']}_h{b['H']}_d{b['D']}_{name}",
                fa_forward(**f),
                [n, n, n, n],
                [-(-b["S"] // bm), b["H"], b["B"]],
                th,
                info,
            )
        except Exception:  # noqa: BLE001 - a tile TileLang refuses leaves the group
            print(f"FAILED fa {name}")


def _gemv(out: Path) -> None:
    c = GEMV
    N, K, g = c["N"], c["K"], c["group"]
    sizes, eb = [K, N * K // 2, N * K // g, N], [2, 1, 2, 2]
    for cache in ("warm", "rotating"):
        rot = 1 if cache == "warm" else _rot(sum(s * b for s, b in zip(sizes, eb, strict=True)))
        info = {"family": "gemv_w4a16", "shape": {"N": N, "K": K}, "pick": c, "cache": cache}
        write(
            out,
            f"gemv_w4a16_n{N}_k{K}_{cache}",
            gemv_w4a16(**c),
            sizes,
            [-(-N // c["n_part"]), 1],
            (c["reduce"], c["n_part"]),
            info,
            elem_bytes=eb,
            rot=rot,
        )


def _rows(out: Path) -> None:
    M, N = ROWS["M"], ROWS["N"]
    for cache in ("warm", "rotating"):
        rot = 1 if cache == "warm" else _rot((2 * M * N + N) * 2)
        info = {"family": "rmsnorm", "shape": ROWS, "pick": {"threads": 128}, "cache": cache}
        write(
            out,
            f"rmsnorm_m{M}_n{N}_{cache}",
            rmsnorm(M, N),
            [M * N, N, M * N],
            [M],
            128,
            info,
            rot=rot,
        )
    info = {"family": "softmax", "shape": ROWS, "pick": {"threads": 128}, "cache": "warm"}
    write(out, f"softmax_m{M}_n{N}_warm", softmax(M, N), [M * N, M * N], [M], 128, info)


def _fp8(out: Path) -> None:
    M, N, K = SHAPE["M"], SHAPE["N"], SHAPE["K"]
    for name, (bm, bn, bk, s, th) in FP8.items():
        h = {"block_M": bm, "block_N": bn, "block_K": bk, "num_stages": s, "threads": th}
        info = {
            "family": "fp8gemm",
            "shape": SHAPE,
            "pick": h,
            "elem_bytes": 1,
            "smem_estimate": s * (bm + bn) * bk,
            "cache": "warm",
        }
        write(
            out,
            f"fp8gemm2048_{name}",
            fp8gemm(bm, bn, bk, s, th),
            [M * K, N * K, M * N],
            [-(-N // bn), -(-M // bm), 1],
            th,
            info,
            elem_bytes=[1, 1, 2],
        )


def _moe(out: Path) -> None:
    c = MOE
    E, T_, N, K = c["E"], c["T"], c["N"], c["K"]
    for name, (bm, bn, bk, s, th) in MOE_TILES.items():
        h = {"block_M": bm, "block_N": bn, "block_K": bk, "num_stages": s, "threads": th}
        info = {"family": "moe_gemm", "shape": c, "pick": h, "cache": "warm"}
        write(
            out,
            f"moe_e{E}_t{T_}_n{N}_k{K}_{name}",
            moe_gemm(E, T_, N, K, bm, bn, bk, s, th),
            [T_ * K, E * K * N, T_ * N],
            [-(-N // bn), -(-T_ // bm), 1],
            th,
            info,
        )


def _w4gemm(out: Path) -> None:
    M, N, K = SHAPE["M"], SHAPE["N"], SHAPE["K"]
    group = W4GEMM_GROUP
    sizes = [M * K, N * K // 2, N * K // group, M * N]
    for name, (bm, bn, bk, s, th) in W4GEMM.items():
        h = {"block_M": bm, "block_N": bn, "block_K": bk, "num_stages": s, "threads": th}
        info = {"family": "w4gemm", "shape": SHAPE, "pick": h}
        fp = sum(x * b for x, b in zip(sizes, [2, 1, 2, 2], strict=True))
        for cache, rot in (("warm", 1), ("rotating", _rot(fp))):
            write(
                out,
                f"w4gemm2048_{name}_{cache}",
                w4gemm(bm, bn, bk, s, th, group),
                sizes,
                [-(-N // bn), -(-M // bm), 1],
                th,
                {**info, "cache": cache},
                elem_bytes=[2, 1, 2, 2],
                rot=rot,
            )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
