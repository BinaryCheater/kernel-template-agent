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
Round 13: design groups of GEMV, FA, decode, RMSNorm and softmax (blind, warm and rotating), own
kernels in `hand13/`. Round 14: the same at new shapes, `hand14/`, the third stage (`qs*`).
Round 15: blind at other divisors, `hand15/`, the DRAM limit (`dl_*`, `gd_*`, `rd_*`), `fad_*`,
`ct_*`.
Round 16: the fixed blind recipe (D874) at a factor 7 in every size, HGEMM a factor 25, `hand16/`;
the memory-hierarchy sweep of the coverage matrix (`_mem`).
Round 17: the same recipe at a factor 9 in every size, HGEMM a factor 27, `hand17/`; sweeps for
rules: `tw_*`, `gr_*`, `pw_*`, `px_*`, `rr_*`, `tv_*`, `fs_*`.
`ONLY` limits generation to the matching names; `SHARD=i/n` generates one share of them.
"""

import json
import math
import os
import random
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
# round 8: NVFP4 GEMM (28 §5), dequantised in the main loop, 16 elements per scale
NVFP4_BLOCK = 16
NVFP4 = {
    "c128x128x32_s3_t128": (128, 128, 32, 3, 128),
    "c128x128x32_s2_t128": (128, 128, 32, 2, 128),
    "c64x128x32_s4_t128": (64, 128, 32, 4, 128),
}
# round 8: what a row kernel's parts cost away from the chip's limits: one (48) and four (192)
# blocks per SM, one change at a time from the row at that size (question 27)
WI_ROWS_M = (48, 192)
WI_RMSB = {
    "b": ({}, "base"),
    "nostore": ({"store": "shared"}, "stores removed"),
    "nocompute": ({"compute": False}, "compute removed"),
    "noload": ({"load": False}, "loads removed"),
    "arith": ({"load": False, "store": "shared"}, "loads and stores removed"),
}
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
# round 9: a shape not on the scorecard, N twice M, the blocks' aspect the other way round
HGEMM_BLIND9_SHAPE = {"M": 1536, "N": 3072, "K": 2048}
HGEMM_BLIND9 = {k: v for k, v in HGEMM_BLIND7.items() if k != "c256x256x32_s2_t256"}
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
    load: bool = True,
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
            if load:
                T.copy(X[bx : bx + 1, :], x)
            else:  # round 8: the arithmetic alone
                for i, j in T.Parallel(1, N):
                    x[i, j] = T.Cast(dtype, j)
            if compute:
                for i, j in T.Parallel(1, N):
                    sq[i, j] = T.Cast("float", x[i, j]) * T.Cast("float", x[i, j])
                T.reduce_sum(sq, ss, dim=1)
                for i in T.Parallel(1):
                    ss[i] = T.rsqrt(ss[i] / N + 1e-6)
                for i, j in T.Parallel(1, N):
                    wj = T.Cast("float", W[j]) if load else T.Cast("float", x[i, j])
                    x[i, j] = T.Cast(dtype, T.Cast("float", x[i, j]) * ss[i] * wj)
            if store == "shared":
                T.copy(x, o_shared)
            else:
                T.copy(x, Y[bx : bx + 1, :])

    return rms


def nvfp4gemm(BM: int, BN: int, BK: int, S: int, threads: int, compute: bool = True):
    """GEMM with an NVFP4 B dequantised in the main loop (round 8): e2m1 elements, two per byte, one
    scale per 16 elements along K; sm_121a's ptxas refuses the block-scaled mma, so the elements and
    their scale are expanded to fp16 by the block's threads, then an fp16 mma (the w4gemm shape)."""
    M, N, K = SHAPE["M"], SHAPE["N"], SHAPE["K"]
    SC = BK // NVFP4_BLOCK

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float16"),
        Bq: T.Tensor((N, K // 2), "int8"),
        Sc: T.Tensor((N, K // NVFP4_BLOCK), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((BM, BK), "float16")
            Bq_shared = T.alloc_shared((BN, BK // 2), "int8")
            Sc_shared = T.alloc_shared((BN, SC), "float16")
            B_shared = T.alloc_shared((BN, BK), "float16")
            C_local = T.alloc_fragment((BM, BN), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=S):
                T.copy(A[by * BM, ko * BK], A_shared)
                T.copy(Bq[bx * BN, ko * (BK // 2)], Bq_shared)
                T.copy(Sc[bx * BN, ko * BK // NVFP4_BLOCK], Sc_shared)
                for i, j in T.Parallel(BN, BK):
                    x = (T.Cast("int32", Bq_shared[i, j // 2]) >> ((j % 2) * 4)) & 15
                    e, m = (x >> 1) & 3, x & 1
                    mag = T.if_then_else(e > 0, (2 + m) << (e - 1), m)
                    v = T.Cast("float16", mag) * T.float16(0.5)
                    B_shared[i, j] = T.if_then_else(x >= 8, -v, v) * Sc_shared[i, j // NVFP4_BLOCK]
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * BM, bx * BN])

    return gemm


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
  // round 14: reps 0 is the preflight: one launch, each buffer followed by a marked guard, read back
  const size_t GUARD = reps == 0 ? 65536 : 0;
  for (int r = 0; r < ROT; r++)
    for (int i = 0; i < %(nbuf)d; i++) {
      CK(cudaMalloc(&buf[r][i], bytes[i] + GUARD));
      if (GUARD) CK(cudaMemset((char*)buf[r][i] + bytes[i], 0xA5, GUARD));
      fill<<<1024, 256>>>((half_t*)buf[r][i], bytes[i] / 2, 0x9E3779B9u * (i + 1) + 77u * r);
    }
  CK(cudaDeviceSynchronize());
%(setup)s
  int next = 0;
  auto launch = [&](cudaStream_t st = 0) {
    void** b = buf[next++ %% ROT];
%(launches)s
  };
  if (reps == 0) {
    launch();
    CK(cudaDeviceSynchronize());
    std::vector<unsigned char> g(GUARD);
    for (int i = 0; i < %(nbuf)d; i++) {
      CK(cudaMemcpy(g.data(), (char*)buf[0][i] + bytes[i], GUARD, cudaMemcpyDeviceToHost));
      for (unsigned char x : g)
        if (x != 0xA5) { printf("PREFLIGHT left buffer %%d\n", i); return 3; }
    }
    printf("PREFLIGHT ok\n");
    return 0;
  }
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
    for p in filter(str.strip, m.group(1).split(",")):  # the arithmetic alone takes no buffer
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
    edit=None,
    smem: int | None = None,
):
    """`sizes` in elements of each buffer, `elem_bytes` per buffer (default 2, fp16); `writes` the
    indices of the buffers the kernel writes (default the last); round 14: `edit` a change to
    TileLang's source, `smem` the launch's dynamic shared memory instead of TileLang's."""
    if not _wanted(name):
        return
    c = _compiled(func)
    if edit:
        c["src"] = edit(c["src"])
    if smem:
        c["smem"] = smem
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


def _rot(footprint: int) -> int:
    """Input sets for the rotating case: at least 64 MB in turn (rot x footprint > the L2)."""
    return max(2, math.ceil(64 * 2**20 / footprint))


def _rots(cache: str, footprint: int, depth: int = 1) -> int:
    return 1 if cache == "warm" else depth * _rot(footprint)


def _pick(bm: int, bn: int, bk: int, s: int, th: int, **more) -> dict:
    return {"block_M": bm, "block_N": bn, "block_K": bk, "num_stages": s, "threads": th, **more}


def _tiles(M: int, N: int, bm: int, bn: int) -> list[int]:
    return [-(-N // bn), -(-M // bm), 1]


# one kernel each: `more` joins its launch.json
def _hgemm(
    out: Path, name: str, tile: tuple, sh: dict, cache: str, opt=None, edit=None, smem=None, **more
) -> None:
    bm, bn, bk, s, th, panel = tile
    M, N, K = sh["M"], sh["N"], sh["K"]
    info = {
        "family": "hgemm",
        "shape": sh,
        "pick": _pick(bm, bn, bk, s, th, swizzle=panel),
        "smem_estimate": s * (bm + bn) * bk * 2,
        "cache": cache,
        **more,
    }
    func = hgemm(bm, bn, bk, s, th, panel, sh, **(opt or {}))
    rot = _rots(cache, (M * K + K * N + M * N) * 2)
    sizes = [M * K, K * N, M * N]
    write(out, name, func, sizes, _tiles(M, N, bm, bn), th, info, rot=rot, edit=edit, smem=smem)


def _fa(out: Path, name: str, f: dict, cache="warm", estimate=False, writes=(-1,), **more) -> None:
    n = f["B"] * f["S"] * f["H"] * f["D"]
    size = {"smem_estimate": (2 * f["BM"] + 2 * f["BN"]) * f["D"] * 2} if estimate else {}
    info = {
        "family": "fa_forward",
        "shape": {k: f[k] for k in ("B", "S", "H", "D")},
        "pick": {k: f[k] for k in ("BM", "BN", "stages", "threads")},
        **size,
        "cache": cache,
        **more,
    }
    grid = [-(-f["S"] // f["BM"]), f["H"], f["B"]]
    rot = _rots(cache, 4 * n * 2)
    write(
        out, name, fa_forward(**f), [n, n, n, n], grid, f["threads"], info, rot=rot, writes=writes
    )


def _decode_part(out: Path, name: str, c: dict, cache: str, depth: int = 1, **more) -> None:
    B, H, G, S, D, sp = c["B"], c["H"], c["G"], c["S"], c["D"], c["split"]
    kv = B * S * G * D
    sizes = [B * H * D, kv, kv, B * H * sp, B * H * sp * D]
    info = {"family": "decode_gqa", **_decode_info(c), "cache": cache, **more}
    write(
        out,
        name,
        decode_partial(**c),
        sizes,
        [B, H // min(c["BH"], H // G), sp],
        128,
        info,
        rot=_rots(cache, sum(sizes) * 2, depth),
        writes=(3, 4),
    )


def _decode_info(c: dict) -> dict:
    return {
        "shape": {k: c[k] for k in ("B", "H", "G", "S", "D")},
        "pick": {k: c[k] for k in ("BN", "BH", "split", "stages")},
    }


def _gemv_row(out: Path, name: str, c: dict, cache: str, **more) -> None:
    N, K, g = c["N"], c["K"], c["group"]
    sizes, eb = [K, N * K // 2, N * K // g, N], [2, 1, 2, 2]
    info = {"family": "gemv_w4a16", "shape": {"N": N, "K": K}, "pick": c, "cache": cache, **more}
    write(
        out,
        name,
        gemv_w4a16(**c),
        sizes,
        [-(-N // c["n_part"]), 1],
        (c["reduce"], c["n_part"]),
        info,
        elem_bytes=eb,
        rot=_rots(cache, sum(s * b for s, b in zip(sizes, eb, strict=True))),
    )


def _rms(out: Path, name: str, sh: dict, cache: str, kw=None, depth: int = 1, **more) -> None:
    M, N = sh["M"], sh["N"]
    kw = kw or {}
    eb = 4 if kw.get("dtype") == "float" else 2
    th = kw.get("threads", 128)
    info = {"family": "rmsnorm", "shape": sh, "pick": {"threads": th}, "cache": cache, **more}
    write(
        out,
        name,
        rmsnorm(M, N, **kw),
        [M * N, N, M * N],
        [M],
        th,
        info,
        elem_bytes=[eb, eb, eb],
        rot=_rots(cache, (2 * M * N + N) * eb, depth),
        writes=() if kw.get("store") else (-1,),
    )


def _softmax(out: Path, name: str, sh: dict, kw=None, **more) -> None:
    M, N = sh["M"], sh["N"]
    kw = kw or {}
    th = kw.get("threads", 128)
    info = {"family": "softmax", "shape": sh, "pick": {"threads": th}, "cache": "warm", **more}
    write(out, name, softmax(M, N, **kw), [M * N, M * N], [M], th, info)


def _w4(out: Path, name: str, tile: tuple, cache: str, kw=None, **more) -> None:
    bm, bn, bk, s, th = tile
    M, N, K = SHAPE["M"], SHAPE["N"], SHAPE["K"]
    g = W4GEMM_GROUP
    sizes, eb = [M * K, N * K // 2, N * K // g, M * N], [2, 1, 2, 2]
    write(
        out,
        name,
        w4gemm(bm, bn, bk, s, th, g, **(kw or {})),
        sizes,
        _tiles(M, N, bm, bn),
        th,
        {
            "family": "w4gemm",
            "shape": SHAPE,
            "pick": _pick(bm, bn, bk, s, th),
            "cache": cache,
            **more,
        },
        elem_bytes=eb,
        rot=_rots(cache, sum(x * b for x, b in zip(sizes, eb, strict=True))),
    )


# round 11. Reads and stores past the L2: r reads and w stores per thread pass over E MiB of
# elements (LOOP §3, hand_triad, hand_stencil): how the DRAM's read and write time combine, and
# whether written bytes churn a warm read.
MB = 1 << 20
WR_POINTS = (
    (1, 0, 8),
    (1, 0, 16),
    (1, 0, 32),
    (1, 0, 64),
    (1, 1, 4),
    (1, 1, 5),
    (1, 1, 6),
    (1, 1, 7),
    (1, 1, 8),
    (1, 1, 10),
    (1, 1, 12),
    (1, 1, 16),
    (1, 1, 32),
    (1, 1, 64),
    (2, 1, 8),
    (2, 1, 16),
    (2, 1, 32),
    (1, 2, 8),
    (1, 2, 16),
    (1, 2, 32),
    (1, 4, 4),
    (1, 4, 8),
    (1, 4, 16),
)
# question 26: what a rotation of `rot` sets keeps, plain loads alone and with stores
WR_ROTATING = ((1, 0, 8, 2), (1, 0, 8, 4), (1, 0, 8, 8), (1, 0, 16, 2), (1, 1, 8, 2), (1, 1, 8, 4))
ATOM_BINS = (1, 16, 256, 1024, 4096, 65536, 1048576)
# bins, bytes apart: same bins at a line and at a page's spacing
ATOM_SPREAD = ((16, 128), (256, 128), (16, 4096), (256, 4096))
# stores 1 element apart per thread: the elements between one pass's store and the next (32 lanes:
# 32 lines apart), with the first lane off a line (S 8: stress_09's pattern) or on it
REWRITE = ((32, 0), (16, 0), (8, 0), (4, 0), (1, 0), (32, 3), (8, 3))
# blind rows for the mechanisms above: predictions committed, no parameter taken from them
HAND11 = {
    "gups": ((1024, 1, 1), (256,), [4 << 20, 16 << 20]),
    "copy": ((512, 1, 1), (256,), [32 << 20, 32 << 20]),
    "scale": ((1024, 1, 1), (256,), [64 << 20, 64 << 20]),
    "stencil512": ((32, 32, 1), (16, 16), [1 << 20, 1 << 20]),
    "smear": ((192, 1, 1), (256,), [1 << 20, 1 << 20]),
}


def _round11(out: Path) -> None:
    for r, w, e, rot in (*((*p, 1) for p in WR_POINTS), *WR_ROTATING):
        reads = "\n".join(f"    s += b[i * {r} + {j}];" for j in range(r))
        writes = "\n".join(f"    a[i * {w} + {j}] = s;" for j in range(w))
        tail = "" if w else "  if (s == 1.2345f) a[0] = s;\n"
        src = f"""// round 11: {r} reads and {w} stores per pass over {e} MiB of elements
__global__ void __launch_bounds__(256) k_wr(float* a, const float* b) {{
  const size_t base = (size_t)blockIdx.x * 256 * 64 + threadIdx.x;
  float s = 0.f;
  for (int k = 0; k < 64; ++k) {{
    size_t i = base + (size_t)k * 256;
{reads}
{writes}
  }}
{tail}}}
"""
        grid = (e * 16, 1, 1)
        sizes = [max(4096, w * e * MB), r * e * MB]
        tag = f"_rot{rot}" if rot > 1 else ""
        name = f"wr_r{r}w{w}_{e}m{tag}"
        group = "wr_rotating" if rot > 1 else "wr_past_l2"
        _source_kernel(out, name, src, grid, (256,), sizes, "hand", group, rot)
    for bins in ATOM_BINS:
        src = f"""// round 11: 16 atomics per thread on {bins} bins
__global__ void __launch_bounds__(256) k_atom(unsigned* h) {{
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (int k = 0; k < 16; ++k)
    atomicAdd(&h[((t + k * 1048576u) * 2654435761u >> 11) & {bins - 1}u], 1u);
}}
"""
        name = f"atom_b{bins}"
        _source_kernel(out, name, src, (1024, 1, 1), (256,), [bins * 4], "hand", "atomic_bins")
    for bins, gap in ATOM_SPREAD:
        src = f"""// round 11: 16 atomics per thread on {bins} bins {gap} bytes apart
__global__ void __launch_bounds__(256) k_atom(unsigned* h) {{
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (int k = 0; k < 16; ++k)
    atomicAdd(&h[(((t + k * 1048576u) * 2654435761u >> 11) & {bins - 1}u) * {gap // 4}u], 1u);
}}
"""
        name = f"atom_b{bins}_gap{gap}"
        _source_kernel(out, name, src, (1024, 1, 1), (256,), [bins * gap], "hand", "atomic_bins")
    for g in (48, 768):
        for sp, off in REWRITE:
            src = f"""// round 11: stores {sp} elements apart per thread pass, first lane {off} off a line
__global__ void __launch_bounds__(256) k_rw(unsigned* out) {{
  const size_t base = (size_t)blockIdx.x * 4096 + threadIdx.x + {off};
  for (int k = 0; k < 16; ++k)
    out[base + (size_t)k * {sp}] = k;
}}
"""
            sizes = [g * 16384 + 4096]
            name = f"rw_s{sp}_o{off}_g{g}"
            written = g * (256 + 15 * sp) * 4
            _source_kernel(
                out, name, src, (g, 1, 1), (256,), sizes, "hand", "store_rewrite", written=written
            )
    src_dir = Path(__file__).parent / "hand"
    for name, (grid, block, sizes) in HAND11.items():
        text = (src_dir / f"{name}.cu").read_text()
        _source_kernel(out, f"hand_{name}", text, grid, block, sizes, "hand", "breadth hand")


# round 12, blind rows: predictions committed before the run, no parameter taken from them. name:
# (grid, block, [buffer bytes], bytes written or None); the first buffer is the written or atomic one
HAND12 = {
    "memset": ((1024, 1, 1), (256,), [64 * MB], None),
    "revcopy": ((512, 1, 1), (256,), [32 * MB, 32 * MB], None),
    "gather": ((128, 1, 1), (256,), [8 * MB, 32 * MB], None),
    "scatter": ((512, 1, 1), (256,), [32 * MB], None),
    "aos": ((128, 1, 1), (256,), [34 * MB, 2 * MB], 16 * MB),
    "rewrite": ((1024, 1, 1), (256,), [5 * MB], None),
    "atomblock": ((1024, 1, 1), (256,), [4096], 256),
    "atomline": ((1024, 1, 1), (256,), [2 * MB], None),
    "atomret": ((1024, 1, 1), (256,), [4096, 1 * MB], None),
    "atommax": ((1024, 1, 1), (256,), [16 << 10], None),
    "atom64": ((1024, 1, 1), (256,), [64 << 10], None),
    "mixrwa": ((512, 1, 1), (256,), [16 << 10, 8 * MB, 8 * MB], None),
    "rankpush": ((1024, 1, 1), (256,), [1 * MB, 4 * MB], None),
    "tally2d": ((64, 64, 1), (16, 16), [1024, 16 * MB, 4 * MB], None),
    "jacobi3d": ((8192, 1, 1), (256,), [8 * MB, 8 * MB], None),
    "axpy": ((1024, 1, 1), (256,), [32 * MB, 32 * MB], None),
    "fanout": ((512, 1, 1), (256,), [5 * MB, MB // 2], None),
    "csr": ((4096, 1, 1), (256,), [4 * MB, 8 * MB, 16 * MB], None),
    "maskstore": ((512, 1, 1), (256,), [16 * MB, 16 * MB], None),
}
T128 = (("256", "128"),)  # threads halved, grid doubled, the same bytes
# one change from a kernel above, by text: base, tag, axis, grid, buffers, written, replacements
WI_HAND12 = (
    ("memset", "m8", "shape", (256, 1, 1), [16 * MB], None, ()),
    ("memset", "t128", "threads", (2048, 1, 1), [64 * MB], None, T128),
    ("revcopy", "t128", "threads", (1024, 1, 1), [32 * MB, 32 * MB], None, T128),
    ("gather", "m4", "shape", None, [8 * MB, 4 * MB], None, (("8388607u", "1048575u"),)),
    ("scatter", "m4", "shape", None, [4 * MB], None, (("8388607u", "1048575u"),)),
    (
        "aos",
        "f1",
        "stores removed",
        None,
        None,
        8 * MB,
        (("a[i * 16 + 1]", "//"), ("a[i * 16 + 2]", "//")),
    ),
    ("rewrite", "j1", "stores removed", None, None, None, (("j < 4", "j < 1"),)),
    ("atomline", "w8", "shape", None, [512 << 10], None, (("* 32u", "* 8u"),)),
    ("atommax", "b64", "shape", None, [256], None, (("4095u", "63u"),)),
    (
        "mixrwa",
        "noatom",
        "compute removed",
        None,
        None,
        None,
        (("atomicAdd(&h[(v ^ (v >> 7)) & 4095u], 1u);", ""),),
    ),
    ("mixrwa", "nostore", "stores removed", None, None, None, (("a[i] = v ^ k;", ""),)),
    (
        "mixrwa",
        "noload",
        "loads removed",
        None,
        None,
        None,
        (("unsigned v = b[i];", "unsigned v = (unsigned)i * 2654435761u;"),),
    ),
    ("mixrwa", "t128", "threads", (1024, 1, 1), None, None, T128),
    ("axpy", "m8", "shape", (256, 1, 1), [8 * MB, 8 * MB], None, ()),
    ("axpy", "t128", "threads", (2048, 1, 1), None, None, T128),
    ("fanout", "s16", "shape", None, None, None, (("* 1024", "* 8"),)),
    ("maskstore", "all", "stores removed", None, None, None, (("if (v & 256u) ", ""),)),
)
# question 29, D800: stack rows at new thread counts, words and resident blocks
STACK12 = (
    *((t, n, g) for t in (64, 128, 256) for n in (64, 256) for g in (48, 144)),
    (64, 1024, 96),
    (128, 1024, 96),
)
HGEMM_BLIND12_SHAPE = {"M": 2560, "N": 2048, "K": 1536}
HGEMM_BLIND12 = {
    **HGEMM_BLIND7,
    "c128x128x32_s2_t128_sw8": (128, 128, 32, 2, 128, 8),
    "c128x128x48_s3_t128": (128, 128, 48, 3, 128, 0),
}
HG12_AXIS = {"s2": "stages", "s4": "stages", "sw8": "raster"}
WK12 = {  # rows behind the wrong what-ifs and t64 (2048 cube, warm): which part moves
    "bk8": ((128, 128, 8, 3, 128, 0), {}, "wi_hg_b_warm", "tile"),
    "bk16_nocompute": (
        (128, 128, 16, 3, 128, 0),
        {"compute": False},
        "wi_hg_bk16_warm",
        "compute removed",
    ),
    "bk16_nostore": (
        (128, 128, 16, 3, 128, 0),
        {"store": "shared"},
        "wi_hg_bk16_warm",
        "stores removed",
    ),
    "t64_nocompute": (
        (128, 128, 32, 3, 64, 0),
        {"compute": False},
        "wi_hg_t64_warm",
        "compute removed",
    ),
    "t64_nostore": (
        (128, 128, 32, 3, 64, 0),
        {"store": "shared"},
        "wi_hg_t64_warm",
        "stores removed",
    ),
    "sw8_nocompute": (
        (128, 128, 32, 3, 128, 8),
        {"compute": False},
        "wi_hg_sw8_warm",
        "compute removed",
    ),
    "sw8_nostore": (
        (128, 128, 32, 3, 128, 8),
        {"store": "shared"},
        "wi_hg_sw8_warm",
        "stores removed",
    ),
}
GEMV12 = {"N": 2048, "K": 8192, "group": 64, "n_part": 4, "reduce": 32}
ROWS12 = {"M": 256, "N": 8192}
DECODE12 = {**DECODE, "H": 16, "G": 4, "S": 2048, "D": 128}
FA12 = {**FA, "S": 1024, "H": 12, "D": 128}


# round 12, second batch: one sweep per miss of the blind rows, none of them a blind row itself
WRITE_FOOT = (1, 4, 8, 16)  # MiB written over and over, 64 MiB in all
RMW = (4, 6, 8, 10, 12, 16)  # MiB of y, y[i] op= x[i] beside as many of x
ATOM_LANES = (1, 2, 4, 8, 16, 32)  # lanes of a warp instruction on one word
ATOM_RET = (1, 16, 1024, 65536)
STACK_U = tuple((t, n, g) for t in (64, 256) for n in (256, 1024) for g in (48, 96, 144))


def _round12b(out: Path) -> None:
    for w in WRITE_FOOT:
        src = f"""// round 12b: 64 MiB of stores over {w} MiB
__global__ void __launch_bounds__(256) k_wf(float* a) {{
  const size_t n = {w * MB // 4}ull;
  const size_t base = (size_t)blockIdx.x * 256 * 64 + threadIdx.x;
  for (int k = 0; k < 64; ++k) a[(base + (size_t)k * 256) % n] = (float)k;
}}
"""
        _source_kernel(out, f"wf_m{w}", src, (1024, 1, 1), (256,), [w * MB], "hand", "wr_footprint")
    for e in RMW:
        n = e * MB // 4
        for name, body in (("rmw", "y[i] += 2.5f * x[i];"), ("wr", "y[i] = 2.5f * x[i];")):
            src = f"""// round 12b: {e} MiB, {body}
__global__ void __launch_bounds__(256) k_rmw(float* y, const float* x) {{
  const size_t base = (size_t)blockIdx.x * 256 * 32 + threadIdx.x;
  for (int k = 0; k < 32; ++k) {{
    size_t i = base + (size_t)k * 256;
    {body}
  }}
}}
"""
            g = n // (256 * 32)
            sizes = [e * MB, e * MB]
            _source_kernel(out, f"{name}_e{e}", src, (g, 1, 1), (256,), sizes, "hand", "wr_rmw")
    for lw in ATOM_LANES:
        for bins in (64, 1 << 20):
            src = f"""// round 12b: {lw} lanes on one word of {bins}
__global__ void __launch_bounds__(256) k_atw(unsigned* h) {{
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (int k = 0; k < 8; ++k)
    atomicAdd(&h[((t / {lw} + k * 4099u) * 2654435761u >> 7) & {bins - 1}u], 1u);
}}
"""
            name = f"atw_l{lw}_b{bins}"
            _source_kernel(out, name, src, (1024, 1, 1), (256,), [bins * 4], "hand", "atomic_bins")
    for bins in ATOM_RET:
        src = f"""// round 12b: fetch-add on {bins} bins, the old value used
__global__ void __launch_bounds__(256) k_atr(unsigned* h, unsigned* a) {{
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned s = 0;
  for (int k = 0; k < 8; ++k) s += atomicAdd(&h[((t + k * 4099u) * 2654435761u >> 7) & {bins - 1}u], k);
  a[t] = s;
}}
"""
        sizes = [bins * 4, 1 * MB]
        _source_kernel(out, f"atr_b{bins}", src, (1024, 1, 1), (256,), sizes, "hand", "atomic_bins")
    for t, n, g in STACK_U:
        src = f"""// round 12b: {n} words of stack per thread, one word per pass, all lanes alike
__global__ void __launch_bounds__({t}) k_stu(unsigned* sink, const unsigned* cfg) {{
  volatile unsigned a[{n}];
  unsigned s = threadIdx.x, z = cfg[0] & 0;
  for (int i = 0; i < {n}; i++) a[(i + z) & {n - 1}] = s;
  for (int r = 0; r < 4; r++)
    for (int i = 0; i < {n}; i++) {{
      a[(i * 5 + z) & {n - 1}] = s + i;
      s ^= a[(i * 3 + r + z) & {n - 1}];
    }}
  if (s == 12345) sink[0] = 1;
}}
"""
        name = f"stu_t{t}_n{n}_g{g}"
        _source_kernel(out, name, src, (g, 1, 1), (t,), [4096, 4096], "hand", "stack_path")


def _round12(out: Path) -> None:
    src_dir = Path(__file__).parent / "hand12"
    for name, (grid, block, sizes, written) in HAND12.items():
        text = (src_dir / f"{name}.cu").read_text()
        _source_kernel(
            out, f"bl_{name}", text, grid, block, sizes, "hand", "breadth blind", written=written
        )
    for name, tag, axis, grid, sizes, written, subs in WI_HAND12:
        g0, block, s0, w0 = HAND12[name]
        text = (src_dir / f"{name}.cu").read_text()
        for old, new in subs:
            if new == "//":  # a statement dropped
                text = re.sub(rf"\n[^\n]*{re.escape(old)}[^\n]*", "", text)
            else:
                text = text.replace(old, new)
        wi = {"whatif": {"base": f"bl_{name}", "axis": axis}}
        if subs is T128:
            block = (128,)
        _source_kernel(
            out,
            f"bl_{name}_{tag}",
            text,
            grid or g0,
            block,
            sizes or s0,
            "hand",
            "breadth blind",
            written=written if written is not None else w0,
            **wi,
        )
    for t, n, g in STACK12:
        src = f"""// round 12: {n} words of stack per thread, stored and loaded four passes
__global__ void __launch_bounds__({t}) k_stk(unsigned* sink) {{
  volatile unsigned a[{n}];
  unsigned s = threadIdx.x;
  for (int i = 0; i < {n}; i++) a[i] = s;
  for (int r = 0; r < 4; r++)
    for (int i = 0; i < {n}; i++) {{
      a[(i * 5 + threadIdx.x) & {n - 1}] = s + i;
      s ^= a[(i * 3 + r) & {n - 1}];
    }}
  if (s == 12345) sink[0] = 1;
}}
"""
        _source_kernel(
            out, f"stk_t{t}_n{n}_g{g}", src, (g, 1, 1), (t,), [4096], "hand", "stack_path"
        )
    sh = HGEMM_BLIND12_SHAPE
    M, N, K = sh["M"], sh["N"], sh["K"]
    pre = f"hgemm_m{M}n{N}k{K}"
    for name, tile in HGEMM_BLIND12.items():
        for cache, tail in (("warm", ""), ("rotating", " rotating")):
            suffix = "" if cache == "warm" else "_rotating"
            base = f"{pre}_c128x128x32_s3_t128{suffix}"
            more = (
                {}
                if f"{pre}_{name}{suffix}" == base
                else {"whatif": {"base": base, "axis": "design"}}
            )
            _hgemm(
                out,
                f"{pre}_{name}{suffix}",
                tile,
                sh,
                cache,
                set=f"hgemm {M}x{N}x{K}{tail}",
                **more,
            )
    base = f"{pre}_c128x128x32_s3_t128"
    for v, opt, axis, stages in (
        ("nocompute", {"compute": False}, "compute removed", 3),
        ("nostore", {"store": "shared"}, "stores removed", 3),
        ("noload", {"load": False}, "loads removed", 0),
    ):
        more = {"whatif": {"base": base, "axis": axis}}
        tile = (128, 128, 32, stages, 128, 0)
        _hgemm(out, f"{base}_{v}", tile, sh, "warm", opt, set="whatif hgemm warm", **more)
    for v, (tile, opt, base, axis) in WK12.items():
        more = {"whatif": {"base": base, "axis": axis}}
        _hgemm(out, f"wk_{v}", tile, SHAPE, "warm", opt, set="whatif hgemm warm", **more)
    for v, (kw, axis) in WI_GEMV.items():
        for cache in ("warm", "rotating"):
            more = _wi(v, f"wi12_gemv_b_{cache}", axis)
            group = f"whatif gemv {cache}"
            _gemv_row(out, f"wi12_gemv_{v}_{cache}", {**GEMV12, **kw}, cache, set=group, **more)
    for v, (kw, axis) in WI_RMS.items():
        for cache in ("warm", "rotating"):
            more = _wi(v, f"wi12_rms_b_{cache}", axis)
            _rms(
                out,
                f"wi12_rms_{v}_{cache}",
                ROWS12,
                cache,
                kw,
                set=f"whatif rmsnorm {cache}",
                **more,
            )
    for v, (kw, axis) in WI_SOFTMAX.items():
        more = _wi(v, "wi12_softmax_b_warm", axis)
        _softmax(out, f"wi12_softmax_{v}_warm", ROWS12, kw, set="whatif softmax warm", **more)
    for v in ("b", "s1", "s3", "split4", "split16", "bn32"):
        kw, axis = WI_DECODE[v]
        for cache in ("warm", "rotating"):
            more = _wi(v, f"wi12b_dec_b_{cache}", axis)
            group = f"whatif decode {cache}"
            _decode_part(
                out, f"wi12b_dec_{v}_{cache}", {**DECODE12, **kw}, cache, set=group, **more
            )
    for v in ("b", "s1", "bn32", "bm128", "t64", "nostore", "nov", "noexp"):
        kw, axis = dict(WI_FA[v][0]), WI_FA[v][1]
        more = _wi(v, "wi12_fa_b_warm", axis)
        writes = () if kw.get("store") else (-1,)
        f = {**FA12, **kw}
        _fa(out, f"wi12_fa_{v}_warm", f, "warm", False, writes, set="whatif fa warm", **more)


# round 13: a blind set off the HGEMM shapes: design groups of the other kernels, warm and rotating,
# one more HGEMM shape, ordinary code in the store, rewrite, atomic and hashed classes
GEMV13 = {"N": 3072, "K": 6144, "group": 64, "n_part": 4, "reduce": 32}
GEMV13_V = {
    "b": ({}, "base"),
    "np2": ({"n_part": 2}, "threads"),
    "np8": ({"n_part": 8}, "threads"),
    "r16": ({"reduce": 16}, "threads"),
    "r64": ({"reduce": 64}, "threads"),
    "mk4": ({"mk": 4}, "bytes per lane"),
    "mk16": ({"mk": 16}, "bytes per lane"),
    "np2_r64": ({"n_part": 2, "reduce": 64}, "design"),
    "np8_r16": ({"n_part": 8, "reduce": 16}, "design"),
}
FA13 = {**FA, "S": 1792, "H": 10, "D": 80}
FA13_AXIS = {
    "c64x64_s1_t128": "stages",
    "c64x64_s3_t128": "stages",
    "c64x32_s3_t128": "design",
    "c64x128_s2_t128": "tile",
    "c128x64_s2_t128": "tile",
    "c128x64_s2_t256": "design",
    "c128x128_s2_t256": "design",
}
DECODE13 = {**DECODE, "H": 24, "G": 6, "S": 3072, "D": 96}
DECODE13_V = {
    "b": ({}, "base"),
    "s1": ({"stages": 1}, "stages"),
    "s3": ({"stages": 3}, "stages"),
    "split4": ({"split": 4}, "blocks"),
    "split16": ({"split": 16}, "blocks"),
    "bn32": ({"BN": 32}, "tile"),
    "bn128": ({"BN": 128}, "tile"),
    "split4_bn32": ({"split": 4, "BN": 32}, "design"),
}
ROWS13 = {"M": 768, "N": 3072}
THREADS13 = (16, 32, 64, 128, 256, 512)  # TileLang's reduce wants powers of two
HGEMM_BLIND13_SHAPE = {"M": 1792, "N": 2560, "K": 2304}
HG13_AXIS = {"s2": "stages", "s4": "stages", "sw8": "raster"}
HG13_WI = {  # name: tile, options, axis, also rotating
    "s1": ((128, 128, 32, 1, 128, 0), None, "stages", True),
    "direct": ((128, 128, 32, 0, 128, 0), None, "cp.async vs direct", False),
    "bk16": ((128, 128, 16, 3, 128, 0), None, "tile", False),
    "t64": ((128, 128, 32, 3, 64, 0), None, "threads", True),
    "t256": ((128, 128, 32, 3, 256, 0), None, "threads", True),
    "nocompute": ((128, 128, 32, 3, 128, 0), {"compute": False}, "compute removed", False),
    "nostore": ((128, 128, 32, 3, 128, 0), {"store": "shared"}, "stores removed", False),
    "noload": ((128, 128, 32, 0, 128, 0), {"load": False}, "loads removed", False),
}
HAND13 = {
    "transpose": ((512, 1, 1), (256,), [4 * MB, 4 * MB], None),
    "interleave": ((512, 1, 1), (256,), [16 * MB, 8 * MB, 8 * MB], None),
    "tilefill": ((32, 32, 1), (16, 16), [16 * MB], None),
    "recordfill": ((1024, 1, 1), (256,), [16 * MB], None),
    "smooth": ((512, 1, 1), (256,), [4 * MB], None),
    "swaprows": ((512, 1, 1), (256,), [8 * MB], None),
    "blockcount": ((1024, 1, 1), (256,), [4 * MB, 4 * MB], None),
    "decay": ((768, 1, 1), (256,), [12 * MB], None),
    "wordcount": ((1024, 1, 1), (256,), [4096, 4 * MB], None),
    "blockmin": ((1024, 1, 1), (256,), [512, 4 * MB], None),
    "bitset": ((1024, 1, 1), (256,), [128 << 10], None),
    "ticket": ((1024, 1, 1), (256,), [4 * MB, 64], None),
    "casmax": ((1024, 1, 1), (256,), [256 << 10], None),
    "floatadd": ((1024, 1, 1), (256,), [16 << 10, 4 * MB], None),
    "probe": ((512, 1, 1), (256,), [4 * MB, 4 * MB, 16 * MB], None),
    "cuckoo": ((1024, 1, 1), (256,), [1 * MB, 32 * MB], None),
    "permute": ((512, 1, 1), (256,), [16 * MB, 8 * MB, 8 * MB], None),
    "bloomadd": ((1024, 1, 1), (256,), [16 * MB], None),
    "bucket": ((1024, 1, 1), (256,), [16 * MB, 4 * MB], None),
    "insort": ((512, 1, 1), (128,), [256 << 10, 4 * MB], None),
    "window": ((512, 1, 1), (256,), [512 << 10, 4 * MB], None),
}
T64 = (("128", "64"),)
LOADED13 = "(t + k) * 2654435761u"  # the same stream of values without the load
WI_HAND13 = (
    ("transpose", "t128", "threads", (1024, 1, 1), None, None, T128),
    ("interleave", "t128", "threads", (1024, 1, 1), None, None, T128),
    ("tilefill", "p2", "shape", None, None, None, (("i < 4", "i < 2"), ("j < 4", "j < 2"))),
    ("recordfill", "w4", "stores removed", None, None, None, (("k < 16", "k < 4"),)),
    ("smooth", "p1", "stores removed", None, None, None, (("p < 4", "p < 1"),)),
    ("swaprows", "l1", "loads removed", None, None, None, (("a[j];", "lo + 1u;"),)),
    (
        "blockcount",
        "nl",
        "loads removed",
        None,
        None,
        None,
        (("key[(t * 8 + k) & 1048575u]", LOADED13),),
    ),
    ("decay", "m4", "shape", (192, 1, 1), [3 * MB], None, ()),
    ("wordcount", "b64", "shape", None, [256, 4 * MB], None, (("1023u", "63u"),)),
    (
        "blockmin",
        "nl",
        "loads removed",
        None,
        None,
        None,
        (("a[(t * 8 + k) & 1048575u]", LOADED13),),
    ),
    ("bitset", "w8", "shape", None, [1 * MB], None, (("32767u", "262143u"),)),
    ("ticket", "noret", "stores removed", None, None, None, (("o[n & 1048575u] = t;", "//"),)),
    ("casmax", "b64", "shape", None, [256], None, (("65535u", "63u"),)),
    ("floatadd", "nl", "loads removed", None, None, None, (("x[i & 1048575u]", "(float)i"),)),
    ("probe", "m4", "shape", None, [4 * MB, 4 * MB, 1 * MB], None, (("4194303u", "262143u"),)),
    (
        "cuckoo",
        "one",
        "loads removed",
        None,
        None,
        None,
        (("unsigned b = table", "//"), ("a < b ? a : b", "a")),
    ),
    ("permute", "m4", "shape", None, [1 * MB, 8 * MB, 8 * MB], None, (("4194303u", "262143u"),)),
    ("bloomadd", "m1", "shape", None, [1 * MB], None, (("16777215u", "1048575u"),)),
    ("bucket", "nl", "loads removed", None, None, None, (("a[(t * 8 + k) & 1048575u]", LOADED13),)),
    ("insort", "t64", "threads", (1024, 1, 1), None, None, T64),
    (
        "insort",
        "w8",
        "shape",
        None,
        None,
        None,
        (("< 16", "< 8"), ("w[8];", "w[4];"), ("w[16]", "w[8]"), ("t * 16", "t * 8")),
    ),
    ("window", "t128", "threads", (1024, 1, 1), None, None, T128),
)


def _hand13(out: Path) -> None:
    src_dir = Path(__file__).parent / "hand13"
    for name, (grid, block, sizes, written) in HAND13.items():
        text = (src_dir / f"{name}.cu").read_text()
        _source_kernel(
            out, f"bl13_{name}", text, grid, block, sizes, "hand", "breadth blind", written=written
        )
    for name, tag, axis, grid, sizes, written, subs in WI_HAND13:
        g0, block, s0, w0 = HAND13[name]
        text = (src_dir / f"{name}.cu").read_text()
        for old, new in subs:
            if new == "//":  # a statement dropped
                text = re.sub(rf"\n[^\n]*{re.escape(old)}[^\n]*", "", text)
            else:
                text = text.replace(old, new)
        if subs in (T128, T64):
            block = (int(subs[0][1]),)
        _source_kernel(
            out,
            f"bl13_{name}_{tag}",
            text,
            grid or g0,
            block,
            sizes or s0,
            "hand",
            "breadth blind",
            written=written if written is not None else w0,
            whatif={"base": f"bl13_{name}", "axis": axis},
        )


# round 13 (question 46): the tile times of every block, with its SM: an HGEMM with one more
# parameter, stamped by thread 0 at the start of each k-tile and at the end of the kernel
TRACE13 = {
    "s3": (128, 128, 32, 3, 128, 0),
    "s2": (128, 128, 32, 2, 128, 0),
    "sw8": (128, 128, 32, 3, 128, 8),
}
TRACE_SLOTS = 66  # k-tiles up to 64, the end of the kernel, the SM
TRACE_STAMP = (
    "if (threadIdx.x == 0) {{ unsigned* tr_ = trc + (blockIdx.y * gridDim.x + blockIdx.x) * "
    f"{2 * TRACE_SLOTS} + 2 * ({{slot}}); unsigned long long gt_; "
    'asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_)); '
    "tr_[0] = (unsigned)clock64(); tr_[1] = (unsigned)(gt_ >> 5); }}"
)
TRACE_SM = (
    'if (threadIdx.x == 0) { unsigned s_; asm volatile("mov.u32 %0, %%smid;" : "=r"(s_)); '
    f"trc[(blockIdx.y * gridDim.x + blockIdx.x) * {2 * TRACE_SLOTS} + {2 * TRACE_SLOTS - 2}] = s_; }}"
)
TRACE_DUMP = """  for (int l = 0; l < 4; l++) {
    launch();
    CK(cudaDeviceSynchronize());
    std::vector<unsigned> h(nt);
    CK(cudaMemcpy(h.data(), trc, nt * 4, cudaMemcpyDeviceToHost));
    printf("TRACE %d", l);
    for (unsigned x : h) printf(" %u", x);
    printf("\\n");
  }
  CK(cudaDeviceSynchronize());
  return 0;
}
"""


def _traced(out: Path, name: str, tile: tuple, sh: dict) -> None:
    if not _wanted(name):
        return
    bm, bn, bk, s, th, panel = tile
    M, N, K = sh["M"], sh["N"], sh["K"]
    c = _compiled(hgemm(bm, bn, bk, s, th, panel, sh))
    src = c["src"].replace("half_t* __restrict__ C)", "half_t* __restrict__ C, unsigned* trc)")
    loop = re.search(r"for \(int ko = 0; ko < \d+; \+\+ko\) \{", src)
    src = src[: loop.end()] + TRACE_STAMP.format(slot="ko") + src[loop.end() :]
    src = src[: loop.start()] + TRACE_SM + src[loop.start() :]
    end = src.rstrip().rfind("}")
    src = src[:end] + TRACE_STAMP.format(slot=TRACE_SLOTS - 2) + src[end:]
    grid = _tiles(M, N, bm, bn)
    nt = math.prod(grid) * 2 * TRACE_SLOTS
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "kernel.cu").write_text(src)
    tail = "  CK(cudaDeviceSynchronize());\n  return 0;\n}\n"
    harness = HARNESS % {
        "includes": '#include "kernel.cu"',
        "sizes": ", ".join(str(x) for x in (M * K * 2, K * N * 2, M * N * 2)),
        "nbuf": 3,
        "rot": 1,
        "setup": _setup(c["entry"], c["smem"])
        + f"\n  dim3 grid({grid[0]}, {grid[1]}, {grid[2]}), block({th});\n"
        + f"  const size_t nt = {nt};\n  unsigned* trc;\n  CK(cudaMalloc(&trc, nt * 4));",
        "launches": f"    {c['entry']}<<<grid, block, smem, st>>>({c['args']}, trc);",
        "name": name,
    }
    assert harness.endswith(tail)
    (d / "harness.cu").write_text(harness[: -len(tail)] + TRACE_DUMP)
    launch = {
        "entry": c["entry"],
        "grid": grid,
        "threads": th,
        "block": [th],
        "smem": c["smem"],
        "rot": 1,
        "footprint_bytes": (M * K + K * N + M * N) * 2,
        "write_bytes": M * N * 2,
        "family": "hgemm",
        "shape": sh,
        "pick": _pick(bm, bn, bk, s, th, swizzle=panel),
        "smem_estimate": s * (bm + bn) * bk * 2,
        "cache": "warm",
        "set": "tile trace",
    }
    (d / "launch.json").write_text(json.dumps(launch, indent=1) + "\n")


ATOM_WORDS = (1, 4, 16, 128, 1024)  # words the blocks share, every lane of a block on its word
ATOM_REPEAT = (1, 2, 4, 8)  # atomics in a row on one word
HASH_FOOT = (16, 1024, 4096, 16384, 65536)  # KiB hashed over, by the whole grid
HASH_LOCAL = (4, 16, 64, 256)  # KiB hashed over, each block its own
RECORD_WORDS = (1, 2, 4, 8, 16)  # of a 64-byte record per thread, MiB 4 and 16


def _sweeps13(out: Path) -> None:
    head = "__global__ void __launch_bounds__(256)"
    tid = "const unsigned t = blockIdx.x * 256 + threadIdx.x;"
    hash_ = "unsigned x = (t + k * 4099u) * 2654435761u;"
    for w in ATOM_WORDS:
        src = f"""// round 13: all lanes of a block add to one word of {w}, the same word every time
{head} k_atb(unsigned* h) {{
  for (unsigned k = 0; k < 8; ++k) atomicAdd(&h[blockIdx.x & {w - 1}u], 1u);
}}
"""
        _source_kernel(out, f"atb_w{w}", src, (1024, 1, 1), (256,), [w * 4], "hand", "atomic_bins")
    for p in ATOM_REPEAT:
        src = f"""// round 13: all lanes of a block on one word of 1024, a new word every {p} atomics
{head} k_atk(unsigned* h) {{
  for (unsigned k = 0; k < 8; ++k) atomicAdd(&h[(blockIdx.x * 37u + k / {p}u * 101u) & 1023u], 1u);
}}
"""
        _source_kernel(out, f"atk_p{p}", src, (1024, 1, 1), (256,), [4096], "hand", "atomic_bins")
    for f in HASH_FOOT:
        src = f"""// round 13: 2 M hashed stores over {f} KiB
{head} k_hs(unsigned* a) {{
  {tid}
  for (unsigned k = 0; k < 8; ++k) {{
    {hash_}
    a[(x >> 7) & {f * 256 - 1}u] = t;
  }}
}}
"""
        _source_kernel(out, f"hs_f{f}", src, (1024, 1, 1), (256,), [f << 10], "hand", "hash_sweep")
        src = f"""// round 13: 2 M hashed loads over {f} KiB, one sum per thread stored
{head} k_hl(unsigned* o, const unsigned* a) {{
  {tid}
  unsigned s = 0;
  for (unsigned k = 0; k < 8; ++k) {{
    {hash_}
    s += a[(x >> 7) & {f * 256 - 1}u];
  }}
  o[t] = s;
}}
"""
        sizes = [1 * MB, f << 10]
        _source_kernel(out, f"hl_f{f}", src, (1024, 1, 1), (256,), sizes, "hand", "hash_sweep")
    for f in HASH_LOCAL:
        src = f"""// round 13: 2 M hashed stores, each block over its own {f} KiB
{head} k_hb(unsigned* a) {{
  {tid}
  for (unsigned k = 0; k < 8; ++k) {{
    {hash_}
    a[blockIdx.x * {f * 256}u + ((x >> 7) & {f * 256 - 1}u)] = t;
  }}
}}
"""
        sizes = [f << 20]
        _source_kernel(out, f"hb_f{f}", src, (1024, 1, 1), (256,), sizes, "hand", "hash_sweep")
    for w in RECORD_WORDS:
        for m in (4, 16):
            src = f"""// round 13: {w} of the 16 words of a 64-byte record per thread, over {m} MiB
{head} k_rf(unsigned* a) {{
  const size_t r = (size_t)blockIdx.x * 256 + threadIdx.x;
  for (int k = 0; k < {w}; ++k) a[r * 16 + k] = (unsigned)r + k;
}}
"""
            g = (m * 64, 1, 1)  # 256 threads of 64 bytes each: 64 blocks per MiB
            name = f"rf_w{w}_m{m}"
            _source_kernel(out, name, src, g, (256,), [m * MB], "hand", "record_sweep")


def _round13b(out: Path) -> None:
    _sweeps13(out)
    for sh, tag in ((HGEMM_BLIND7_SHAPE, "m3072"), (HGEMM_BLIND9_SHAPE, "m1536"), (SHAPE, "c2048")):
        for name, tile in TRACE13.items():
            _traced(out, f"tr_{tag}_{name}", tile, sh)


def _tried(gen, out: Path, name: str, *a, **kw) -> None:
    """One kernel; a variant TileLang refuses leaves its group."""
    try:
        gen(out, name, *a, **kw)
    except Exception:  # noqa: BLE001 - reported, the group keeps its other variants
        print(f"FAILED {name}")
        traceback.print_exc()


def _round13(out: Path) -> None:
    _hand13(out)
    sh = HGEMM_BLIND13_SHAPE
    M, N, K = sh["M"], sh["N"], sh["K"]
    pre = f"hgemm_m{M}n{N}k{K}"
    for name, tile in HGEMM_BLIND12.items():
        one = name.removeprefix("c128x128x32_").replace("t128_", "").removesuffix("_t128")
        for cache, tail in (("warm", ""), ("rotating", " rotating")):
            suffix = "" if cache == "warm" else "_rotating"
            base = f"{pre}_c128x128x32_s3_t128{suffix}"
            more = (
                {}
                if f"{pre}_{name}{suffix}" == base
                else {"whatif": {"base": base, "axis": HG13_AXIS.get(one, "design")}}
            )
            _hgemm(
                out,
                f"{pre}_{name}{suffix}",
                tile,
                sh,
                cache,
                set=f"hgemm {M}x{N}x{K}{tail}",
                **more,
            )
    for v, (tile, opt, axis, both) in HG13_WI.items():
        for cache in ("warm", "rotating") if both else ("warm",):
            suffix = "" if cache == "warm" else "_rotating"
            more = {"whatif": {"base": f"{pre}_c128x128x32_s3_t128{suffix}", "axis": axis}}
            _hgemm(
                out, f"{pre}_{v}{suffix}", tile, sh, cache, opt, set=f"whatif hgemm {cache}", **more
            )
    tag = f"{GEMV13['N']}x{GEMV13['K']}"
    for cache, tail in (("warm", ""), ("rotating", " rotating")):
        for v, (kw, axis) in GEMV13_V.items():
            more = _wi(v, f"gemv13_b_{cache}", axis)
            group = f"gemv {tag}{tail}"
            _gemv_row(out, f"gemv13_{v}_{cache}", {**GEMV13, **kw}, cache, set=group, **more)
        more = _wi("", f"gemv13_b_{cache}", "compute removed")
        group = f"whatif gemv {cache}"
        _gemv_row(
            out, f"gemv13_nocompute_{cache}", {**GEMV13, "compute": False}, cache, set=group, **more
        )
    f13 = FA13
    tag = f"{f13['S']}x{f13['H']}x{f13['D']}"
    for cache, tail in (("warm", ""), ("rotating", " rotating")):
        for name, (bm, bn, s, th) in FA_BLIND7_V.items():
            f = {**f13, "BM": bm, "BN": bn, "stages": s, "threads": th}
            axis = FA13_AXIS.get(name, "base")
            more = _wi(name, f"fa13_c64x64_s2_t128_{cache}", axis)
            _tried(_fa, out, f"fa13_{name}_{cache}", f, cache, set=f"fa {tag}{tail}", **more)
        for v in ("nostore", "nov", "noexp"):
            kw, axis = dict(WI_FA[v][0]), WI_FA[v][1]
            writes = () if kw.get("store") else (-1,)
            more = _wi(v, f"fa13_c64x64_s2_t128_{cache}", axis)
            group = f"whatif fa {cache}"
            _fa(out, f"fa13_{v}_{cache}", {**f13, **kw}, cache, False, writes, set=group, **more)
    tag = f"{DECODE13['H']}x{DECODE13['S']}x{DECODE13['D']}"
    for cache, tail in (("warm", ""), ("rotating", " rotating")):
        for v, (kw, axis) in DECODE13_V.items():
            more = _wi(v, f"dec13_b_{cache}", axis)
            group = f"decode {tag}{tail}"
            c = {**DECODE13, **kw}
            _tried(_decode_part, out, f"dec13_{v}_{cache}", c, cache, set=group, **more)
    tag = f"{ROWS13['M']}x{ROWS13['N']}"
    for th in THREADS13:
        axis = "base" if th == 128 else "threads"
        kw = {"threads": th}
        for cache, tail in (("warm", ""), ("rotating", " rotating")):
            more = _wi("", f"rms13_t128_{cache}", axis)
            group = f"rmsnorm {tag}{tail}"
            _tried(_rms, out, f"rms13_t{th}_{cache}", ROWS13, cache, kw, set=group, **more)
        more = _wi("", "sm13_t128_warm", axis)
        _tried(_softmax, out, f"sm13_t{th}_warm", ROWS13, kw, set=f"softmax {tag}", **more)
    for v in ("nocompute", "nostore", "fp32"):
        kw, axis = WI_RMS[v]
        for cache in ("warm", "rotating"):
            more = _wi("", f"rms13_t128_{cache}", axis)
            _rms(
                out, f"rms13_{v}_{cache}", ROWS13, cache, kw, set=f"whatif rmsnorm {cache}", **more
            )
    kw, axis = WI_SOFTMAX["nostore"]
    more = _wi("", "sm13_t128_warm", axis)
    _softmax(out, "sm13_nostore_warm", ROWS13, kw, set="whatif softmax warm", **more)


# round 14: a blind set at shapes not on the scorecard; sweeps for records, stack reads by lane and
# the third stage (question 47), none of them blind rows
GEMV14 = {"N": 5120, "K": 2560, "group": 32, "n_part": 4, "reduce": 32}
FA14 = {**FA, "S": 2560, "H": 6, "D": 112}
DECODE14 = {**DECODE, "H": 40, "G": 10, "S": 2560, "D": 112}
ROWS14 = {"M": 1280, "N": 2560}
HGEMM_BLIND14_SHAPE = {"M": 2304, "N": 1792, "K": 1792}
HGEMM_BLIND14 = {
    "c128x128x32_s3_t128": (128, 128, 32, 3, 128, 0),
    "c128x128x32_s2_t128": (128, 128, 32, 2, 128, 0),
    "c128x128x32_s4_t128": (128, 128, 32, 4, 128, 0),
    "c128x128x64_s2_t128": (128, 128, 64, 2, 128, 0),
    "c128x256x32_s3_t256": (128, 256, 32, 3, 256, 0),
    "c256x128x32_s3_t256": (256, 128, 32, 3, 256, 0),
    "c64x128x32_s4_t128": (64, 128, 32, 4, 128, 0),
    "c128x256x64_s2_t256": (128, 256, 64, 2, 256, 0),
    "c128x128x32_s3_t128_sw8": (128, 128, 32, 3, 128, 8),
    "c64x64x32_s4_t64": (64, 64, 32, 4, 64, 0),
}
HAND14 = {
    "histlane": ((1024, 1, 1), (256,), [1024, 4 * MB], None),
    "countblock": ((1024, 1, 1), (256,), [256, 4 * MB], None),
    "minblock": ((1024, 1, 1), (256,), [512, 4 * MB], None),
    "xorwide": ((1024, 1, 1), (256,), [16 * MB], None),
    "exchring": ((1024, 1, 1), (256,), [1 * MB, 256 << 10], None),
    "stridefill": ((1024, 1, 1), (256,), [32 * MB], None),
    "brevfill": ((1024, 1, 1), (256,), [8 * MB], None),
    "hashput": ((1024, 1, 1), (256,), [8 * MB, 4 * MB], None),
    "maskput": ((1024, 1, 1), (256,), [8 * MB, 4 * MB], None),
    "recordinit": ((1024, 1, 1), (256,), [8 * MB], None),
    "recordpatch": ((1024, 1, 1), (256,), [16 * MB], None),
    "recordwide": ((512, 1, 1), (256,), [16 * MB], None),
    "stackscan": ((512, 1, 1), (256,), [1 * MB, 4 * MB], None),
    "stackfilter": ((512, 1, 1), (256,), [1 * MB, 4 * MB], None),
    "stackmerge": ((512, 1, 1), (256,), [1 * MB, 4 * MB], None),
}
WI_HAND14 = (
    ("histlane", "b64", "shape", None, None, None, (("255u", "63u"),)),
    ("countblock", "nl", "loads removed", None, None, None, (("a[(t * 8 + k) & 1048575u]", "t"),)),
    ("xorwide", "m1", "shape", None, [1 * MB], None, (("4194303u", "262143u"),)),
    ("exchring", "t128", "threads", (2048, 1, 1), None, None, T128),
    ("stridefill", "m4", "shape", None, [4 * MB], None, (("8388607u", "1048575u"),)),
    ("hashput", "m1", "shape", None, [1 * MB, 4 * MB], None, (("2097151u", "262143u"),)),
    ("recordinit", "w4", "stores removed", None, None, None, (("k < 8", "k < 4"),)),
    ("recordwide", "w8", "stores removed", None, None, None, (("k < 32", "k < 8"),)),
    ("stackscan", "t128", "threads", (1024, 1, 1), None, None, T128),
    ("stackfilter", "t128", "threads", (1024, 1, 1), None, None, T128),
)
OCCUPANCY = {  # threads: blocks per SM on average, around the limit by 1536 and by 1024 threads
    64: (8, 16, 24, 25),
    128: (1, 6, 8, 9, 12, 13),
    256: (1, 3, 4, 5, 6, 7),
    512: (1, 2, 3, 4),
    1024: (1, 2),
}
STACK_READ = (16, 64, 256, 1024)  # words of the frame, read four times over
QS_SHAPE = {"M": 3072, "N": 1536, "K": 2048}
QS_ONE, QS_TWO = {"M": 1024, "N": 768, "K": 2048}, {"M": 1536, "N": 1024, "K": 2048}


def _first_barrier_out(src: str) -> str:
    return re.sub(r"(for \(int ko = 0; ko < \d+; \+\+ko\) \{\s*)__syncthreads\(\);\s*", r"\1", src)


def _copies_out(src: str) -> str:
    """The stages' rotation and barriers without a byte copied."""
    for pat in (
        r"tl::cp_async_gs<\d+>\(.*\);",
        r"tl::cp_async_commit\(\);",
        r"tl::cp_async_wait<\d+>\(\);",
    ):
        src = re.sub(pat, ";", src)
    return src


def _wait_one(src: str) -> str:
    return src.replace("tl::cp_async_wait<2>();", "tl::cp_async_wait<1>();", 1)


def _twelve_registers(src: str) -> str:
    """Twelve words live over the whole kernel: registers per thread as s3 has them, little code."""
    names = [f"keep{i}" for i in range(12)]
    defs = "".join(
        f"  const unsigned {n} = ((const unsigned*)A)[threadIdx.x + {i * 128}];\n"
        for i, n in enumerate(names)
    )
    src = src.replace("buf_dyn_shmem[];\n", "buf_dyn_shmem[];\n" + defs, 1)
    end = src.rstrip().rfind("}")
    use = f"  if (({' ^ '.join(names)}) == 0x5a5a5a5au) C[0] = half_t(0.0f);\n"
    return src[:end] + use + src[end:]


def _copies_late(src: str) -> str:
    """The next tile's copies after the compute: nothing but the wait stands before the compute."""
    m = re.search(r"for \(int ko = 0; ko < \d+; \+\+ko\) \{\s*__syncthreads\(\);", src)
    end = src.index("tl::cp_async_commit();", m.end()) + len("tl::cp_async_commit();")
    copies = src[m.end() : end]
    w = re.compile(r"\s*tl::cp_async_wait<\d+>\(\);\s*__syncthreads\(\);").match(src, end)
    depth, i = 0, w.end()
    while True:  # the compute's block ends where the loop's body does
        depth += {"{": 1, "}": -1}.get(src[i], 0)
        if depth < 0:
            break
        i += 1
    return src[: m.end()] + src[end:i] + copies + src[i:]


def _hand14(out: Path) -> None:
    src_dir = Path(__file__).parent / "hand14"
    for name, (grid, block, sizes, written) in HAND14.items():
        text = (src_dir / f"{name}.cu").read_text()
        _source_kernel(
            out, f"bl14_{name}", text, grid, block, sizes, "hand", "breadth blind", written=written
        )
    for name, tag, axis, grid, sizes, written, subs in WI_HAND14:
        g0, block, s0, w0 = HAND14[name]
        text = (src_dir / f"{name}.cu").read_text()
        for old, new in subs:
            text = text.replace(old, new)
        if subs == T128:
            block = (128,)
        _source_kernel(
            out,
            f"bl14_{name}_{tag}",
            text,
            grid or g0,
            block,
            sizes or s0,
            "hand",
            "breadth blind",
            written=written if written is not None else w0,
            whatif={"base": f"bl14_{name}", "axis": axis},
        )


def _sweeps14(out: Path) -> None:
    head = "__global__ void __launch_bounds__(256)"
    for n in STACK_READ:
        for tag, index in (("u", "(k * 5u)"), ("l", "(k * 5u + t * 3u)")):
            src = f"""// round 14: {n} words of stack, read {4 * n} times at an index {tag}
{head} k_stl(unsigned* o) {{
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned r[{n}];
  for (unsigned i = 0; i < {n}; ++i) r[i] = i * t;
  unsigned s = 0;
  for (unsigned k = 0; k < {4 * n}; ++k) s += r[{index} & {n - 1}u];
  o[t] = s;
}}
"""
            sizes = [1 * MB]
            _source_kernel(
                out, f"stl_{tag}_n{n}", src, (256, 1, 1), (256,), sizes, "hand", "stack_load"
            )
    t2, t3 = (128, 128, 32, 2, 128, 0), (128, 128, 32, 3, 128, 0)
    sh, group = QS_SHAPE, {"set": "stage_question"}
    _hgemm(out, "qs_s2", t2, sh, "warm", **group)
    _hgemm(out, "qs_s3", t3, sh, "warm", **group)
    _hgemm(out, "qs_s2_smem3", t2, sh, "warm", smem=49152, **group)
    _hgemm(out, "qs_s2_reg12", t2, sh, "warm", edit=_twelve_registers, **group)
    _hgemm(out, "qs_s3_wait1", t3, sh, "warm", edit=_wait_one, **group)
    _hgemm(out, "qs_s2_b1", t2, sh, "warm", edit=_first_barrier_out, **group)
    _hgemm(out, "qs_s3_b1", t3, sh, "warm", edit=_first_barrier_out, **group)
    _hgemm(out, "qs_s2_nocopy", t2, sh, "warm", edit=_copies_out, **group)
    _hgemm(out, "qs_s3_nocopy", t3, sh, "warm", edit=_copies_out, **group)
    _hgemm(out, "qs_s2_late", t2, sh, "warm", edit=_copies_late, **group)
    _hgemm(out, "qs_s3_late", t3, sh, "warm", edit=_copies_late, **group)
    for tag, shape in (("qs1", QS_ONE), ("qs2", QS_TWO)):
        for s in (1, 2, 3):
            _hgemm(out, f"{tag}_s{s}", (128, 128, 32, s, 128, 0), shape, "warm", **group)
    _hgemm(out, "qs1_s2_nocopy", t2, QS_ONE, "warm", edit=_copies_out, **group)
    for tag, tile in (("s2", t2), ("s3", t3)):
        _hgemm(out, f"qs2_{tag}_nocopy", tile, QS_TWO, "warm", edit=_copies_out, **group)
        _hgemm(out, f"qs2_{tag}_b1", tile, QS_TWO, "warm", edit=_first_barrier_out, **group)
    for th, ks in OCCUPANCY.items():
        for k in ks:
            src = f"""// round 14: blocks of {th} threads, {k} per SM on average: one warp chases loads, the rest wait
__global__ void __launch_bounds__({th}) k_occ(unsigned* o, const unsigned* a) {{
  unsigned x = blockIdx.x * {th} + threadIdx.x;
  if (threadIdx.x < 32)
    for (int i = 0; i < 64; ++i) x = a[(x * 2654435761u >> 12) & 1048575u];
  __syncthreads();
  o[blockIdx.x & 65535u] = x;
}}
"""
            sizes = [256 << 10, 4 * MB]
            g = (48 * k, 1, 1)
            _source_kernel(out, f"occ_t{th}_k{k}", src, g, (th,), sizes, "hand", "occupancy")


def _round14(out: Path) -> None:
    _hand14(out)
    _sweeps14(out)
    sh = HGEMM_BLIND14_SHAPE
    M, N, K = sh["M"], sh["N"], sh["K"]
    pre = f"hgemm_m{M}n{N}k{K}"
    for name, tile in HGEMM_BLIND14.items():
        for cache, tail in (("warm", ""), ("rotating", " rotating")):
            suffix = "" if cache == "warm" else "_rotating"
            base = f"{pre}_c128x128x32_s3_t128{suffix}"
            more = _wi(name, base, "base" if f"{pre}_{name}{suffix}" == base else "design")
            group = f"hgemm {M}x{N}x{K}{tail}"
            _tried(_hgemm, out, f"{pre}_{name}{suffix}", tile, sh, cache, set=group, **more)
    tag = f"{GEMV14['N']}x{GEMV14['K']}"
    for cache, tail in (("warm", ""), ("rotating", " rotating")):
        for v, (kw, axis) in GEMV13_V.items():
            more = _wi(v, f"gemv14_b_{cache}", axis)
            group = f"gemv {tag}{tail}"
            c = {**GEMV14, **kw}
            _tried(_gemv_row, out, f"gemv14_{v}_{cache}", c, cache, set=group, **more)
    tag = f"{FA14['S']}x{FA14['H']}x{FA14['D']}"
    for cache, tail in (("warm", ""), ("rotating", " rotating")):
        for name, (bm, bn, s, th) in FA_BLIND7_V.items():
            f = {**FA14, "BM": bm, "BN": bn, "stages": s, "threads": th}
            more = _wi(name, f"fa14_c64x64_s2_t128_{cache}", FA13_AXIS.get(name, "base"))
            _tried(_fa, out, f"fa14_{name}_{cache}", f, cache, set=f"fa {tag}{tail}", **more)
    tag = f"{DECODE14['H']}x{DECODE14['S']}x{DECODE14['D']}"
    for cache, tail in (("warm", ""), ("rotating", " rotating")):
        for v, (kw, axis) in DECODE13_V.items():
            more = _wi(v, f"dec14_b_{cache}", axis)
            group = f"decode {tag}{tail}"
            c = {**DECODE14, **kw}
            _tried(_decode_part, out, f"dec14_{v}_{cache}", c, cache, set=group, **more)
    tag = f"{ROWS14['M']}x{ROWS14['N']}"
    for th in THREADS13:
        axis = "base" if th == 128 else "threads"
        kw = {"threads": th}
        for cache, tail in (("warm", ""), ("rotating", " rotating")):
            more = _wi("", f"rms14_t128_{cache}", axis)
            group = f"rmsnorm {tag}{tail}"
            _tried(_rms, out, f"rms14_t{th}_{cache}", ROWS14, cache, kw, set=group, **more)
        more = _wi("", "sm14_t128_warm", axis)
        _tried(_softmax, out, f"sm14_t{th}_warm", ROWS14, kw, set=f"softmax {tag}", **more)


# round 15: blind again at shapes with other divisors; new address forms; what differs at the DRAM
# limit (question 50), FA at D by 8 (question 48), compute alone at block turnover (question 47)
GEMV15 = {"N": 6144, "K": 3072, "group": 48, "n_part": 4, "reduce": 32}
FA15 = {**FA, "S": 2304, "H": 6, "D": 144}
DECODE15 = {**DECODE, "H": 36, "G": 12, "S": 2304, "D": 144}
ROWS15 = {"M": 1920, "N": 1792}
HGEMM_BLIND15_SHAPE = {"M": 2688, "N": 1920, "K": 1728}
HGEMM_BLIND15 = {
    "c128x128x32_s3_t128": (128, 128, 32, 3, 128, 0),
    "c128x128x32_s2_t128": (128, 128, 32, 2, 128, 0),
    "c128x128x32_s4_t128": (128, 128, 32, 4, 128, 0),
    "c128x128x64_s2_t128": (128, 128, 64, 2, 128, 0),
    "c128x256x32_s3_t256": (128, 256, 32, 3, 256, 0),
    "c256x128x32_s2_t256": (256, 128, 32, 2, 256, 0),
    "c64x128x32_s4_t128": (64, 128, 32, 4, 128, 0),
    "c128x128x32_s3_t128_sw8": (128, 128, 32, 3, 128, 8),
}
GEMV15_V = ("b", "np2", "np8", "r16", "r64", "mk16", "np2_r64")
DECODE15_V = ("b", "s1", "s3", "split4", "split16", "bn32", "split4_bn32")
FA15_V = (
    "c64x64_s2_t128",
    "c64x64_s3_t128",
    "c64x64_s1_t128",
    "c64x32_s3_t128",
    "c128x64_s2_t128",
    "c64x64_s2_t64",
    "c32x64_s2_t64",
)
HAND15 = {
    "divseven": ((1024, 1, 1), (256,), [1 * MB, 8 * MB], None),
    "divrun": ((1024, 1, 1), (256,), [16 * MB], None),
    "modring": ((2048, 1, 1), (128,), [8 * MB, 8 * MB], None),
    "triangle": ((1024, 1, 1), (256,), [1 * MB, 40 * MB], None),
    "morton": ((1024, 1, 1), (256,), [1 * MB, 16 * MB], None),
    "clampread": ((2048, 1, 1), (128,), [1 * MB, 4 * MB], None),
    "twobuf": ((1024, 1, 1), (256,), [1 * MB, 4 * MB, 4 * MB], None),
    "popindex": ((1024, 1, 1), (256,), [1 * MB, 64 << 10], None),
    "reverse": ((1024, 1, 1), (256,), [16 * MB, 16 * MB], None),
    "wideindex": ((1024, 1, 1), (256,), [1 * MB, 8 * MB], None),
    "floatindex": ((2048, 1, 1), (128,), [1 * MB, 1 * MB], None),
    "wrapstencil": ((8190, 1, 1), (256,), [8 * MB, 8 * MB], None),
    "chunked": ((2048, 1, 1), (128,), [16 * MB, 16 * MB], None),
    "gather3": ((1024, 1, 1), (256,), [1 * MB, 8 * MB], None),
}
WI_HAND15 = (
    ("divseven", "t128", "threads", (2048, 1, 1), None, None, T128),
    ("morton", "k1", "loads removed", None, None, None, (("k < 4", "k < 1"),)),
    ("reverse", "m4", "shape", (128, 1, 1), [4 * MB, 4 * MB], None, (("4194303", "1048575"),)),
    ("chunked", "w6", "stores removed", None, None, None, (("k < 12", "k < 6"),)),
)
DL_WIDTH = {4: ("unsigned", 16, "x"), 8: ("uint2", 8, "x.x ^ x.y"), 16: ("uint4", 4, "x.x ^ x.y ^ x.z ^ x.w")}
DL_T = (32, 128, 512)
DL_BYTES = 8 * MB
FA_D = (64, 80, 96, 112, 128, 144, 160, 176)
CT_K = (480, 960, 1920, 3840)
CT_SHAPE = {"M": 1536, "N": 3072}


def _hand15(out: Path) -> None:
    src_dir = Path(__file__).parent / "hand15"
    for name, (grid, block, sizes, written) in HAND15.items():
        text = (src_dir / f"{name}.cu").read_text()
        _source_kernel(
            out, f"bl15_{name}", text, grid, block, sizes, "hand", "breadth blind", written=written
        )
    for name, tag, axis, grid, sizes, written, subs in WI_HAND15:
        g0, block, s0, w0 = HAND15[name]
        text = (src_dir / f"{name}.cu").read_text()
        for old, new in subs:
            text = text.replace(old, new)
        if subs == T128:
            block = (128,)
        _source_kernel(
            out,
            f"bl15_{name}_{tag}",
            text,
            grid or g0,
            block,
            sizes or s0,
            "hand",
            "breadth blind",
            written=written if written is not None else w0,
            whatif={"base": f"bl15_{name}", "axis": axis},
        )


def _dl(out: Path) -> None:
    """Question 50: the same bytes read at the DRAM limit, by bytes per lane and threads."""
    for w, (ty, m, sum_) in DL_WIDTH.items():
        for th in DL_T:
            for cache in ("rotating", "warm"):
                if cache == "warm" and (w != 8 or th == 32):
                    continue
                src = f"""// round 15: {DL_BYTES >> 20} MiB read in order, {w} bytes per lane, {m} loads per thread
__global__ void __launch_bounds__({th}) k_dl(unsigned* o, const {ty}* a) {{
  const unsigned t = blockIdx.x * {th} + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < {m}; ++k) {{
    const {ty} x = a[t * {m} + k];
    s += {sum_};
  }}
  o[blockIdx.x & 65535u] = s;
}}
"""
                grid = (DL_BYTES // (th * 64), 1, 1)
                rot = _rots(cache, DL_BYTES)
                _source_kernel(
                    out,
                    f"dl_w{w}_t{th}_{cache}",
                    src,
                    grid,
                    (th,),
                    [1 * MB, DL_BYTES],
                    "hand",
                    "dram_limit",
                    rot=rot,
                    written=1024,
                )


def _sweeps15(out: Path) -> None:
    _dl(out)
    # question 50: the rows themselves with a deeper rotation: what stays in the L2 goes
    for v in ("b", "mk16", "np2_r64"):
        c = {**GEMV14, **GEMV13_V[v][0]}
        _tried(_gemv_deep, out, f"gd_{v}", c, set="dram_limit")
    for th in (16, 512):
        kw = {"threads": th}
        _tried(_rms, out, f"rd_t{th}", ROWS14, "rotating", kw, depth=4, set="dram_limit")
    # question 48: D by 16 at t128: bytes per thread by 32
    for d in FA_D:
        f = {**FA, "S": 768, "H": 4, "D": d, "BM": 64, "BN": 64, "stages": 2, "threads": 128}
        _tried(_fa, out, f"fad_d{d}_t128", f, "warm", set="fa_stride")
    # question 47: compute alone, blocks fixed, tiles by K: the intercept is the turnover
    t2 = (128, 128, 32, 2, 128, 0)
    for k in CT_K:
        sh = {**CT_SHAPE, "K": k}
        _hgemm(out, f"ct_k{k}", t2, sh, "warm", edit=_copies_out, set="turnover")


def _round15b(out: Path) -> None:
    """The write limit beside single-use reads, between r11b's 6 and 7 MiB: rows of softmax."""
    for m in (1920, 1960, 2000, 2040, 2080, 2240):
        sh = {"M": m, "N": 1792}
        _tried(_softmax, out, f"wl_m{m}", sh, {"threads": 128}, set="write_limit")


def _gemv_deep(out: Path, name: str, c: dict, **more) -> None:
    N, K, g = c["N"], c["K"], c["group"]
    sizes, eb = [K, N * K // 2, N * K // g, N], [2, 1, 2, 2]
    info = {"family": "gemv_w4a16", "shape": {"N": N, "K": K}, "pick": c, "cache": "rotating", **more}
    write(
        out,
        name,
        gemv_w4a16(**c),
        sizes,
        [-(-N // c["n_part"]), 1],
        (c["reduce"], c["n_part"]),
        info,
        elem_bytes=eb,
        rot=_rots("rotating", sum(a * b for a, b in zip(sizes, eb, strict=True)), 4),
    )


def _round15(out: Path) -> None:
    _hand15(out)
    _sweeps15(out)
    sh = HGEMM_BLIND15_SHAPE
    M, N, K = sh["M"], sh["N"], sh["K"]
    pre = f"hgemm_m{M}n{N}k{K}"
    for name, tile in HGEMM_BLIND15.items():
        for cache, tail in (("warm", ""), ("rotating", " rotating")):
            suffix = "" if cache == "warm" else "_rotating"
            base = f"{pre}_c128x128x32_s3_t128{suffix}"
            more = _wi(name, base, "base" if f"{pre}_{name}{suffix}" == base else "design")
            group = f"hgemm {M}x{N}x{K}{tail}"
            _tried(_hgemm, out, f"{pre}_{name}{suffix}", tile, sh, cache, set=group, **more)
    tag = f"{GEMV15['N']}x{GEMV15['K']}"
    for cache, tail in (("warm", ""), ("rotating", " rotating")):
        for v in GEMV15_V:
            kw, axis = GEMV13_V[v]
            more = _wi(v, f"gemv15_b_{cache}", axis)
            c = {**GEMV15, **kw}
            _tried(_gemv_row, out, f"gemv15_{v}_{cache}", c, cache, set=f"gemv {tag}{tail}", **more)
    tag = f"{FA15['S']}x{FA15['H']}x{FA15['D']}"
    for cache, tail in (("warm", ""), ("rotating", " rotating")):
        for name in FA15_V:
            bm, bn, s, th = FA_BLIND7_V[name]
            f = {**FA15, "BM": bm, "BN": bn, "stages": s, "threads": th}
            more = _wi(name, f"fa15_c64x64_s2_t128_{cache}", FA13_AXIS.get(name, "base"))
            _tried(_fa, out, f"fa15_{name}_{cache}", f, cache, set=f"fa {tag}{tail}", **more)
    tag = f"{DECODE15['H']}x{DECODE15['S']}x{DECODE15['D']}"
    for cache, tail in (("warm", ""), ("rotating", " rotating")):
        for v in DECODE15_V:
            kw, axis = DECODE13_V[v]
            more = _wi(v, f"dec15_b_{cache}", axis)
            c = {**DECODE15, **kw}
            group = f"decode {tag}{tail}"
            _tried(_decode_part, out, f"dec15_{v}_{cache}", c, cache, set=group, **more)
    tag = f"{ROWS15['M']}x{ROWS15['N']}"
    for th in THREADS13:
        axis = "base" if th == 128 else "threads"
        kw = {"threads": th}
        for cache, tail in (("warm", ""), ("rotating", " rotating")):
            more = _wi("", f"rms15_t128_{cache}", axis)
            group = f"rmsnorm {tag}{tail}"
            _tried(_rms, out, f"rms15_t{th}_{cache}", ROWS15, cache, kw, set=group, **more)
        more = _wi("", "sm15_t128_warm", axis)
        _tried(_softmax, out, f"sm15_t{th}_warm", ROWS15, kw, set=f"softmax {tag}", **more)


# round 16: the blind recipe of round 14 at new sizes. Rule, chosen before any result: off HGEMM a
# factor 7 in every size, HGEMM a factor 25 in M; tiles and divisors as TileLang accepts them.
GEMV16 = {"N": 7168, "K": 2688, "group": 56, "n_part": 4, "reduce": 32}
FA16 = {**FA, "S": 2240, "H": 7, "D": 112}
DECODE16 = {**DECODE, "H": 28, "G": 7, "S": 2240, "D": 112}
ROWS16 = {"M": 896, "N": 3584}  # N takes T 512; 2240 was refused at T 128 and above
HGEMM_BLIND16_SHAPE = {"M": 3200, "N": 1280, "K": 1600}
HAND16 = {
    "atomhash": ((1024, 1, 1), (256,), [4 * MB], None),
    "blockmin": ((1024, 1, 1), (128,), [1 * MB, 4 * MB], None),
    "recordthree": ((1024, 1, 1), (256,), [8 * MB], None),
    "packrows": ((2048, 1, 1), (256,), [2 * MB, 8 * MB], None),
    "stridewalk": ((2048, 1, 1), (128,), [1 * MB, 8 * MB], None),
    "revrows": ((1024, 1, 1), (128,), [4 * MB, 8 * MB], None),
    "primering": ((1024, 1, 1), (256,), [4 * MB, 1 * MB], None),
    "stackwin": ((512, 1, 1), (256,), [1 * MB, 1 * MB], None),
    "stackhist": ((1024, 1, 1), (128,), [1 * MB, 1 * MB], None),
    "twolevel": ((1024, 1, 1), (256,), [1 * MB, 1 * MB, 4 * MB], None),
    "gatherfour": ((1024, 1, 1), (256,), [1 * MB, 1 * MB, 4 * MB], None),
    "ninepoint": ((2048, 1, 1), (256,), [8 * MB, 8 * MB], None),
    "xorput": ((1024, 1, 1), (256,), [8 * MB], None),
}
WI_HAND16 = (
    ("stridewalk", "t256", "threads", (1024, 1, 1), None, None, ()),
    ("primering", "k2", "stores removed", None, None, None, (("k < 4", "k < 2"),)),
    ("xorput", "k2", "stores removed", None, None, None, (("k < 4", "k < 2"),)),
    ("gatherfour", "two", "loads removed", None, None, None, (("b[i] + b[i + 3] + b[i + 64] + b[i + 1024]", "b[i] + b[i + 1024]"),)),
)
MEM_READ = """// round 16: {what}
__global__ void __launch_bounds__({t}) k_mem(unsigned* o, const {ty}* a) {{
  const unsigned t = blockIdx.x * {t} + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < {n}; ++k) {{
    const {ty} x = {load};
    s += {sum_};
  }}
  o[blockIdx.x & 65535u] = s;
}}
"""
MEM_WRITE = """// round 16: {what}
__global__ void __launch_bounds__({t}) k_mem(unsigned* o{more}) {{
  const unsigned t = blockIdx.x * {t} + threadIdx.x;
  for (unsigned k = 0; k < {n}; ++k) {{
    {body}
  }}
}}
"""
HASH = "2654435761u"
W8 = ("uint2", "x.x ^ x.y")


def _mem_read(out, name, group, what, t, n, load, grid, bytes_, cache="warm", ty=W8, rot=None):
    src = MEM_READ.format(what=what, t=t, ty=ty[0], n=n, load=load, sum_=ty[1])
    r = _rots(cache, bytes_) if rot is None else rot
    _source_kernel(out, name, src, (grid, 1, 1), (t,), [1 * MB, bytes_], "hand", group, rot=r, written=1024)


def _mem_dram(out: Path) -> None:
    """Read × DRAM (rotating) and L2 retention: pattern × footprint × threads, one sweep."""
    g = "mem_dram"
    for f in (1, 2, 4, 16, 32, 64):
        _mem_read(out, f"mr_o_f{f}", g, f"{f} MiB in order", 128, 8, "a[t * 8 + k]", f * MB // (128 * 64), f * MB, "rotating")
    for d in (2, 4, 8):
        _mem_read(out, f"mr_o_f8_d{d}", g, "8 MiB in order, deeper rotation", 128, 8, "a[t * 8 + k]", 1024, 8 * MB, rot=d * _rot(8 * MB))
    for t in (32, 128, 512):
        n = 4
        grid = 8 * MB // (t * 16 * n)
        _mem_read(out, f"mr_cs_t{t}", g, "8 MiB in order, evict-first", t, n, "__ldcs(a + t * 4 + k)", grid, 8 * MB, "rotating", ("int4", "x.x ^ x.y ^ x.z ^ x.w"))
        _mem_read(out, f"mr_cg_t{t}", g, "8 MiB in order, plain, the same width", t, n, "a[t * 4 + k]", grid, 8 * MB, "rotating", ("int4", "x.x ^ x.y ^ x.z ^ x.w"))
    for s in (64, 128, 256, 1024, 4096):
        grid = 8 * MB // (128 * s)
        _mem_read(out, f"mr_s{s}", g, f"8 MiB, lanes {s} bytes apart", 128, 1, f"a[t * {s // 8}]", grid, 8 * MB, "rotating")
    for r in (512, 2048, 4096):
        grid = 8 * MB // 8192
        n = r // (32 * 8)
        _mem_read(out, f"mr_row_r{r}", g, f"rows of {r} bytes, pitch 8 KiB", 32, n, f"a[blockIdx.x * 1024 + threadIdx.x * {n} + k]", grid, 8 * MB, "rotating")
    for f, t in ((8, 128), (8, 512), (32, 128), (32, 512)):
        grid = f * MB // (t * 64)
        mask = f * MB // 8 - 1
        _mem_read(out, f"mr_h_f{f}_t{t}", g, f"hashed through {f} MiB", t, 8, f"a[((t * 8 + k) * {HASH}) >> 7 & {mask}u]", grid, f * MB, "rotating")
    for kind, load in (("cg", "p[(size_t)i * 48 * 1024]"), ("cs", "__ldcs(p + (size_t)i * 48 * 1024)")):
        src = f"""// round 16: dram_burst's kernel: 48 blocks in step, 8 MiB, {kind}
__global__ void __launch_bounds__(1024) k_mem(int* o, const int4* a) {{
  const int4* p = a + blockIdx.x * 1024 + threadIdx.x;
  int s = 0;
  for (int i = 0; i < 10; i++) {{
    const int4 v = {load};
    s ^= v.x ^ v.y ^ v.z ^ v.w;
  }}
  if (s == 0x7fff1234) o[0] = s;
}}
"""
        _source_kernel(out, f"mr_b48_{kind}", src, (48, 1, 1), (1024,), [1 * MB, 8 * MB], "hand", g, rot=_rot(8 * MB), written=0)
    for grid in (1, 12, 48):
        _mem_read(out, f"mr_g{grid}", g, f"{grid} blocks, 256 KiB each", 128, 256, "a[t * 256 + k]", grid, grid * 256 * 1024, "rotating")
    for f, r in ((8, 2), (8, 3), (8, 4), (8, 6), (4, 4), (4, 6), (4, 8)):
        _mem_read(out, f"mk_f{f}_r{r}", "mem_retain", f"{f} MiB, {r} sets in turn", 128, 8, "a[t * 8 + k]", f * MB // (128 * 64), f * MB, rot=r)


def _mem_threads(out: Path) -> None:
    """Rows warm by threads: 1920 rows of 4 KiB: loads, stores, both, loads and a barrier."""
    for t in (32, 128, 256, 512):
        n = 4096 // (t * 8)
        g = "mem_threads"
        idx = f"blockIdx.x * 512 + threadIdx.x * {n}"
        _mem_read(out, f"mt_r_t{t}", g, "rows warm, loads", t, n, f"a[{idx} + k]", 1920, 8 * MB)
        src = MEM_WRITE.format(what="rows warm, stores", t=t, n=n, more="", body=f"((uint2*)o)[{idx} + k] = make_uint2(t, k);")
        _source_kernel(out, f"mt_w_t{t}", src, (1920, 1, 1), (t,), [8 * MB], "hand", g)
        src = MEM_WRITE.format(what="rows warm, loads then stores", t=t, n=n, more=", const uint2* a", body=f"((uint2*)o)[{idx} + k] = a[{idx} + k];")
        _source_kernel(out, f"mt_rw_t{t}", src, (1920, 1, 1), (t,), [8 * MB, 8 * MB], "hand", g)
        src = MEM_READ.format(what="rows warm, loads and a barrier", t=t, ty="uint2", n=n, load=f"a[{idx} + k]", sum_="x.x ^ x.y").replace("  o[blockIdx", "  __syncthreads();\n  o[blockIdx")
        _source_kernel(out, f"mt_rb_t{t}", src, (1920, 1, 1), (t,), [1 * MB, 8 * MB], "hand", g, written=1024)


def _mem_write(out: Path) -> None:
    """Write × L2 at the dirty limit, by blocks, with reads beside; write × DRAM; RMW."""
    g = "mem_write"
    st = "((uint2*)o)[t * 8 + k] = make_uint2(t, k);"
    rw = "((uint2*)o)[t * 8 + k] = a[t * 8 + k];"
    for f10 in (50, 60, 65, 70, 75, 80, 120):
        b = f10 * MB // 10
        src = MEM_WRITE.format(what=f"{f10 / 10} MiB written", t=128, n=8, more="", body=st)
        _source_kernel(out, f"mw_o_f{f10}", src, (b // (128 * 64), 1, 1), (128,), [b], "hand", g)
        src = MEM_WRITE.format(what=f"{f10 / 10} MiB written, as much read once", t=128, n=8, more=", const uint2* a", body=rw)
        _source_kernel(out, f"mw_rw_f{f10}", src, (b // (128 * 64), 1, 1), (128,), [b, b], "hand", g)
    for t in (32, 512):
        b = 7 * MB
        src = MEM_WRITE.format(what="7 MiB written, read once, by threads", t=t, n=8, more=", const uint2* a", body=rw)
        _source_kernel(out, f"mw_rw_f70_t{t}", src, (b // (t * 64), 1, 1), (t,), [b, b], "hand", g)
    b = 7 * MB
    src = MEM_WRITE.format(what="7 MiB written, read once, few blocks", t=1024, n=8, more=", const uint2* a", body=rw)
    _source_kernel(out, "mw_rw_f70_t1024", src, (b // (1024 * 64), 1, 1), (1024,), [b, b], "hand", g)
    for f in (16, 32, 64):
        b = f * MB
        src = MEM_WRITE.format(what=f"{f} MiB written", t=128, n=8, more="", body=st)
        _source_kernel(out, f"mw_o_f{f}0", src, (b // (128 * 64), 1, 1), (128,), [b], "hand", g)
    for f in (16, 64):
        b = f * MB
        body = f"o[((t * 8 + k) * {HASH}) >> 6 & {b // 4 - 1}u] = t;"
        src = MEM_WRITE.format(what=f"{f} MiB hashed", t=128, n=8, more="", body=body)
        _source_kernel(out, f"mw_h_f{f}", src, (b // (128 * 32), 1, 1), (128,), [b], "hand", g)
    src = MEM_WRITE.format(what="16 MiB, lanes 4 KiB apart", t=128, n=1, more="", body="o[t * 1024] = t;")
    _source_kernel(out, "mw_s4096_f16", src, (16 * MB // (128 * 4096), 1, 1), (128,), [16 * MB], "hand", g)
    for f in (4, 8, 16, 32):
        b = f * MB
        src = MEM_WRITE.format(what=f"{f} MiB read and written in place", t=128, n=8, more="", body="o[t * 8 + k] += 1u;")
        _source_kernel(out, f"mx_f{f}", src, (b // (128 * 32), 1, 1), (128,), [b], "hand", g)


def _mem_stride(out: Path) -> None:
    """Bytes a thread holds by 32: global loads, shared memory, copies; D as FA's rows."""
    g = "mem_sector"
    for d in (48, 64, 80, 96, 112, 128):
        w = d // 16
        _mem_read(out, f"ms_g_d{d}", g, f"16-byte loads, {d} bytes apart", 128, 4, f"a[(t * 4 + k) * {w}]", 4 * MB // (128 * 4 * d), 4 * MB, ty=("uint4", "x.x ^ x.y ^ x.z ^ x.w"))
        src = f"""// round 16: shared memory, 16 bytes each thread, {d} bytes apart: stores, barrier, loads
__global__ void __launch_bounds__(128) k_mem(unsigned* o) {{
  __shared__ uint4 v[{128 * w}];
  const unsigned t = threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 16; ++k) {{
    v[t * {w}] = make_uint4(t, k, t, k);
    __syncthreads();
    const uint4 x = v[((t + 1) & 127u) * {w}];
    s += x.x;
    __syncthreads();
  }}
  o[blockIdx.x] = s;
}}
"""
        _source_kernel(out, f"ms_s_d{d}", src, (96, 1, 1), (128,), [1 * MB], "hand", g, written=1024)
        src = f"""// round 16: copies of 16 bytes per thread, rows {d} bytes a thread
__global__ void __launch_bounds__(128) k_mem(unsigned* o, const char* a) {{
  __shared__ __align__(128) unsigned char buf[32768];
  const unsigned sm = (unsigned)__cvta_generic_to_shared(buf);
  const unsigned t = threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {{
    const char* p = a + ((size_t)(blockIdx.x * 128 + t) * 8 + k) * {d};
    const unsigned at = sm + (t * 16 + k * 2048);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(at), "l"(p) : "memory");
  }}
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  o[blockIdx.x] = t;
}}
"""
        _source_kernel(out, f"ms_c_d{d}", src, (24, 1, 1), (128,), [1 * MB, 8 * MB], "hand", g, written=1024)


def _mem_l1(out: Path) -> None:
    """L1 capacity and replacement: one or two blocks per SM, 8 passes of a region of P KiB."""
    g = "mem_l1"
    ty = ("uint4", "x.x ^ x.y ^ x.z ^ x.w")
    for p in (16, 32, 64, 96, 128, 160, 192, 256, 384):
        n = p * 1024 // (256 * 16)
        load = f"a[blockIdx.x * {p * 64} + threadIdx.x * {n} + k % {n}u]"
        _mem_read(out, f"ml_p{p}", g, f"8 passes of {p} KiB, a block per SM", 256, 8 * n, load, 48, 48 * p * 1024, ty=ty)
    for p in (32, 64, 96, 128):
        n = p * 1024 // (128 * 16)
        load = f"a[blockIdx.x * {p * 64} + threadIdx.x * {n} + k % {n}u]"
        _mem_read(out, f"ml2_p{p}", g, f"8 passes of {p} KiB, two blocks per SM", 128, 8 * n, load, 96, 96 * p * 1024, ty=ty)
    for p in (128, 192):
        n = p * 1024 // (256 * 16)
        load = f"a[blockIdx.x * {p * 64} + ((threadIdx.x + k) * {HASH} >> 8) % {p * 64}u]"
        _mem_read(out, f"ml_h_p{p}", g, f"8 passes hashed over {p} KiB", 256, 8 * n, load, 48, 48 * p * 1024, ty=ty)


def _mem_far(out: Path) -> None:
    """DRAM channels and banks by stride; hashed reach (L2, TLB); warm strides."""
    g = "mem_far"
    for s in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
        b = 32768 * s
        rot = 1 if b >= 64 * MB else _rot(b)
        _mem_read(out, f"md_s{s}", g, f"32 K lanes, {s} bytes apart", 128, 1, f"a[(size_t)t * {s // 8}]", 256, b, rot=rot)
    for f in (1, 4, 16, 32, 64, 256, 1024):
        b = f * MB
        _mem_read(out, f"mh_f{f}", g, f"256 K lanes hashed through {f} MiB", 128, 1, f"a[(t * {HASH}) >> 4 & {b // 8 - 1}u]", 2048, b)
    for f in (16, 64):
        b = f * MB
        _mem_read(out, f"mh_f{f}_t512", g, f"256 K lanes hashed through {f} MiB", 512, 1, f"a[(t * {HASH}) >> 4 & {b // 8 - 1}u]", 512, b)
    for s in (64, 128, 256, 1024, 4096):
        _mem_read(out, f"mst_s{s}", g, f"warm, 4 MiB, lanes {s} bytes apart", 128, 1, f"a[t * {s // 8}]", 4 * MB // (128 * s), 4 * MB)
    _mem_read(out, "mst_s256_t512", g, "warm, 4 MiB, lanes 256 bytes apart", 512, 1, "a[t * 32]", 4 * MB // (512 * 256), 4 * MB)


def _mem(out: Path) -> None:
    _mem_dram(out)
    _mem_threads(out)
    _mem_write(out)
    _mem_stride(out)
    _mem_l1(out)
    _mem_far(out)


def _hand16(out: Path) -> None:
    src_dir = Path(__file__).parent / "hand16"
    for name, (grid, block, sizes, written) in HAND16.items():
        text = (src_dir / f"{name}.cu").read_text()
        _source_kernel(
            out, f"bl16_{name}", text, grid, block, sizes, "hand", "breadth blind", written=written
        )
    for name, tag, axis, grid, sizes, written, subs in WI_HAND16:
        g0, block, s0, w0 = HAND16[name]
        text = (src_dir / f"{name}.cu").read_text()
        for old, new in subs:
            text = text.replace(old, new)
        if tag == "t256":
            text = text.replace("(128)", "(256)").replace("* 128", "* 256")
            block = (256,)
        _source_kernel(
            out,
            f"bl16_{name}_{tag}",
            text,
            grid or g0,
            block,
            sizes or s0,
            "hand",
            "breadth blind",
            written=written if written is not None else w0,
            whatif={"base": f"bl16_{name}", "axis": axis},
        )


def _round16(out: Path) -> None:
    _hand16(out)
    _mem(out)
    sh = HGEMM_BLIND16_SHAPE
    M, N, K = sh["M"], sh["N"], sh["K"]
    pre = f"hgemm_m{M}n{N}k{K}"
    for name, tile in HGEMM_BLIND14.items():
        for cache, tail in (("warm", ""), ("rotating", " rotating")):
            suffix = "" if cache == "warm" else "_rotating"
            base = f"{pre}_c128x128x32_s3_t128{suffix}"
            more = _wi(name, base, "base" if f"{pre}_{name}{suffix}" == base else "design")
            group = f"hgemm {M}x{N}x{K}{tail}"
            _tried(_hgemm, out, f"{pre}_{name}{suffix}", tile, sh, cache, set=group, **more)
    tag = f"{GEMV16['N']}x{GEMV16['K']}"
    for cache, tail in (("warm", ""), ("rotating", " rotating")):
        for v, (kw, axis) in GEMV13_V.items():
            more = _wi(v, f"gemv16_b_{cache}", axis)
            c = {**GEMV16, **kw}
            _tried(_gemv_row, out, f"gemv16_{v}_{cache}", c, cache, set=f"gemv {tag}{tail}", **more)
    tag = f"{FA16['S']}x{FA16['H']}x{FA16['D']}"
    for cache, tail in (("warm", ""), ("rotating", " rotating")):
        for name, (bm, bn, s, th) in FA_BLIND7_V.items():
            f = {**FA16, "BM": bm, "BN": bn, "stages": s, "threads": th}
            more = _wi(name, f"fa16_c64x64_s2_t128_{cache}", FA13_AXIS.get(name, "base"))
            _tried(_fa, out, f"fa16_{name}_{cache}", f, cache, set=f"fa {tag}{tail}", **more)
    tag = f"{DECODE16['H']}x{DECODE16['S']}x{DECODE16['D']}"
    for cache, tail in (("warm", ""), ("rotating", " rotating")):
        for v, (kw, axis) in DECODE13_V.items():
            more = _wi(v, f"dec16_b_{cache}", axis)
            c = {**DECODE16, **kw}
            _tried(_decode_part, out, f"dec16_{v}_{cache}", c, cache, set=f"decode {tag}{tail}", **more)
    tag = f"{ROWS16['M']}x{ROWS16['N']}"
    for th in THREADS13:
        axis = "base" if th == 128 else "threads"
        kw = {"threads": th}
        for cache, tail in (("warm", ""), ("rotating", " rotating")):
            more = _wi("", f"rms16_t128_{cache}", axis)
            _tried(_rms, out, f"rms16_t{th}_{cache}", ROWS16, cache, kw, set=f"rmsnorm {tag}{tail}", **more)
        more = _wi("", "sm16_t128_warm", axis)
        _tried(_softmax, out, f"sm16_t{th}_warm", ROWS16, kw, set=f"softmax {tag}", **more)


# round 17: the blind recipe (D874). Rule, chosen before any result: off HGEMM a factor 9 in every
# size, HGEMM a factor 27 in every size; tiles and divisors as TileLang accepts them.
GEMV17 = {"N": 7200, "K": 2592, "group": 72, "n_part": 4, "reduce": 32}
FA17 = {**FA, "S": 2016, "H": 9, "D": 72}
DECODE17 = {**DECODE, "H": 27, "G": 9, "S": 2160, "D": 72}
ROWS17 = {"M": 720, "N": 4608}
HGEMM_BLIND17_SHAPE = {"M": 3456, "N": 1296, "K": 1728}
HAND17 = {
    "atomrange": ((1024, 1, 1), (128,), [4 * MB], None),
    "blockmax": ((1024, 1, 1), (256,), [1 * MB, 4 * MB], None),
    "recordfive": ((512, 1, 1), (128,), [9 * MB], None),
    "unpackrows": ((2048, 1, 1), (128,), [9 * MB, 2 * MB], None),
    "strideloop": ((1024, 1, 1), (256,), [1 * MB, 16 * MB], None),
    "revcols": ((1024, 1, 1), (128,), [1 * MB, 4 * MB], None),
    "primefold": ((1024, 1, 1), (128,), [3 * MB, 1 * MB], None),
    "stackrun": ((1024, 1, 1), (128,), [1 * MB, 1 * MB], None),
    "stackmed": ((512, 1, 1), (256,), [1 * MB, 1 * MB], None),
    "threelevel": ((1024, 1, 1), (128,), [1 * MB, 1 * MB, 1 * MB, 4 * MB], None),
    "gathertwo": ((1024, 1, 1), (128,), [1 * MB, 1 * MB, 3 * MB], None),
    "fivepoint": ((2048, 1, 1), (128,), [4 * MB, 4 * MB], None),
    "addput": ((1024, 1, 1), (128,), [4 * MB], None),
}
WI_HAND17 = (
    ("strideloop", "t128", "threads", (2048, 1, 1), None, None, ()),
    ("primefold", "k2", "stores removed", None, None, None, (("k < 3", "k < 2"),)),
    ("addput", "k2", "stores removed", None, None, None, (("k < 3", "k < 2"),)),
    ("gathertwo", "one", "loads removed", None, None, None, (("b[i] + b[i + 1] + b[j] + b[j + 256]", "b[i] + b[j]"),)),
)
TWIN = """// round 17: {what}
__global__ void __launch_bounds__({t}) k_mem(unsigned* o, const uint2* a) {{
  const unsigned t = blockIdx.x * {t} + threadIdx.x;
  const uint2 x = a[{first}];
  unsigned s = t;
  {gap}
  const uint2 y = a[{second}];
  o[blockIdx.x & 65535u] = x.x ^ y.y ^ s;
}}
"""
BUSY = 'for (unsigned d = 0; d < {g}; ++d) {{ s += d; asm volatile("" : "+r"(s)); }}'
TWO_BLOCKS = """// round 17: {what}
__global__ void __launch_bounds__(128) k_mem(unsigned* o, const uint2* a) {{
  const unsigned p = {pair}, side = {side};
  unsigned s = 0;
  for (unsigned k = 0; k < 16; ++k) {{
    const uint2 x = a[(p * 128 + threadIdx.x) * 128 + k * 8 + side];
    s ^= x.x ^ x.y;
  }}
  o[blockIdx.x & 65535u] = s;
}}
"""
GEMV_ROWS = """// round 17: GEMV's load: each warp a row, {w} bytes per lane, pitch {p}
__global__ void __launch_bounds__({t}) k_mem(unsigned* o, const char* a{more}) {{
  const unsigned row = blockIdx.x * {np} + threadIdx.y, kr = threadIdx.x;
  unsigned s = 0;
  for (unsigned ko = 0; ko < {n}; ++ko) {{
    const size_t at = (size_t)row * {p} + (ko * {r} + kr) * {w};
    s ^= *(const {ty}*)(a + at);{beside}
  }}
  if (kr == 0) o[row & 65535u] = s;
}}
"""
TURNOVER = """// round 17: block turnover: compute for about {us} us, {what}
__global__ void __launch_bounds__({t}) k_mem(unsigned* o) {{
{shared}  unsigned s = threadIdx.x;
  for (unsigned k = 0; k < {n}; ++k) {{ s = s * 3u + k; asm volatile("" : "+r"(s)); }}
  if (s == 0x7fff1234u) o[0] = s;
}}
"""


def _twins(out: Path) -> None:
    """Question 52: read × DRAM: when and by whom a burst's second sector is asked."""
    g = "twin"
    F = 8 * MB
    for gap in (0, 1, 8, 32, 128, 512):
        busy = BUSY.format(g=gap) if gap else ""
        src = TWIN.format(what=f"twin sectors, {gap} passes apart", t=128, first="t * 8", gap=busy, second="t * 8 + 4")
        _source_kernel(out, f"tw_g{gap}", src, (F // (128 * 64), 1, 1), (128,), [1 * MB, F], "hand", g, rot=_rot(F), written=1024)
    src = TWIN.format(what="the twin after the first sector's data", t=128, first="t * 8", gap="if (x.x == 0x7fff1234u) s = 1;", second="t * 8 + 4")
    _source_kernel(out, "tw_dep", src, (F // (128 * 64), 1, 1), (128,), [1 * MB, F], "hand", g, rot=_rot(F), written=1024)
    src = TWIN.format(what="two sectors of bursts far apart", t=128, first="t * 8", gap="", second="(t ^ 65536u) * 8 + 4")
    _source_kernel(out, "tw_far", src, (F // (128 * 64), 1, 1), (128,), [1 * MB, F], "hand", g, rot=_rot(F), written=1024)
    src = TWIN.format(what="two sectors of a line, not of a burst", t=128, first="t * 16", gap="", second="t * 16 + 8")
    _source_kernel(out, "tw_l", src, (F // (128 * 128), 1, 1), (128,), [1 * MB, F], "hand", g, rot=_rot(F), written=1024)
    for t in (32, 512):
        for gap in (0, 32):
            busy = BUSY.format(g=gap) if gap else ""
            src = TWIN.format(what=f"twin sectors at T {t}, {gap} passes apart", t=t, first="t * 8", gap=busy, second="t * 8 + 4")
            _source_kernel(out, f"tw_t{t}_g{gap}", src, (F // (t * 64), 1, 1), (t,), [1 * MB, F], "hand", g, rot=_rot(F), written=1024)
    src = f"""// round 17: twin sectors by neighbouring warps of a block
__global__ void __launch_bounds__(128) k_mem(unsigned* o, const uint2* a) {{
  const unsigned l = threadIdx.x & 31u, w = threadIdx.x >> 5;
  const uint2 x = a[(blockIdx.x * 64 + (w >> 1) * 32 + l) * 8 + (w & 1u) * 4];
  o[blockIdx.x & 65535u] = x.x ^ x.y;
}}
"""
    _source_kernel(out, "tw_sw", src, (F // (64 * 64), 1, 1), (128,), [1 * MB, F], "hand", g, rot=_rot(F), written=1024)
    pairs = (
        ("tw_ss", "twin sectors by two blocks on an SM", "blockIdx.x % 48u", "(blockIdx.x / 48u) * 4"),
        ("tw_sc", "twin sectors by blocks on two SMs", "blockIdx.x / 2u", "(blockIdx.x & 1u) * 4"),
        ("tw_s0", "first sectors alone, as many bursts as blocks ask", "blockIdx.x", "0"),
    )
    for name, what, pair, side in pairs:
        src = TWO_BLOCKS.format(what=what, pair=pair, side=side)
        _source_kernel(out, name, src, (96, 1, 1), (128,), [1 * MB, 16 * MB], "hand", g, rot=_rot(16 * MB), written=1024)


def _gemv_rows(out: Path) -> None:
    """Question 52: GEMV's rotating read without its arithmetic: width × pitch, warps × lanes."""
    g = "gemv_rows"
    tys = {2: "unsigned short", 4: "unsigned", 8: "unsigned long long"}

    def row(name, w, p, r=32, np=4, more="", beside="", sizes=()):
        n = -(-p // (r * w))
        src = GEMV_ROWS.format(w=w, p=p, t=r * np, more=more, np=np, n=n, r=r, ty=tys[w], beside=beside)
        grid = 8 * MB // (np * p)
        _source_kernel(out, name, src, (grid, 1, 1), (r, np), [1 * MB, 9 * MB, *sizes], "hand", g, rot=_rot(9 * MB), written=1024)

    for w in (2, 4, 8):
        for p in (1280, 1344, 1360):
            row(f"gr_w{w}_p{p}", w, p)
    for r, np in ((16, 4), (64, 4), (32, 2), (32, 8)):
        row(f"gr_r{r}_np{np}", 4, 1344, r, np)
    sc = "\n    s ^= c[row * 48 + (ko * 256 + kr * 8) / 56];"
    row("gr_sc", 4, 1344, more=", const unsigned short* c", beside=sc, sizes=(1 * MB,))


def _part_sectors(out: Path) -> None:
    """Question 53: write and read-modify-write × L2: lanes against sectors and lines."""
    g = "part_sector"
    for f in (4, 12):
        F = f * MB
        for s in (8, 16, 32, 64, 128):
            n = s // 8
            idx = f"t * {n} + k"
            src = MEM_WRITE.format(what=f"8-byte lanes {s} bytes apart, sectors filled", t=128, n=n, more="", body=f"((uint2*)o)[{idx}] = make_uint2(t, k);")
            _source_kernel(out, f"pw_f{f}_s{s}", src, (F // (128 * s), 1, 1), (128,), [F], "hand", g)
            if s > 8:
                src = MEM_WRITE.format(what=f"8-byte lanes {s} bytes apart, one store", t=128, n=1, more="", body=f"((uint2*)o)[t * {n}] = make_uint2(t, k);")
                _source_kernel(out, f"pw_f{f}_s{s}_one", src, (F // (128 * s), 1, 1), (128,), [F], "hand", g)
    for s in (16, 32, 128):
        src = MEM_WRITE.format(what=f"4-byte lanes {s} bytes apart, one store", t=128, n=1, more="", body=f"o[t * {s // 4}] = t;")
        _source_kernel(out, f"pw4_f4_s{s}_one", src, (4 * MB // (128 * s), 1, 1), (128,), [4 * MB], "hand", g)
    for s in (16, 32, 64, 128):
        n = s // 8
        src = MEM_WRITE.format(what=f"8-byte lanes {s} bytes apart read and written in place", t=128, n=n, more="", body=f"((uint2*)o)[t * {n} + k].x += 1u;")
        _source_kernel(out, f"px_f4_s{s}", src, (4 * MB // (128 * s), 1, 1), (128,), [4 * MB], "hand", g)


def _rows_warm(out: Path) -> None:
    """Rows warm, flat in threads at 12.7 µs: each part of the kernel, then by rows; RMSNorm's own
    shape of round 16, a seen shape."""
    g = "rows_warm"
    for t in (32, 128, 512):
        for tag, kw in (("nostore", {"store": "shared"}), ("noload", {"load": False}), ("nocompute", {"compute": False})):
            _tried(_rms, out, f"rr_{tag}_t{t}_warm", ROWS16, "warm", {"threads": t, **kw}, set=g)
    for m in (224, 448, 1792):
        _tried(_rms, out, f"rr_m{m}_t128_warm", {"M": m, "N": ROWS16["N"]}, "warm", {"threads": 128}, set=g)


def _turnover(out: Path) -> None:
    """Warm rows sit 2 to 5 µs under by waves of blocks: blocks that follow blocks on an SM."""
    g = "turnover"
    big = "  __shared__ unsigned buf[20000];\n  if (threadIdx.x == 0) buf[0] = blockIdx.x;\n"
    for t in (32, 128):
        for grid in (48, 96, 192, 384):
            for tag, sh in (("one", big), ("many", "")):
                src = TURNOVER.format(us=2, what=f"{grid} blocks of {t}, {tag} on an SM", t=t, shared=sh, n=400)
                _source_kernel(out, f"tv_{tag}_t{t}_g{grid}", src, (grid, 1, 1), (t,), [1 * MB], "hand", g, written=1024)


def _fa_stage(out: Path) -> None:
    """One stage against two at FA's seen shape: which part keeps the stage's row 20 % under."""
    g = "fa_stage"
    for s in (1, 2):
        base = {**FA16, "BM": 64, "BN": 64, "stages": s, "threads": 128}
        for tag, kw in (("nov", {"second": False}), ("noexp", {"exp": False}), ("shared", {"store": "shared"})):
            more = {"writes": ()} if tag == "shared" else {}
            _tried(_fa, out, f"fs_{tag}_s{s}_warm", {**base, **kw}, "warm", set=g, **more)


def _hand17(out: Path) -> None:
    src_dir = Path(__file__).parent / "hand17"
    for name, (grid, block, sizes, written) in HAND17.items():
        text = (src_dir / f"{name}.cu").read_text()
        _source_kernel(
            out, f"bl17_{name}", text, grid, block, sizes, "hand", "breadth blind", written=written
        )
    for name, tag, axis, grid, sizes, written, subs in WI_HAND17:
        g0, block, s0, w0 = HAND17[name]
        text = (src_dir / f"{name}.cu").read_text()
        for old, new in subs:
            text = text.replace(old, new)
        if tag == "t128":
            text = text.replace("(256)", "(128)").replace("* 256", "* 128")
            block = (128,)
        _source_kernel(
            out,
            f"bl17_{name}_{tag}",
            text,
            grid or g0,
            block,
            sizes or s0,
            "hand",
            "breadth blind",
            written=written if written is not None else w0,
            whatif={"base": f"bl17_{name}", "axis": axis},
        )


def _round17(out: Path) -> None:
    _hand17(out)
    _twins(out)
    _gemv_rows(out)
    _part_sectors(out)
    _rows_warm(out)
    _turnover(out)
    _fa_stage(out)
    sh = HGEMM_BLIND17_SHAPE
    M, N, K = sh["M"], sh["N"], sh["K"]
    pre = f"hgemm_m{M}n{N}k{K}"
    for name, tile in HGEMM_BLIND14.items():
        for cache, tail in (("warm", ""), ("rotating", " rotating")):
            suffix = "" if cache == "warm" else "_rotating"
            base = f"{pre}_c128x128x32_s3_t128{suffix}"
            more = _wi(name, base, "base" if f"{pre}_{name}{suffix}" == base else "design")
            group = f"hgemm {M}x{N}x{K}{tail}"
            _tried(_hgemm, out, f"{pre}_{name}{suffix}", tile, sh, cache, set=group, **more)
    tag = f"{GEMV17['N']}x{GEMV17['K']}"
    for cache, tail in (("warm", ""), ("rotating", " rotating")):
        for v, (kw, axis) in GEMV13_V.items():
            more = _wi(v, f"gemv17_b_{cache}", axis)
            c = {**GEMV17, **kw}
            _tried(_gemv_row, out, f"gemv17_{v}_{cache}", c, cache, set=f"gemv {tag}{tail}", **more)
    tag = f"{FA17['S']}x{FA17['H']}x{FA17['D']}"
    for cache, tail in (("warm", ""), ("rotating", " rotating")):
        for name, (bm, bn, s, th) in FA_BLIND7_V.items():
            f = {**FA17, "BM": bm, "BN": bn, "stages": s, "threads": th}
            more = _wi(name, f"fa17_c64x64_s2_t128_{cache}", FA13_AXIS.get(name, "base"))
            _tried(_fa, out, f"fa17_{name}_{cache}", f, cache, set=f"fa {tag}{tail}", **more)
    tag = f"{DECODE17['H']}x{DECODE17['S']}x{DECODE17['D']}"
    for cache, tail in (("warm", ""), ("rotating", " rotating")):
        for v, (kw, axis) in DECODE13_V.items():
            more = _wi(v, f"dec17_b_{cache}", axis)
            c = {**DECODE17, **kw}
            _tried(_decode_part, out, f"dec17_{v}_{cache}", c, cache, set=f"decode {tag}{tail}", **more)
    tag = f"{ROWS17['M']}x{ROWS17['N']}"
    for th in THREADS13:
        axis = "base" if th == 128 else "threads"
        kw = {"threads": th}
        for cache, tail in (("warm", ""), ("rotating", " rotating")):
            more = _wi("", f"rms17_t128_{cache}", axis)
            _tried(_rms, out, f"rms17_t{th}_{cache}", ROWS17, cache, kw, set=f"rmsnorm {tag}{tail}", **more)
        more = _wi("", "sm17_t128_warm", axis)
        _tried(_softmax, out, f"sm17_t{th}_warm", ROWS17, kw, set=f"softmax {tag}", **more)


def main(out: Path) -> None:
    _hand(out)
    _round11(out)
    _stress(out)
    _wq(out)
    for name, tile in HGEMM.items():
        # round 4: each variant also rotating (cold operands; the replay's DRAM rule without the
        # warm L2 question)
        _hgemm(out, f"hgemm2048_{name}", tile, SHAPE, "warm")
        _hgemm(out, f"hgemm2048_{name}_rotating", tile, SHAPE, "rotating")
    _fa(out, "fa_fwd_s2048_h8_d128", FA, estimate=True)
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
        ("rows8", _rows8),
        ("nvfp4", _nvfp4),
        ("blind9", _blind9),
        ("round12", _round12),
        ("round12b", _round12b),
        ("round13", _round13),
        ("round13b", _round13b),
        ("round14", _round14),
        ("round15", _round15),
        ("round15b", _round15b),
        ("round16", _round16),
        ("round17", _round17),
    ):
        try:  # a kernel that fails to generate is reported; the others stand
            gen(out)
        except Exception:  # noqa: BLE001 - reported, the batch keeps the other kernels
            print(f"FAILED {name}")
            traceback.print_exc()


def _decode(out: Path) -> None:
    c = DECODE
    B, H, S, D, sp = c["B"], c["H"], c["S"], c["D"], c["split"]
    for cache in ("warm", "rotating"):
        _decode_part(out, f"decode_gqa_part_s{S}_{cache}", c, cache)
    info = {"family": "decode_gqa_combine", **_decode_info(c), "cache": "warm"}
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


def _hgemm_group(out: Path, sh: dict, name: str, tile: tuple) -> None:
    """A warm and a rotating row of a blind group of HGEMM variants."""
    M, N, K = sh["M"], sh["N"], sh["K"]
    tag = f"hgemm_m{M}n{N}k{K}_{name}"
    _hgemm(out, tag, tile, sh, "warm", set=f"hgemm {M}x{N}x{K}")
    _hgemm(out, f"{tag}_rotating", tile, sh, "rotating", set=f"hgemm {M}x{N}x{K} rotating")


def _blind(out: Path) -> None:
    """Round 6: rows no engine change has seen."""
    for (M, N, K), variants in HGEMM_BLIND.items():
        for name, tile in variants.items():
            _hgemm_group(out, {"M": M, "N": N, "K": K}, name, tile)
    f = FA_BLIND
    _fa(out, f"fa_fwd_s{f['S']}_h{f['H']}_d{f['D']}", f, estimate=True)
    M, N = RMS_BLIND["M"], RMS_BLIND["N"]
    for cache in ("warm", "rotating"):
        _rms(out, f"rmsnorm_m{M}_n{N}_{cache}", RMS_BLIND, cache)
    M, N = SOFTMAX_BLIND["M"], SOFTMAX_BLIND["N"]
    _softmax(out, f"softmax_m{M}_n{N}_warm", SOFTMAX_BLIND)


def _wi(name: str, base: str, axis: str) -> dict:
    return {} if axis == "base" else {"whatif": {"base": base, "axis": axis}}


# round 10: kernels from outside TileLang, first-scored blind, a separate breadth set
# name: (grid, block, smem, [buffer bytes, read or written last]). Sources in testset/hand/
HAND = {
    "triad": ((1024, 1, 1), (256,), [64 * MB, 64 * MB, 64 * MB]),
    "stencil": ((128, 128, 1), (16, 16), [16 * MB, 16 * MB]),
    "nbody": ((64, 1, 1), (256,), [1 * MB, 1 * MB]),
    "hist": ((1024, 1, 1), (256,), [4096, 16 * MB]),
    "oddeven": ((1024, 1, 1), (128,), [4 * MB, 4 * MB]),
    "mandel": ((32, 32, 1), (16, 16), [1 * MB]),
    "dotd": ((512, 1, 1), (256,), [2 * MB, 64 * MB, 64 * MB]),
    "hashmix": ((512, 1, 1), (256,), [1 * MB, 32 * MB]),
    "sobel": ((127, 127, 1), (16, 16), [16 * MB, 4 * MB]),
    "bcast": ((512, 1, 1), (256,), [1 * MB, 32 * MB]),
}


def _source_kernel(
    out: Path,
    name: str,
    src: str,
    grid,
    block,
    sizes,
    family: str,
    group: str,
    rot: int = 1,
    written: int | None = None,
    **more,
):
    """A kernel with its own source: `k_*`, buffers in the order of its parameters (the first is
    written), random 16-bit words in every buffer; `rot` sets of them used in turn."""
    if not _wanted(name):
        return
    entry = re.search(r"void(?: __launch_bounds__\([^)]*\))? (k_\w+)\(", src)[1]
    sig = re.search(rf"{entry}\(([^)]*)\)", src)[1]
    types = [p.rsplit(" ", 1)[0] for p in sig.split(",")]
    args = ", ".join(f"({t})b[{i}]" for i, t in enumerate(types))
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "kernel.cu").write_text(src.replace("__global__", 'extern "C" __global__', 1))
    g, b = (", ".join(str(x) for x in v) for v in (grid, block))
    (d / "harness.cu").write_text(
        HARNESS
        % {
            "includes": '#include <tl_templates/cuda/common.h>\n#include "kernel.cu"',
            "sizes": ", ".join(str(s) for s in sizes),
            "nbuf": len(sizes),
            "rot": rot,
            "setup": f"  const int smem = 0;\n  dim3 grid({g}), block({b});",
            "launches": f"    {entry}<<<grid, block, smem, st>>>({args});",
            "name": name,
        }
    )
    launch = {
        "entry": entry,
        "grid": list(grid),
        "threads": math.prod(block),
        "block": list(block),
        "smem": 0,
        "rot": rot,
        "footprint_bytes": sum(sizes),
        "write_bytes": sizes[0] if written is None else written,
        "family": family,
        "set": group,
        "cache": "rotating" if rot > 1 else "warm",
        **more,
    }
    (d / "launch.json").write_text(json.dumps(launch, indent=1) + "\n")


def _hand(out: Path) -> None:
    src = Path(__file__).parent / "hand"
    for name, (grid, block, sizes) in HAND.items():
        text = (src / f"{name}.cu").read_text()
        _source_kernel(out, f"hand_{name}", text, grid, block, sizes, "hand", "breadth hand")


def _stress(out: Path) -> None:
    """Round 10: random instruction mixes and access patterns, seeded (the set's own worst rows
    say where the model is worst)."""
    rng = random.Random(1010)
    stmts = {
        "fadd": "f{a} = f{b} + f{c};",
        "fmul": "f{a} = f{b} * 0.999f + f{c};",
        "fdiv": "f{a} = f{b} / (f{c} + 1.5f);",
        "fsqrt": "f{a} = sqrtf(fabsf(f{b}) + 1.f);",
        "fexp": "f{a} = expf(f{b} * 0.001f);",
        "imul": "u{a} = u{b} * 2654435761u + u{c};",
        "ixor": "u{a} = (u{b} ^ (u{c} >> 7)) + (u{a} << 3);",
        "i2f": "f{a} = (float)u{b} * 1e-9f;",
        "f2i": "u{a} = (unsigned)fabsf(f{b}) + u{c};",
        "sel": "f{a} = (u{b} & 1) ? f{b} : f{c};",
        "ld1": "u{a} ^= in[(base + (size_t)k * {s1}) & MASK];",
        "ld2": "f{a} += (float)in[(base * {s2} + (size_t)k * 4099) & MASK];",
        "ldx": "u{a} = in[(base ^ u{b}) & MASK];",
        "lds": "f{a} += s[(t + k) & 63];",
        "sts": "s[t & 63] = f{a};",
        "stg": "out[(base + (size_t)k * {s1}) & OMASK] = u{a};",
    }
    for i in range(16):
        keys = rng.choices(sorted(stmts), k=rng.randint(4, 16))
        body = []
        for key in keys:
            body.append(
                "    "
                + stmts[key].format(
                    a=rng.randrange(4),
                    b=rng.randrange(4),
                    c=rng.randrange(4),
                    s1=rng.choice((1, 8, 64, 256, 1024)),
                    s2=rng.choice((3, 17, 97)),
                )
            )
        nbody = "\n".join(body)
        threads = rng.choice((64, 128, 256))
        trips = rng.choice((8, 16, 32))
        src = f"""// generated, seed 1010, kernel {i}
#define MASK ((1u << 22) - 1)
#define OMASK ((1u << 18) - 1)
__global__ void __launch_bounds__({threads}) k_stress{i}(unsigned* out, const unsigned* in) {{
  __shared__ float s[64];
  const int t = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * {threads} + t;
  float f0 = t, f1 = 1.f, f2 = 2.f, f3 = 3.f;
  unsigned u0 = t, u1 = 1, u2 = 2, u3 = 3;
  for (int k = 0; k < {trips}; ++k) {{
{nbody}
  }}
  s[t & 63] = f0 + f1 + f2 + f3;
  out[base & OMASK] = u0 ^ u1 ^ u2 ^ u3 ^ (unsigned)s[0];
}}
"""
        grid = (rng.choice((48, 192, 768)), 1, 1)
        _source_kernel(
            out,
            f"stress_{i:02d}",
            src,
            grid,
            (threads,),
            [1 << 20, 1 << 24],
            "stress",
            "breadth stress",
        )


# round 10, question 36: K 32 against K 64 at t256, each with one part removed; question 32: s2
# against s3 at 3072 x 1536 x 2048 with the same parts removed
WQ = {
    "k32_s3_t256": (128, 256, 32, 3, 256),
    "k64_s2_t256": (128, 256, 64, 2, 256),
    "k32_s2_t256": (128, 256, 32, 2, 256),
    "k64_s3_t256": (128, 256, 64, 3, 256),
}
WQ_PARTS = {
    "nocompute": ({"compute": False}, "compute removed"),
    "nostore": ({"store": "shared"}, "stores removed"),
}


def _wq(out: Path) -> None:
    for tag, tile in WQ.items():
        base = f"wq_{tag}"
        _hgemm(out, base, (*tile, 0), SHAPE, "warm", {}, set="whatif hgemm warm")
        for v, (opt, axis) in WQ_PARTS.items():
            more = {"whatif": {"base": base, "axis": axis}}
            _hgemm(
                out, f"{base}_{v}", (*tile, 0), SHAPE, "warm", opt, set="whatif hgemm warm", **more
            )
    sh = {"M": 3072, "N": 1536, "K": 2048}
    for st in (2, 3):
        base = f"wq_m3072_s{st}"
        tile = (128, 128, 32, st, 128, 0)
        _hgemm(out, base, tile, sh, "warm", {}, set="whatif hgemm warm")
        for v, (opt, axis) in WQ_PARTS.items():
            more = {"whatif": {"base": base, "axis": axis}}
            _hgemm(out, f"{base}_{v}", tile, sh, "warm", opt, set="whatif hgemm warm", **more)


def _whatif(out: Path) -> None:
    """Round 7: one change at a time from a base kernel, each measured and replayed from its own
    compiled code; the base is in the same batch."""
    for table, tag in ((WI_HGEMM, "b"), (WI_HGEMM2, "L_s3"), (WI_HGEMM3, "S_s4")):
        for cache in ("warm", "rotating"):
            for v, (*tile, opt, axis) in table.items():
                more = _wi(v, f"wi_hg_{tag}_{cache}", axis)
                name, group = f"wi_hg_{v}_{cache}", f"whatif hgemm {cache}"
                _hgemm(out, name, tuple(tile), SHAPE, cache, opt, set=group, **more)
    for v, (kw, axis) in WI_FA.items():
        kw = dict(kw)
        base = kw.pop("base", "b")
        for cache in ("warm", "rotating"):
            if cache == "rotating" and v not in ("b", "s1", "s3", "nostore"):
                continue
            more = _wi(v, f"wi_fa_{base}_{cache}", axis)
            writes = () if kw.get("store") else (-1,)
            f = {**FA, **kw}
            _fa(
                out, f"wi_fa_{v}_{cache}", f, cache, False, writes, set=f"whatif fa {cache}", **more
            )
    for v, (kw, axis) in WI_GEMV.items():
        for cache in ("warm", "rotating"):
            more = _wi(v, f"wi_gemv_b_{cache}", axis)
            group = f"whatif gemv {cache}"
            _gemv_row(out, f"wi_gemv_{v}_{cache}", {**GEMV, **kw}, cache, set=group, **more)
    for v, (kw, axis) in WI_RMS.items():
        for cache in ("warm", "rotating"):
            more = _wi(v, f"wi_rms_b_{cache}", axis)
            group = f"whatif rmsnorm {cache}"
            _rms(out, f"wi_rms_{v}_{cache}", ROWS, cache, kw, set=group, **more)
    for v, (kw, axis) in WI_SOFTMAX.items():
        more = _wi(v, "wi_softmax_b_warm", axis)
        _softmax(out, f"wi_softmax_{v}_warm", ROWS, kw, set="whatif softmax warm", **more)
    for v, (kw, axis) in WI_DECODE.items():
        for cache in ("warm", "rotating"):
            more = _wi(v, f"wi_dec_b_{cache}", axis)
            group = f"whatif decode {cache}"
            _decode_part(out, f"wi_dec_{v}_{cache}", {**DECODE, **kw}, cache, set=group, **more)
    for x in WI_ROT:
        more = _wi("", "wi_dec_b_rotating", "rotation depth")
        group = "whatif decode rotating"
        _decode_part(out, f"wi_dec_rot{x}_rotating", DECODE, "rotating", x, set=group, **more)
        more = _wi("", "wi_rms_b_rotating", "rotation depth")
        group = "whatif rmsnorm rotating"
        _rms(out, f"wi_rms_rot{x}_rotating", ROWS, "rotating", None, x, set=group, **more)
    for v, (kw, axis) in WI_W4.items():
        more = _wi(v, "wi_w4_b_warm", axis)
        tile = W4GEMM["c128x128x32_s3_t128"]
        _w4(out, f"wi_w4_{v}_warm", tile, "warm", kw, set="whatif w4gemm warm", **more)


def _blind7(out: Path) -> None:
    """Round 7: larger variant groups at a shape no engine has seen, for a blind Kendall tau."""
    for name, tile in HGEMM_BLIND7.items():
        _hgemm_group(out, HGEMM_BLIND7_SHAPE, name, tile)
    b = FA_BLIND7
    for name, (bm, bn, s, th) in FA_BLIND7_V.items():
        f = {**b, "BM": bm, "BN": bn, "stages": s, "threads": th}
        try:
            cid = f"fa_s{b['S']}_h{b['H']}_d{b['D']}_{name}"
            _fa(out, cid, f, set=f"fa {b['S']}x{b['H']}x{b['D']}")
        except Exception:  # noqa: BLE001 - a tile TileLang refuses leaves the group
            print(f"FAILED fa {name}")


def _blind9(out: Path) -> None:
    """Round 9: a new shape for the blind score; predictions committed before the run."""
    for name, tile in HGEMM_BLIND9.items():
        _hgemm_group(out, HGEMM_BLIND9_SHAPE, name, tile)


def _gemv(out: Path) -> None:
    N, K = GEMV["N"], GEMV["K"]
    for cache in ("warm", "rotating"):
        _gemv_row(out, f"gemv_w4a16_n{N}_k{K}_{cache}", GEMV, cache)


def _rows(out: Path) -> None:
    M, N = ROWS["M"], ROWS["N"]
    for cache in ("warm", "rotating"):
        _rms(out, f"rmsnorm_m{M}_n{N}_{cache}", ROWS, cache)
    _softmax(out, f"softmax_m{M}_n{N}_warm", ROWS)


def _fp8(out: Path) -> None:
    M, N, K = SHAPE["M"], SHAPE["N"], SHAPE["K"]
    for name, (bm, bn, bk, s, th) in FP8.items():
        info = {
            "family": "fp8gemm",
            "shape": SHAPE,
            "pick": _pick(bm, bn, bk, s, th),
            "elem_bytes": 1,
            "smem_estimate": s * (bm + bn) * bk,
            "cache": "warm",
        }
        write(
            out,
            f"fp8gemm2048_{name}",
            fp8gemm(bm, bn, bk, s, th),
            [M * K, N * K, M * N],
            _tiles(M, N, bm, bn),
            th,
            info,
            elem_bytes=[1, 1, 2],
        )


def _moe(out: Path) -> None:
    c = MOE
    E, T_, N, K = c["E"], c["T"], c["N"], c["K"]
    for name, (bm, bn, bk, s, th) in MOE_TILES.items():
        info = {"family": "moe_gemm", "shape": c, "pick": _pick(bm, bn, bk, s, th), "cache": "warm"}
        write(
            out,
            f"moe_e{E}_t{T_}_n{N}_k{K}_{name}",
            moe_gemm(E, T_, N, K, bm, bn, bk, s, th),
            [T_ * K, E * K * N, T_ * N],
            _tiles(T_, N, bm, bn),
            th,
            info,
        )


def _w4gemm(out: Path) -> None:
    for name, tile in W4GEMM.items():
        for cache in ("warm", "rotating"):
            _w4(out, f"w4gemm2048_{name}_{cache}", tile, cache)


def _rows8(out: Path) -> None:
    for m in WI_ROWS_M:
        sh = {"M": m, "N": ROWS["N"]}
        for v, (kw, axis) in WI_RMSB.items():
            more = _wi(v, f"rmsb_m{m}_b", axis)
            _rms(out, f"rmsb_m{m}_{v}", sh, "warm", kw, set=f"whatif rmsnorm m{m}", **more)
        for v, (kw, axis) in (("b", ({}, "base")), ("nostore", WI_RMSB["nostore"])):
            more = _wi(v, f"smb_m{m}_b", axis)
            _softmax(out, f"smb_m{m}_{v}", sh, kw, set=f"whatif softmax m{m}", **more)


def _nv(out: Path, name: str, tile: tuple, cache: str) -> None:
    bm, bn, bk, s, th = tile
    M, N, K = SHAPE["M"], SHAPE["N"], SHAPE["K"]
    sizes, eb = [M * K, N * K // 2, N * K // NVFP4_BLOCK, M * N], [2, 1, 2, 2]
    write(
        out,
        name,
        nvfp4gemm(bm, bn, bk, s, th),
        sizes,
        _tiles(M, N, bm, bn),
        th,
        {
            "family": "nvfp4gemm",
            "shape": SHAPE,
            "pick": _pick(bm, bn, bk, s, th),
            "cache": cache,
        },
        elem_bytes=eb,
        rot=_rots(cache, sum(x * b for x, b in zip(sizes, eb, strict=True))),
    )


def _nvfp4(out: Path) -> None:
    for name, tile in NVFP4.items():
        for cache in ("warm", "rotating"):
            _nv(out, f"nvfp4gemm2048_{name}_{cache}", tile, cache)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
