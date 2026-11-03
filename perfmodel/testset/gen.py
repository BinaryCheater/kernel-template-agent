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
"""

import json
import math
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
GEMV = {"N": 4096, "K": 4096, "group": 128, "n_part": 4, "reduce": 32}
ROWS = {"M": 512, "N": 4096}


def hgemm(BM: int, BN: int, BK: int, S: int, threads: int, panel: int):
    M, N, K = SHAPE["M"], SHAPE["N"], SHAPE["K"]

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
            if panel:
                T.use_swizzle(panel_size=panel)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=S):
                T.copy(A[by * BM, ko * BK], A_shared)
                T.copy(B[ko * BK, bx * BN], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * BM, bx * BN])

    return gemm


def fa_forward(B: int, S: int, H: int, D: int, BM: int, BN: int, stages: int, threads: int):
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
                    sc[i] = T.exp2(m_prev[i] * scale - m[i] * scale)
                for i, j in T.Parallel(BM, D):
                    acc_o[i, j] *= sc[i]
                for i, j in T.Parallel(BM, BN):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - m[i] * scale)
                T.copy(acc_s, acc_s_cast)
                T.copy(V[bz, k * BN : (k + 1) * BN, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, ssum, dim=1)
                for i in T.Parallel(BM):
                    lsum[i] = lsum[i] * sc[i] + ssum[i]
            for i, j in T.Parallel(BM, D):
                acc_o[i, j] /= lsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Out[bz, bx * BM : (bx + 1) * BM, by, :])

    return fa


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


def gemv_w4a16(N: int, K: int, group: int, n_part: int, reduce: int):
    """TileLang's dequantize-GEMV pattern (examples/dequantize_gemm/example_dequant_gemv_fp16xint4):
    y = W x with W int4 (unsigned, offset 8) packed two per int8 along K, fp16 scale per group of
    `group` K elements; each warp one output row, 8 fp16 / 4 packed bytes per lane per step."""
    mk = 8  # fp16 per 16-byte load
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
                for ki in T.serial(mk):
                    bd[ki] = (
                        T.Cast("float16", (T.Cast("int32", bq[ki // 2]) >> ((ki % 2) * 4)) & 15)
                        - T.float16(8)
                    ) * Sc[bx * n_part + ni, (ko * bk + kr * mk) // group]
                for ki in T.serial(mk):
                    acc[0] += T.Cast("float", a[ki]) * T.Cast("float", bd[ki])
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


def rmsnorm(M: int, N: int):
    """RMSNorm with weight (TileLang's examples/norm/rms_norm.py, fp16 in and out): one row per
    block of 128 threads, the row in registers, an fp32 sum of squares."""

    @T.prim_func
    def rms(
        X: T.Tensor((M, N), "float16"),
        W: T.Tensor((N,), "float16"),
        Y: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(M, threads=128) as bx:
            x = T.alloc_fragment((1, N), "float16")
            sq = T.alloc_fragment((1, N), "float")
            ss = T.alloc_fragment((1,), "float")
            T.copy(X[bx : bx + 1, :], x)
            for i, j in T.Parallel(1, N):
                sq[i, j] = T.Cast("float", x[i, j]) * T.Cast("float", x[i, j])
            T.reduce_sum(sq, ss, dim=1)
            for i in T.Parallel(1):
                ss[i] = T.rsqrt(ss[i] / N + 1e-6)
            for i, j in T.Parallel(1, N):
                x[i, j] = T.Cast(
                    "float16", T.Cast("float", x[i, j]) * ss[i] * T.Cast("float", W[j])
                )
            T.copy(x, Y[bx : bx + 1, :])

    return rms


def softmax(M: int, N: int):
    """Row softmax (fp16 in and out, exp2 with the log2(e) scale, fp32 max and sum): one row per
    block of 128 threads."""

    @T.prim_func
    def smax(X: T.Tensor((M, N), "float16"), Y: T.Tensor((M, N), "float16")):
        with T.Kernel(M, threads=128) as bx:
            x = T.alloc_fragment((1, N), "float16")
            s = T.alloc_fragment((1, N), "float")
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
            T.copy(x, Y[bx : bx + 1, :])

    return smax


HARNESS = r"""// Test-set harness (perfmodel/testset/gen.py): TileLang's kernel, launched as TileLang launches it,
// on random inputs (fp16 values in [-1, 1) written over every buffer's bytes); WARM untimed launches,
// then REPS launches timed one by one with CUDA events (no profiler). ROT input sets are used in
// turn (ROT 1: warm; ROT x footprint above the L2: rotating). Usage: harness [reps] (reps 1 = one
// launch after warm-up, for the profiler).
#include <cuda.h>
#include "../../../probes/common.cuh"
#include "kernel.cu"
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
  const int smem = %(smem)d;
  CK(cudaFuncSetAttribute(%(entry)s, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  dim3 grid(%(grid)s), block(%(block)s);
  int next = 0;
  auto launch = [&] { void** b = buf[next++ %% ROT]; %(entry)s<<<grid, block, smem>>>(%(args)s); };
  for (int i = 0; i < 10; i++) launch();
  CK(cudaDeviceSynchronize());
  if (reps == 1) { launch(); CK(cudaDeviceSynchronize()); printf("RESULT %(name)s once\n"); return 0; }
  std::vector<double> v;
  for (int r = 0; r < reps; r++) v.push_back(event_us(launch));
  Stat s = stat_of(v);
  printf("RESULT %(name)s kernel median=%%.4f iqr=%%.4f unit=us n=%%d samples=", s.median, s.iqr, s.n);
  for (double x : v) printf("%%.3f,", x);
  printf("\n");
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
):
    """`sizes` in elements of each buffer, `elem_bytes` per buffer (default 2, fp16)."""
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
    eb = elem_bytes or [2] * len(sizes)
    nbytes = [n * b for n, b in zip(sizes, eb, strict=True)]
    # buffers by the prim_func's parameter names; the kernel takes them in its own order
    names = [func.buffer_map[p].name for p in func.params]
    params = _params(src, entry)
    block = list(threads) if isinstance(threads, tuple) else [threads]
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "kernel.cu").write_text(src)
    (d / "harness.cu").write_text(
        HARNESS
        % {
            "sizes": ", ".join(str(s) for s in nbytes),
            "nbuf": len(sizes),
            "rot": rot,
            "smem": smem,
            "entry": entry,
            "grid": ", ".join(str(g) for g in grid),
            "block": ", ".join(str(b) for b in block),
            "args": ", ".join(f"({t})b[{names.index(n)}]" for t, n in params),
            "name": name,
        }
    )
    nthreads = math.prod(block)
    launch = {
        "entry": entry,
        "grid": grid,
        "threads": nthreads,
        "block": block,
        "smem": smem,
        "rot": rot,
        "footprint_bytes": sum(nbytes),
        **info,
    }
    (d / "launch.json").write_text(json.dumps(launch, indent=1) + "\n")
    print(f"generated {name}: entry {entry}, grid {grid}, block {block}, smem {smem}, rot {rot}")


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
        info = {
            "family": "hgemm",
            "shape": SHAPE,
            "pick": h,
            "smem_estimate": s * (bm + bn) * bk * 2,
            "cache": "warm",
        }
        grid = [-(-N // bn), -(-M // bm), 1]
        write(
            out,
            f"hgemm2048_{name}",
            hgemm(bm, bn, bk, s, th, panel),
            [M * K, K * N, M * N],
            grid,
            th,
            info,
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
    for name, gen in (("decode", _decode), ("gemv", _gemv), ("rows", _rows)):
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


if __name__ == "__main__":
    main(Path(sys.argv[1]))
