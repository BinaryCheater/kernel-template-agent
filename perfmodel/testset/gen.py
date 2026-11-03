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
"""

import json
import re
import sys
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


HARNESS = r"""// Test-set harness (perfmodel/testset/gen.py): TileLang's kernel, launched as TileLang launches it,
// on random fp16 inputs in [-1, 1); warm cache state: WARM untimed launches, then REPS launches
// timed one by one with CUDA events (no profiler). Usage: harness [reps] (reps 1 = one launch after
// warm-up, for the profiler).
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
  const size_t sizes[] = {%(sizes)s};
  half_t* buf[%(nbuf)d];
  for (int i = 0; i < %(nbuf)d; i++) { CK(cudaMalloc(&buf[i], sizes[i] * 2)); fill<<<1024, 256>>>(buf[i], sizes[i], 0x9E3779B9u * (i + 1)); }
  CK(cudaDeviceSynchronize());
  const int smem = %(smem)d;
  CK(cudaFuncSetAttribute(%(entry)s, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  dim3 grid(%(grid)s), block(%(threads)d);
  auto launch = [&] { %(entry)s<<<grid, block, smem>>>(%(args)s); };
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


def write(out: Path, name: str, func, sizes: list[int], grid: list[int], threads: int, info: dict):
    k = tilelang.compile(func, target="cuda", pass_configs=PASS_CONFIGS)
    src = k.get_kernel_source()
    m = re.search(r"__global__ void(?: __launch_bounds__\([^)]*\))? (\w+)\(", src)
    assert m, "no kernel entry in TileLang's source"
    entry = m.group(1)
    sm = re.search(r"cudaFuncAttributeMaxDynamicSharedMemorySize, (\d+)", k.get_host_source() or "")
    smem = int(sm.group(1)) if sm else int(info["smem_estimate"])
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "kernel.cu").write_text(src)
    (d / "harness.cu").write_text(
        HARNESS
        % {
            "sizes": ", ".join(str(s) for s in sizes),
            "nbuf": len(sizes),
            "smem": smem,
            "entry": entry,
            "grid": ", ".join(str(g) for g in grid),
            "threads": threads,
            "args": ", ".join(f"buf[{i}]" for i in range(len(sizes))),
            "name": name,
        }
    )
    launch = {"entry": entry, "grid": grid, "threads": threads, "smem": smem, **info}
    (d / "launch.json").write_text(json.dumps(launch, indent=1) + "\n")
    print(f"generated {name}: entry {entry}, grid {grid}, threads {threads}, smem {smem}")


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


if __name__ == "__main__":
    main(Path(sys.argv[1]))
