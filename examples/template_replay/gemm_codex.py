"""FP16 GEMM with FP32 accumulation and shape-specialized CTA tiles."""

from functools import cache

import tilelang
import tilelang.language as T


# Cache compiled kernels only; every invocation allocates a new output.
@cache
@tilelang.jit(out_idx=[2])
def gemm(M, N, K, BM, BN, BK, stages, threads):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
            As = T.alloc_shared((BM, BK), "float16")
            Bs = T.alloc_shared((BK, BN), "float16")
            acc = T.alloc_fragment((BM, BN), "float32")
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=stages):
                # TileLang predicates tails and zero-fills invalid operand lanes.
                T.copy(A[by * BM, k * BK], As)
                T.copy(B[k * BK, bx * BN], Bs)
                T.gemm(As, Bs, acc)
            T.copy(acc, C[by * BM, bx * BN])

    return main


def entry(a, b):
    m, k = a.shape
    n = b.shape[1]
    if m == 32:
        config = (32, 64, 256, 2, 128)
    elif m == 127:
        config = (16, 64, 64, 3, 64)
    else:
        config = (64, 128, 32, 4, 256)
    return gemm(m, n, k, *config)(a, b)
