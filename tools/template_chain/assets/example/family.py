"""Example family: plain FP16 GEMM. `prepare` compiles once; `run` launches the kernel."""

import tilelang
import tilelang.language as T

# The form's constants come from kernels built this way, with threads as written.
PASSES = {"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}


def prepare(shape, x, inputs):
    M, N, K = shape["M"], shape["N"], shape["K"]
    BM, BN, BK, S, threads = x["BM"], x["BN"], x["BK"], x["s"], x["T"]

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((N, K), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
            A_s = T.alloc_shared((BM, BK), "float16")
            B_s = T.alloc_shared((BN, BK), "float16")
            C_l = T.alloc_fragment((BM, BN), "float32")
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=S):
                T.copy(A[by * BM, k * BK], A_s)
                T.copy(B[bx * BN, k * BK], B_s)
                T.gemm(A_s, B_s, C_l, transpose_B=True)
            T.copy(C_l, C[by * BM, bx * BN])

    kernel = tilelang.compile(gemm, target="cuda", pass_configs=PASSES)
    a, b = inputs["a"], inputs["b"]
    c = a.new_empty((M, N))

    def run():
        kernel(a, b, c)
        return {"c": c}

    return run
