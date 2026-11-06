"""# kta:fixture hand-written causal forward, for harness runs without a model."""

import tilelang
import tilelang.language as T

# The form's constants come from kernels built this way, with threads as written.
PASSES = {"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}


def prepare(shape, x, inputs):
    S, H, D = shape["S"], shape["H"], shape["D"]
    BM, BN, stages, threads = x["BM"], x["BN"], x["s"], x["T"]
    scale = (1.0 / D) ** 0.5 * 1.44269504
    dtype, accum = "float16", "float"
    size = [1, S, H, D]

    @T.prim_func
    def fa(
        Q: T.Tensor(size, dtype),
        K: T.Tensor(size, dtype),
        V: T.Tensor(size, dtype),
        Out: T.Tensor(size, dtype),
        Lse: T.Tensor([1, H, S], accum),
    ):
        with T.Kernel(T.ceildiv(S, BM), H, threads=threads) as (bx, by):
            Q_s = T.alloc_shared([BM, D], dtype)
            K_s = T.alloc_shared([BN, D], dtype)
            V_s = T.alloc_shared([BN, D], dtype)
            O_s = T.alloc_shared([BM, D], dtype)
            acc_s = T.alloc_fragment([BM, BN], accum)
            acc_c = T.alloc_fragment([BM, BN], dtype)
            acc_o = T.alloc_fragment([BM, D], accum)
            m = T.alloc_fragment([BM], accum)
            m_prev = T.alloc_fragment([BM], accum)
            sc = T.alloc_fragment([BM], accum)
            ssum = T.alloc_fragment([BM], accum)
            lsum = T.alloc_fragment([BM], accum)
            T.copy(Q[0, bx * BM : (bx + 1) * BM, by, :], Q_s)
            T.fill(acc_o, 0)
            T.fill(lsum, 0)
            T.fill(m, -T.infinity(accum))
            for k in T.Pipelined(T.ceildiv((bx + 1) * BM, BN), num_stages=stages):
                T.copy(K[0, k * BN : (k + 1) * BN, by, :], K_s)
                T.clear(acc_s)
                T.gemm(Q_s, K_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(BM, BN):
                    acc_s[i, j] = T.if_then_else(
                        k * BN + j <= bx * BM + i, acc_s[i, j], -T.infinity(accum)
                    )
                T.copy(m, m_prev)
                T.fill(m, -T.infinity(accum))
                T.reduce_max(acc_s, m, dim=1, clear=False)
                for i in T.Parallel(BM):
                    m[i] = T.max(m[i], m_prev[i])
                for i in T.Parallel(BM):
                    sc[i] = T.exp2(m_prev[i] * scale - m[i] * scale)
                for i, j in T.Parallel(BM, D):
                    acc_o[i, j] *= sc[i]
                for i, j in T.Parallel(BM, BN):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - m[i] * scale)
                T.copy(acc_s, acc_c)
                T.copy(V[0, k * BN : (k + 1) * BN, by, :], V_s)
                T.gemm(acc_c, V_s, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, ssum, dim=1)
                for i in T.Parallel(BM):
                    lsum[i] = lsum[i] * sc[i] + ssum[i]
            for i, j in T.Parallel(BM, D):
                acc_o[i, j] /= lsum[i]
            for i in T.Parallel(BM):
                Lse[0, by, bx * BM + i] = (T.log2(lsum[i]) + m[i] * scale) * 0.6931471805599453
            T.copy(acc_o, O_s)
            T.copy(O_s, Out[0, bx * BM : (bx + 1) * BM, by, :])

    kernel = tilelang.compile(fa, target="cuda", pass_configs=PASSES)
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    o = q.new_empty(size)
    lse = q.new_empty([1, H, S], dtype=q.dtype).float()

    def run():
        kernel(q, k, v, o, lse)
        return {"o": o, "lse": lse}

    return run
