"""Causal flash attention in TileLang (FP16 in/out, FP32 accumulation)."""

import tilelang
import tilelang.language as T

_LOG2E = 1.4426950408889634


def _pick_config(seq_len, dim):
    # (block_M, block_N, num_stages, threads)
    if seq_len <= 576:
        return 64, 64, 2, 128
    if dim <= 64:
        return 128, 64, 2, 256
    return 128, 64, 2, 256


@tilelang.jit(out_idx=[3])
def _flash_kernel(batch, heads, seq_len, dim, block_m, block_n, num_stages, threads):
    dtype = "float16"
    accum = "float32"
    scale = (1.0 / (dim**0.5)) * _LOG2E

    @T.prim_func
    def main(
        Q: T.Tensor((batch, heads, seq_len, dim), dtype),
        K: T.Tensor((batch, heads, seq_len, dim), dtype),
        V: T.Tensor((batch, heads, seq_len, dim), dtype),
        Output: T.Tensor((batch, heads, seq_len, dim), dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_m), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_m, dim], dtype)
            K_shared = T.alloc_shared([block_n, dim], dtype)
            V_shared = T.alloc_shared([block_n, dim], dtype)
            acc_s = T.alloc_fragment([block_m, block_n], accum)
            acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
            acc_o = T.alloc_fragment([block_m, dim], accum)
            scores_max = T.alloc_fragment([block_m], accum)
            scores_max_prev = T.alloc_fragment([block_m], accum)
            scores_scale = T.alloc_fragment([block_m], accum)
            scores_sum = T.alloc_fragment([block_m], accum)
            logsum = T.alloc_fragment([block_m], accum)

            T.copy(Q[bz, by, bx * block_m : (bx + 1) * block_m, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum))

            loop_range = T.ceildiv((bx + 1) * block_m, block_n)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, by, k * block_n : (k + 1) * block_n, :], K_shared)
                for i, j in T.Parallel(block_m, block_n):
                    acc_s[i, j] = T.if_then_else(
                        bx * block_m + i >= k * block_n + j, 0.0, -T.infinity(accum)
                    )
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, by, k * block_n : (k + 1) * block_n, :], V_shared)

                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_m):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_m, block_n):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_m):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] = acc_o[i, j] * scores_scale[i]
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i in T.Parallel(block_m):
                scores_scale[i] = 1.0 / logsum[i]
            for i, j in T.Parallel(block_m, dim):
                if bx * block_m + i < seq_len:
                    Output[bz, by, bx * block_m + i, j] = acc_o[i, j] * scores_scale[i]

    return main


_KERNELS = {}


def _get_kernel(batch, heads, seq_len, dim):
    key = (batch, heads, seq_len, dim)
    kern = _KERNELS.get(key)
    if kern is None:
        bm, bn, stages, threads = _pick_config(seq_len, dim)
        kern = _flash_kernel(batch, heads, seq_len, dim, bm, bn, stages, threads)
        _KERNELS[key] = kern
    return kern


def entry(q, k, v):
    batch, heads, seq_len, dim = q.shape
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    kern = _get_kernel(batch, heads, seq_len, dim)
    return kern(q, k, v)
