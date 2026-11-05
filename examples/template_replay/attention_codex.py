from functools import lru_cache

import tilelang
import tilelang.language as T


@lru_cache(None)
def _kernel(B, H, S, D, BM=64, BN=128, threads=128):
    scale = (D**-0.5) * 1.4426950408889634

    @T.prim_func
    def main(
        Q: T.Tensor((B, H, S, D), "float16"),
        K: T.Tensor((B, H, S, D), "float16"),
        V: T.Tensor((B, H, S, D), "float16"),
        Output: T.Tensor((B, H, S, D), "float16"),
    ):
        with T.Kernel(H, T.ceildiv(S, BM), B, threads=threads) as (h, tile, b):
            # Longest causal rows first; heads are adjacent in launch order.
            bx = T.ceildiv(S, BM) - 1 - tile
            qs = T.alloc_shared((BM, D), "float16")
            # K and V occupy this buffer sequentially, with CTA barriers.
            ks = T.alloc_shared((BN, D), "float16")
            ps = T.alloc_fragment((BM, BN), "float16")
            score = T.alloc_fragment((BM, BN), "float32")
            acc = T.alloc_fragment((BM, D), "float32")
            maximum = T.alloc_fragment((BM,), "float32")
            old = T.alloc_fragment((BM,), "float32")
            alpha = T.alloc_fragment((BM,), "float32")
            mass = T.alloc_fragment((BM,), "float32")
            tile_mass = T.alloc_fragment((BM,), "float32")
            T.copy(Q[b, h, bx * BM : bx * BM + BM, :], qs)
            T.fill(acc, 0)
            T.fill(mass, 0)
            T.fill(maximum, -T.infinity("float32"))
            # Keep this loop serial: pipelining the aliased copies is unsafe.
            for kb in T.serial(T.ceildiv(T.min((bx + 1) * BM, S), BN)):
                T.copy(K[b, h, kb * BN : kb * BN + BN, :], ks)
                T.clear(score)
                T.gemm(qs, ks, score, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                if (kb + 1) * BN <= bx * BM:
                    for i, j in T.Parallel(BM, BN):
                        score[i, j] *= scale
                else:
                    for i, j in T.Parallel(BM, BN):
                        score[i, j] = T.if_then_else(
                            (kb * BN + j <= bx * BM + i) & (kb * BN + j < S),
                            score[i, j] * scale,
                            -T.infinity("float32"),
                        )
                T.copy(maximum, old)
                T.reduce_max(score, maximum, dim=1, clear=False)
                for i in T.Parallel(BM):
                    alpha[i] = T.exp2(old[i] - maximum[i])
                for i, j in T.Parallel(BM, BN):
                    score[i, j] = T.exp2(score[i, j] - maximum[i])
                T.reduce_sum(score, tile_mass, dim=1)
                for i in T.Parallel(BM):
                    mass[i] = mass[i] * alpha[i] + tile_mass[i]
                for i, j in T.Parallel(BM, D):
                    acc[i, j] *= alpha[i]
                # Correct only the old state before adding the new PV term.
                T.copy(score, ps)
                T.sync_threads()
                T.copy(V[b, h, kb * BN : kb * BN + BN, :], ks)
                T.gemm(ps, ks, acc, policy=T.GemmWarpPolicy.FullRow)
                T.sync_threads()
            for i, j in T.Parallel(BM, D):
                acc[i, j] /= mass[i]
            T.copy(acc, Output[b, h, bx * BM : bx * BM + BM, :])

    return tilelang.compile(main, out_idx=[3])


def entry(q, k, v):
    B, H, S, D = q.shape
    if D == 128:
        return _kernel(B, H, S, D, BN=64)(q, k, v)
    return _kernel(B, H, S, D)(q, k, v)
