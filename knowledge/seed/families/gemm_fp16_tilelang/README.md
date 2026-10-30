# gemm_fp16_tilelang

FP16 GEMM `C[M,N] = A[M,K] @ B[K,N]` (row-major, FP32 accumulation, FP16 output) as a TileLang
family, built by WP7 as the first real chain (M1). `gen.py` emits one `@T.prim_func gemm(A, B, C)`
per assignment and shape; M, N and K are compile-time constants.

| Chain item | Template | Family variables | Code marker |
| --- | --- | --- | --- |
| `tiles` | `output_tiling@2` | BM, BN | `tiles` (T.Kernel grid), `tiles_store` (clipped store of C) |
| `k_split` | `reduction_partition@1` | KT, BK | `k_split` (the k-tile loop) |
| `ab_smem` | `physical_storage@1` | S (copies) | `ab_smem` (A_shared, B_shared) |
| `ab_async` | `async_transfer@1` | constants: cp.async, 16 bytes | `ab_async` (T.copy global -> shared) |
| `ab_stages` | `multi_buffer_rotation@1` | S, P (glue `P == S - 1`) | `ab_stages` (T.Pipelined num_stages) |
| `ab_swizzle` | `shared_resource_binding@1` | constant: xor_swizzle | `ab_swizzle` (TileLang's swizzled layout, not overridden) |
| `mma` | `compute_instruction_selection@2` | constant: mma.sync m16n8k16 f16 | `mma` (T.gemm) |

Glue: `g_prefetch` (TileLang prefetches S - 1 tiles), `g_acc_regs` (accumulator registers),
`g_no_combine` (`split == 1`: no step combines partial sums across CTAs). The generator does
emit `split > 1` with plain stores, which is a write-write race; WP7's legality fault injection
removes `g_no_combine` to show that the checks catch it.

Cost modeling choices (decision D282): the `mma` step's `p.instances` counts occupied CTA slots
including the idle slots of the last wave; the swizzle step's access count assumes each warp
reads its share of the A/B tiles once per k-tile (`1 + threads // 64` passes over the tiles).
