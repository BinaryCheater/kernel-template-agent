# vLLM RMSNorm — decomposition

Source: vllm-project/vllm @ `2eae56c4ba0cf12087fec035514eedede83c257d`. Kernel
`rms_norm_kernel<scalar_t, VEC_SIZE, NUM_DIMS, HasWeight>` (`csrc/layernorm_kernels.cu:14-101`),
launcher `rms_norm` (`csrc/layernorm_kernels.cu:221-290`), vector read helper
`vectorize_read_with_alignment` (`csrc/quantization/vectorization_utils.cuh:132-195`).

Pool: seed + proposals of the first six references. Status: analysis of source; not built or run.
ISA eligibility for GB10: plain CUDA, cub; no architecture-specific instruction.

## 1. Workload

`out[t, :] = x[t, :] * rsqrt(mean(x[t, :]^2) + eps) * w` for T rows of hidden size D, f16 in and
out, statistics in f32. 3-D and 4-D inputs (per-head q/k norm) use the same program with a
different row index map.

## 2. Transaction inventory

| # | Class | Transaction | Where | Explained by |
| --- | --- | --- | --- | --- |
| T01 | schedule | grid = T rows, one CTA per row; block = min(D / VEC_SIZE, 1024 or 256) | `csrc/layernorm_kernels.cu:254-272` | out_cta |
| T02 | schedule | row index map for rank 3 / 4 | `csrc/layernorm_kernels.cu:31-51` | out_cta |
| T03 | reduce | per-thread strided partial sum of squares in f32 | `csrc/layernorm_kernels.cu:53-66` | row_part |
| T04 | transfer | vector reads with scalar prefix for a misaligned start | `csrc/quantization/vectorization_utils.cuh:132-180` | vec, align_prefix |
| T05 | reduce | cub::BlockReduce: warp shuffle, then smem across warps | `csrc/layernorm_kernels.cu:68-70` | xlane, smem_combine |
| T06 | compute + sync | thread 0: rsqrtf(var / D + eps) into smem, __syncthreads | `csrc/layernorm_kernels.cu:72-76` | rsqrt, smem_combine (result_to all_threads) |
| T07 | transfer | pass 2 re-reads x (not cached on chip) and w, vector loads | `csrc/layernorm_kernels.cu:78-86` | row_reread, vec |
| T08 | compute | x * scale * w in f32, cast, vector store | `csrc/layernorm_kernels.cu:87-100` | fuse, vec |
| T09 | numeric | reduction grouping depends on block size (batch-size dependent) | `csrc/layernorm_kernels.cu:254-257` | none (ux-01) |
| T10 | schedule | VEC_SIZE / rank / HasWeight dispatch | `csrc/layernorm_kernels.cu:262-290` | none (ux-02) |
| T11 | other | unrolling / registers by nvcc | `csrc/layernorm_kernels.cu:55-58, 89-99` | none (ux-03) |

Observation (not counted): pass 2 reinterprets the row as vectors (`csrc/layernorm_kernels.cu:79-83`)
with no misaligned-prefix path, unlike pass 1; whether callers guarantee alignment was not checked.

## 3. Refinement chain

fuse (operator_fusion) -> out_cta (output_tiling, one row per CTA) -> row_part
(reduction_partition, strided threads) -> xlane (cross_lane_reduction) -> smem_combine
(smem_partial_combine, broadcast to all threads) -> rsqrt (approximate_math_rewrite); row_reread
(physical_storage, level gmem); vec (vectorized_access) -> align_prefix (boundary_predication,
alignment_prefix). All 9 steps matched (M); 3 unmatched transactions.

## 4. Least certain judgements

1. **row_reread as physical_storage(level = gmem).** Choosing not to keep the row on chip is a
   storage decision, but the seed Template is about placing values in reusable buffers; binding
   level = gmem with share_order = read_twice stretches it. A Template for "recompute / re-read
   instead of store" might be warranted if this recurs.
2. **xlane + smem_combine for one library call.** cub::BlockReduce was split into its two known
   phases from cub's documented algorithm, not from code in this repository.
3. **No numeric step for f16 -> f32 statistics.** Accumulating in f32 is a precision raise, which
   no Template models; it was treated as part of the workload's arithmetic.
