# Marlin (int4 weight x fp16 activation GEMM) — decomposition

Source: IST-DASLab/marlin @ `1f25790bdd49fba53106164a24666dade68d7c90`. Kernel
`Marlin<256, thread_m_blocks, thread_n_blocks, thread_k_blocks, 4, group_blocks>`
(`marlin/marlin_cuda_kernel.cu:189-700`), host driver `marlin_cuda`
(`marlin/marlin_cuda_kernel.cu:731-822`), weight packing `Layer.pack` (`marlin/__init__.py:102-139`).

Pool: seed + proposals of the GEMM, FlashAttention-2 and FlashAttention-3 references. Status:
analysis of source; not built or run. ISA eligibility for GB10: mma.sync f16, ldmatrix, cp.async
with L2 cache-policy (createpolicy), lop3, ld.acquire / red.relaxed are sm_80 features available on
sm_12x.

## 1. Workload

`C = A @ dequant(B)`, A f16 [M, K] row-major, B symmetric int4 [K, N] in the Marlin packed layout,
scales s f16 per output channel or per 128-row group, C f16. `dequant(q) = (q - 8) * s`.
N must be a multiple of thread_n and K of thread_k (`marlin/marlin_cuda_kernel.cu:770-771`). A
zeroed int32 lock workspace is an input. Partial sums between CTAs are stored in fp16 in C.

## 2. Transaction inventory

| # | Class | Transaction | Where | Explained by |
| --- | --- | --- | --- | --- |
| T01 | storage | offline permutation + packing of B and s | `marlin/__init__.py:40-68, 102-139` | prepack |
| T02 | schedule | host splits M into <= 64-row chunks, par 64-row problems per launch | `marlin/marlin_cuda_kernel.cu:784-816` | m_split |
| T03 | schedule | grid = #SMs; stripes of `iters` k-tiles in column-major order | `marlin/marlin_cuda_kernel.cu:229-298, 768` | stripes, k_part |
| T04 | schedule | config dispatch: (thread_k, thread_n) by M; ten CALL_IF instances | `marlin/marlin_cuda_kernel.cu:754-764, 800-812` | none (ux-07) |
| T05 | executor | 8 warps: thread_n_blocks / 4 along N, the rest along K | `marlin/marlin_cuda_kernel.cu:300-325, 466-470` | out_warp, warp_k |
| T06 | transfer | A via cp.async.cg 16 B, predicated rows | `marlin/marlin_cuda_kernel.cu:54-66, 387-396` | ab_async, vec, a_guard |
| T07 | transfer | B and s via cp.async with L2::evict_first policy | `marlin/marlin_cuda_kernel.cu:69-80, 398-410` | ab_async, b_stream_hint |
| T08 | sync | commit group per stage; wait_group<stages-2> + __syncthreads | `marlin/marlin_cuda_kernel.cu:82-90, 413, 417-422` | ab_async, stages |
| T09 | storage | 4-stage smem ring for A, B, s | `marlin/marlin_cuda_kernel.cu:369-373, 621-630, 637-646` | ab_smem, stages |
| T10 | access | XOR transform of A smem addresses | `marlin/marlin_cuda_kernel.cu:340-361` | a_swizzle |
| T11 | transfer | A fragments via ldmatrix.x4; B fragments via one 128-bit LDS | `marlin/marlin_cuda_kernel.cu:107-114, 425-440` | a_frag, b_frag |
| T12 | storage | register fragments double-buffered by k % 2 | `marlin/marlin_cuda_kernel.cu:374-377, 425-440` | frag_stages (partial) |
| T13 | numeric | lop3 magic-number int4 -> fp16, folded zero point, group scale | `marlin/marlin_cuda_kernel.cu:116-160, 442-457` | dequant |
| T14 | compute | mma.sync m16n8k16 f16 -> f32 | `marlin/marlin_cuda_kernel.cu:93-105, 459-461` | mma |
| T15 | schedule | M innermost so dequant overlaps MMA (left to nvcc) | `marlin/marlin_cuda_kernel.cu:443-462` | none (ux-08) |
| T16 | reduce | warp partials combined by smem log tree | `marlin/marlin_cuda_kernel.cu:466-509` | warp_k, smem_combine |
| T17 | reduce + sync | cross-CTA partials added into C in slice order under a lock | `marlin/marlin_cuda_kernel.cu:162-188, 514-571, 674-678` | global_combine |
| T18 | compute | per-channel scale and f32 -> f16 at write-out | `marlin/marlin_cuda_kernel.cu:588-592, 659-672` | epi_scale |
| T19 | transfer + sync | results staged in smem, barrier, 16-byte stores to C | `marlin/marlin_cuda_kernel.cu:573-619` | epi_stage |
| T20 | storage | accumulators frag_c in registers | `marlin/marlin_cuda_kernel.cu:374-384` | acc_store |
| T21 | other | register allocation / unrolling by nvcc | `marlin/marlin_cuda_kernel.cu:635-647` | none (ux-09) |

## 3. Refinement chain

| Step | Template | | Step | Template | |
| --- | --- | --- | --- | --- | --- |
| prepack | offline_operand_prepack | N | ab_async | async_transfer | M |
| dequant | fused_dequantization | N | b_stream_hint | cache_eviction_hint | M |
| m_split | output_tiling | M | stages | multi_buffer_rotation | M |
| stripes | linearized_work_partition | N | a_swizzle | shared_resource_binding | M |
| k_part | reduction_partition | M | a_frag | collective_fragment_load | M |
| global_combine | ordered_global_combine | N | b_frag | collective_fragment_load (lds128) | M |
| warp_k | reduction_partition | M | frag_stages | multi_buffer_rotation | P |
| smem_combine | smem_partial_combine | N | vec | vectorized_access | M |
| out_warp | output_tiling | M | a_guard | boundary_predication | M |
| mma | compute_instruction_selection | M | epi_scale | operator_fusion | M |
| acc_store | physical_storage | M | epi_stage | staged_relayout | M |
| ab_smem | physical_storage | M | | | |

Match rate 17/23. 9 unmatched transactions (5 new-Template steps, 1 partial, 3 unexplained).

## 4. Least certain judgements

1. **Two reduction_partition steps (k_part across CTAs, warp_k across warps).** Marlin splits K at
   both levels; each is a separate partition with its own combine (global_combine,
   smem_combine). The binding `num_groups: red_off_times_2` for warp_k is read from the reduction
   loop, not from an explicit constant, and may be off by the factor used for thread_k_blocks.
2. **b_frag as collective_fragment_load with lds128.** A plain vector load is not "collective"; it
   was matched because the Template's variable domain lists lds128 and the obligation (lane gets
   its fragment elements) holds by construction of the prepack. One could argue the step belongs
   to offline_operand_prepack instead.
3. **epi_scale as operator_fusion.** Applying the per-channel scale at write-out is the
   distributive move of a scale out of the K sum (dequant's `c_site` constraint), so it is
   arguably part of fused_dequantization rather than a separate fusion.
