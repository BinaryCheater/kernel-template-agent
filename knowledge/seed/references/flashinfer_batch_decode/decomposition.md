# FlashInfer batch decode (paged KV) — decomposition

Source: flashinfer-ai/flashinfer @ `732134f2ff4da63268b01c7c4ef9f61e9a1b8a43`. Kernel
`BatchDecodeWithPagedKVCacheDevice` (`include/flashinfer/attention/decode.cuh:401-619`), launch
`BatchDecodeWithPagedKVCacheDispatched` (`include/flashinfer/attention/decode.cuh:760-840`), plan
`DecodePlan` (`include/flashinfer/attention/scheduler.cuh:426-527`), merge
`VariableLengthMergeStates` (`include/flashinfer/attention/cascade.cuh:690-725`). This is the
CUDA-core decode path (FlashInfer routes large GQA groups to tensor-core prefill kernels; that path
is not analysed).

Pool: seed + proposals of the first four references. Status: analysis of source; not built or
run. ISA eligibility for GB10: cp.async, shfl, ex2.approx; griddepcontrol needs sm_90+ and is
compiled in for sm_12x.

## 1. Workload

For each request b (ragged KV length L_b stored in pages of page_size rows) and query head h:
`o[b,h] = softmax(scale q[b,h] K_b^T) V_b` with K_b, V_b the rows of kv head h / g gathered through
`kv_indices[kv_indptr[b] : kv_indptr[b+1]]`; also `lse[b,h]`. f16 inputs and output, head_dim
128, group size g in {1, 2, 3, 4, 8}.

## 2. Transaction inventory

| # | Class | Transaction | Where | Explained by |
| --- | --- | --- | --- | --- |
| T01 | schedule | host plan: occupancy grid, binary search of chunk size, split decision | `include/flashinfer/attention/scheduler.cuh:74-99, 150-208` | plan |
| T02 | schedule | descriptors request_indices, kv_tile_indices, o_indptr, block_valid_mask; padded grid | `include/flashinfer/attention/scheduler.cuh:349-363, 426-512`, `include/flashinfer/attention/decode.cuh:425-431` | plan |
| T03 | schedule | grid (padded_batch_size, Hk), block (bdx, bdy = g, bdz) | `include/flashinfer/attention/decode.cuh:770-791` | out_cta |
| T04 | schedule | dispatch of g, vec_size, bdx, tile_size_per_bdx, num_stages_smem | `include/flashinfer/attention/decode.cuh:770-781` | none (ux-05) |
| T05 | reduce | KV range of a request split into chunks | `include/flashinfer/attention/decode.cuh:432-437` | split |
| T06 | reduce | bdz thread groups take different KV rows of each tile | `include/flashinfer/attention/decode.cuh:493-496, 550-556` | warp_part |
| T07 | storage | q (f32) and state (m, d, o) in registers | `include/flashinfer/attention/decode.cuh:452-483, 530-531` | q_reg |
| T08 | transfer | page table -> offsets in smem every bdx iterations | `include/flashinfer/attention/decode.cuh:491-499, 535-547` | paged_gather |
| T09 | transfer | K/V cp.async pred_load, separate commit groups | `include/flashinfer/attention/decode.cuh:503-528, 569-595` | kv_async, vec |
| T10 | storage + sync | S-slot ring, wait_group<2S-1> + block.sync before and after use | `include/flashinfer/attention/decode.cuh:548-549, 557, 579-583` | kv_stages |
| T11 | compute | q.k by FMAs, shfl_xor over bdx lanes | `include/flashinfer/attention/decode.cuh:79-89` | qk_dot, qk_xlane |
| T12 | compute | online softmax update with ptx_exp2; p.v by FMAs | `include/flashinfer/attention/decode.cuh:93-117, 134-146` | online, exp2, qk_dot |
| T13 | transfer | K residue not loaded (score -inf), V residue zero-filled | `include/flashinfer/attention/decode.cuh:99, 511-525` | k_guard, v_guard |
| T14 | reduce + sync | bdz states merged through smem (smem reused from K/V ring) | `include/flashinfer/attention/decode.cuh:158-190, 597-602` | state_merge, kv_smem |
| T15 | transfer | o / lse stored by tz == 0 (to tmp buffers when split) | `include/flashinfer/attention/decode.cuh:609-615, 815-822` | vec, merge |
| T16 | reduce | merge kernel over chunks of each request | `include/flashinfer/attention/cascade.cuh:374-475, 690-725` | merge |
| T17 | sync | programmatic dependent launch (griddepcontrol) | `include/flashinfer/attention/decode.cuh:470-481, 616-618` | none (ux-04) |
| T18 | other | register allocation by nvcc | `include/flashinfer/attention/decode.cuh:530-534` | none (ux-06) |

## 3. Refinement chain

| Step | Template | | Step | Template | |
| --- | --- | --- | --- | --- | --- |
| online | online_rescaled_reduction | M | exp2 | approximate_math_rewrite | M |
| fuse | operator_fusion | M | q_reg | physical_storage | M |
| out_cta | output_tiling (GQA group per CTA) | M | kv_smem | physical_storage | M |
| split | reduction_partition (runtime) | M | kv_async | async_transfer | M |
| plan | runtime_work_descriptors | N | kv_stages | multi_buffer_rotation | M |
| merge | split_combine_kernel | N | paged_gather | indexed_gather | N |
| warp_part | reduction_partition | M | vec | vectorized_access | M |
| state_merge | smem_partial_combine (rescaled_merge) | M | k_guard | boundary_predication | M |
| qk_dot | compute_instruction_selection (fma) | M | v_guard | boundary_predication | M |
| qk_xlane | cross_lane_reduction | M | | | |

Match rate 16/19. 6 unmatched transactions (3 new-Template steps, 3 unexplained).

## 4. Least certain judgements

1. **split matched with a runtime num_groups.** The seed reduction_partition was assumed to allow a
   runtime-bound group count. If WP4 defines `num_groups` as compile-time only, this step becomes
   partial and the gap moves from `plan` to `split`.
2. **out_cta absorbing GQA reuse.** Loading K/V once for all g query heads is a reuse decision
   (K/V shared by g output rows); it is expressed only through `index_map: grouped_heads` and
   `tile_m: group_size`. A Template that states operand reuse across tiles explicitly might be
   warranted.
3. **kv_stages P = S - 1, beta = 1.** Derived from the prologue issuing S stages and the refill of
   a slot right after the barrier that follows its use. K and V use separate commit groups and a
   wait depth of 2S - 1; the binding ignores that K and V are in different phases (similar to the
   FlashAttention-2 finding, but here the obligation still holds).
