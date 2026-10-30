# FlashAttention-2 forward — decomposition

Source: Dao-AILab/flash-attention @ `a3a9dfa9bf10b52e55e8507b16ab0fe0dc1f38a0`, directory
`csrc/flash_attn/src/` (paths below are relative to the repository root). Program: `compute_attn_1rowblock`
(`csrc/flash_attn/src/flash_fwd_kernel.h:54-501`), launched by `run_mha_fwd_hdim128`
(`csrc/flash_attn/src/flash_fwd_launch_template.h:246-266`). Configuration decomposed: f16, head_dim 128, no dropout,
the non-sm8x branch `Flash_fwd_kernel_traits<128, 128, 64, 4, false, false>` (kBlockM = 128,
kBlockN = 64, 4 warps, Q not in registers, Q and K not sharing smem). GB10 reports compute
capability 12.1, so it would take this branch.

Pool: seed + proposals of `cutlass_ampere_tf32_gemm`. Status: analysis of source; not built or
run. ISA eligibility for GB10: mma.sync f16, cp.async, ldmatrix (incl. .trans) are sm_80
instructions available on sm_12x.

## 1. Workload

Per batch b and query head h: `O = softmax(scale * Q K^T + mask) V`, `LSE = logsumexp(scale * Q
K^T + mask)` per row. Q [B, Sq, H, 128], K, V [B, Sk, Hk, 128] f16, H a multiple of Hk (GQA: K/V
head = h / (H/Hk)). Mask: none or causal (bottom-right aligned when Sq != Sk). O f16, LSE f32.
Runtime: scale, causal flag (selects a compiled variant), Sq, Sk, B, H, Hk.

## 2. Transaction inventory

| # | Class | Transaction | Where | Explained by |
| --- | --- | --- | --- | --- |
| T01 | schedule | grid (ceil(Sq/128), B, H); one CTA per 128-row Q tile per head | `csrc/flash_attn/src/flash_fwd_launch_template.h:64`, `csrc/flash_attn/src/flash_fwd_kernel.h:1083-1098` | out_cta |
| T02 | schedule | K/V head index = h / h_h_k_ratio (GQA) | `csrc/flash_attn/src/flash_fwd_kernel.h:155-161` | out_cta (index_map) |
| T03 | schedule | tile config chosen by device and causal flag | `csrc/flash_attn/src/flash_fwd_launch_template.h:246-266` | none (ux-07) |
| T04 | executor | 4 warps along M, each owning full rows of S and O | `csrc/flash_attn/src/kernel_traits.h:74-77` | out_warp |
| T05 | compute | S = Q K^T per key block, mma.sync m16n8k16 | `csrc/flash_attn/src/flash_fwd_kernel.h:326-329, 393-396`, `csrc/flash_attn/src/utils.h:139-160` | mma |
| T06 | compute | online softmax: row max, exp2, row sum, rescale acc_o | `csrc/flash_attn/src/softmax.h:128-167` | online, exp2 |
| T07 | compute | quad (4-lane) allreduce of row max / sum | `csrc/flash_attn/src/softmax.h:38-52`, `csrc/flash_attn/src/utils.h:112-130` | row_xlane |
| T08 | numeric | exp folded into exp2 with scale_log2 FFMA; __logf for LSE | `csrc/flash_attn/src/softmax.h:66-90, 180` | exp2 |
| T09 | numeric | P converted f32 -> f16 | `csrc/flash_attn/src/flash_fwd_kernel.h:354, 416` | p_convert |
| T10 | transfer | P accumulator reinterpreted as A operand | `csrc/flash_attn/src/flash_fwd_kernel.h:372-374, 434-435`, `csrc/flash_attn/src/utils.h:200-226` | p_reuse |
| T11 | compute | O += P V, mma.sync with A from registers | `csrc/flash_attn/src/flash_fwd_kernel.h:374, 435`, `csrc/flash_attn/src/utils.h:166-185` | mma, fuse |
| T12 | compute | final O / l and LSE | `csrc/flash_attn/src/softmax.h:169-185`, `csrc/flash_attn/src/flash_fwd_kernel.h:440` | online (finalize) |
| T13 | reduce | key blocks traversed from n_block_max - 1 down to n_block_min | `csrc/flash_attn/src/flash_fwd_kernel.h:274, 309, 385` | kv_part, online (traversal) |
| T14 | reduce | causal n_block_max at the diagonal; early exit for empty rows | `csrc/flash_attn/src/flash_fwd_kernel.h:90-134` | causal_prune |
| T15 | reduce | peeled masking iterations, then mask-free iterations | `csrc/flash_attn/src/flash_fwd_kernel.h:297-382, 385-412`, `csrc/flash_attn/src/mask.h:129-200` | causal_prune |
| T16 | storage | acc_o, acc_s, row_max, row_sum in registers | `csrc/flash_attn/src/flash_fwd_kernel.h:290-292, 310` | acc_store |
| T17 | storage | sQ; sO aliases sQ | `csrc/flash_attn/src/flash_fwd_kernel.h:167, 444` | q_smem |
| T18 | storage | one sK and one sV buffer | `csrc/flash_attn/src/flash_fwd_kernel.h:170-173` | kv_smem |
| T19 | transfer | Q/K/V gmem -> smem cp.async.cg 16 B | `csrc/flash_attn/src/kernel_traits.h:124-133`, `csrc/flash_attn/src/flash_fwd_kernel.h:257, 276, 317, 342, 390, 404` | kv_async, vec |
| T20 | sync | wait_group 0 + __syncthreads twice per iteration; fences inside the if to avoid races | `csrc/flash_attn/src/flash_fwd_kernel.h:312-313, 339-346, 388-389, 401-408` | kv_async, kv_rotation (partial) |
| T21 | storage | K slot refilled mid-iteration (one tile lookahead with one slot) | `csrc/flash_attn/src/flash_fwd_kernel.h:339-346, 401-408` | kv_rotation (partial) |
| T22 | transfer | K residue not cleared (scores masked), V residue zero-filled, Even_MN/K variants | `csrc/flash_attn/src/flash_fwd_kernel.h:275-277, 316-323` | k_guard, v_guard |
| T23 | access | Swizzle<3,3,3> smem layouts | `csrc/flash_attn/src/kernel_traits.h:70-72, 79-100` | swizzle |
| T24 | transfer | ldmatrix.x4 for Q, K; ldmatrix.x4.trans for V | `csrc/flash_attn/src/kernel_traits.h:40-41`, `csrc/flash_attn/src/flash_fwd_kernel.h:212` | qk_frag, v_frag |
| T25 | storage | k-slice i+1 fragment loaded before MMA i | `csrc/flash_attn/src/utils.h:149-158, 174-182` | frag_prefetch |
| T26 | transfer + sync | O -> f16 -> sO -> barrier -> 128-bit gmem stores; LSE stores | `csrc/flash_attn/src/flash_fwd_kernel.h:443-500` | epi_stage, o_guard |
| T27 | other | register allocation / unrolling by nvcc | `csrc/flash_attn/src/flash_fwd_kernel.h:308`, `csrc/flash_attn/src/utils.h:152-153` | none (ux-08) |

Excluded variants (compiled out for this workload): dropout and Return_softmax
(`csrc/flash_attn/src/flash_fwd_kernel.h:357-368`), softcap (`331-333`), alibi and local window (`mask.h`), Q in
registers / shared Q-K smem (`csrc/flash_attn/src/flash_fwd_kernel.h:259-288`), split-KV and paged KV
(`compute_attn_1rowblock_splitkv`, `csrc/flash_attn/src/flash_fwd_kernel.h:505-`).

## 3. Refinement chain

| Step | Template | | Refines |
| --- | --- | --- | --- |
| online | online_rescaled_reduction | N | two-pass softmax -> rescaled single pass |
| fuse | operator_fusion | M | QK^T, softmax, PV -> one kernel, on-chip S/P |
| out_cta | output_tiling | M | O -> 128-row tiles per (b, h) |
| kv_part | reduction_partition | M | Sk -> 64-key blocks |
| causal_prune | masked_tile_pruning | N | masked key blocks skipped / peeled |
| out_warp | output_tiling | M | CTA tile -> 4 row-slabs of 32 rows |
| mma | compute_instruction_selection | M | m16n8k16 f16 MMAs |
| p_convert | operand_precision_conversion | M | P f32 -> f16 |
| p_reuse | accumulator_operand_reuse | N | S accumulator -> PV A operand |
| exp2 | approximate_math_rewrite | N | exp -> exp2 with folded scale |
| row_xlane | cross_lane_reduction | N | quad allreduce |
| acc_store | physical_storage | M | accumulators -> registers |
| q_smem | physical_storage | M | Q -> smem, later O |
| kv_smem | physical_storage | M | K, V -> one smem slot each |
| kv_async | async_transfer | M | gmem -> smem via cp.async, staggered issue |
| kv_rotation | multi_buffer_rotation | P | single slot with one-tile lookahead |
| vec | vectorized_access | M | 128-bit thread accesses |
| k_guard, v_guard | boundary_predication | M | Sk residue |
| swizzle | shared_resource_binding | M | Swizzle<3,3,3> |
| qk_frag, v_frag | collective_fragment_load | M | ldmatrix (V transposed) |
| frag_prefetch | multi_buffer_rotation | M | k-slice lookahead in registers |
| epi_stage | staged_relayout | M | O through sO |
| o_guard | boundary_predication | M | Sq residue on O / LSE |

Match rate 19/25. 8 unmatched transactions (6 steps + T03 + T27), each with a Gap.

## 4. Least certain judgements

1. **fuse matched before online is proposed.** operator_fusion's precondition requires the
   consumer's reduction to be rescalable; the chain therefore puts `online` first. Counting fuse
   as matched is fair (the Template existed), but the fusion is only legal because of a Template
   proposed in this same reference.
2. **frag_prefetch as a matched multi_buffer_rotation.** Every k-slice has its own registers, so
   there is no slot reuse inside one gemm call; the binding S = number of slices, P = 1 satisfies
   the obligation trivially. It may be more honest to treat this as instruction ordering and drop
   the step.
3. **kv_rotation partial.** The judgement that the seed obligation does not cover a mid-iteration
   barrier depends on the assumed variable set of multi_buffer_rotation (S, P, beta). If WP4 gives
   the Template a barrier-position variable, this becomes a match.
