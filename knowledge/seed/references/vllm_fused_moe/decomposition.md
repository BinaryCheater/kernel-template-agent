# vLLM fused MoE grouped GEMM — decomposition

Source: vllm-project/vllm @ `2eae56c4ba0cf12087fec035514eedede83c257d`. Programs: the CUDA routing
metadata kernels `moe_align_block_size_kernel` / `count_and_sort_expert_tokens_kernel`
(`csrc/moe/moe_align_sum_kernels.cu:134-229, 340-395`) and the Triton `fused_moe_kernel`
(`vllm/model_executor/layers/fused_moe/fused_moe.py:297-621`), unquantized f16 path (no SWAP_AB,
no TMA descriptors), launched by `invoke_fused_moe_triton_kernel`
(`vllm/model_executor/layers/fused_moe/fused_moe.py:774-`).

Pool: seed + proposals of the first five references. Status: analysis of source; not built or
run. ISA eligibility for GB10: Triton targets sm_12x; the metadata kernel uses cub and atomics.

## 1. Workload

Given routing (topk_ids, topk_weights) for T tokens and E experts:
`C[t, j, :] = A[t, :] @ W[topk_ids[t, j]]^T`, optionally times `topk_weights[t, j]` (second GEMM of
the MoE layer). Group sizes (tokens per expert) are data dependent and uneven. f16 in and out, f32
accumulation.

## 2. Transaction inventory

| # | Class | Transaction | Where | Explained by |
| --- | --- | --- | --- | --- |
| T01 | metadata | init sorted_token_ids with the padding id numel | `csrc/moe/moe_align_sum_kernels.cu:152-159` | align |
| T02 | metadata | per-expert counts with smem atomics | `csrc/moe/moe_align_sum_kernels.cu:172-189` | none (ux-01) |
| T03 | metadata | pad counts to BLOCK_SIZE_M, exclusive scan, total | `csrc/moe/moe_align_sum_kernels.cu:192-211` | align (scan itself: ux-02) |
| T04 | metadata | expert_ids per M-block; inactive blocks marked | `csrc/moe/moe_align_sum_kernels.cu:214-229` | align |
| T05 | metadata | scatter pair ids by atomic rank into sorted_token_ids | `csrc/moe/moe_align_sum_kernels.cu:340-368` | align |
| T06 | schedule | grid = upper bound EM / BLOCK_M x N / BLOCK_N; blocks past num_tokens_post_padded exit | `vllm/model_executor/layers/fused_moe/fused_moe.py:404-406, 845-850` | align (padded_upper_bound) |
| T07 | schedule | pid -> (pid_m, pid_n) grouped by GROUP_SIZE_M | `vllm/model_executor/layers/fused_moe/fused_moe.py:388-396` | raster |
| T08 | schedule | block's expert from expert_ids; -1 writes zeros | `vllm/model_executor/layers/fused_moe/fused_moe.py:423-439` | align |
| T09 | transfer | A rows gathered through sorted ids // top_k | `vllm/model_executor/layers/fused_moe/fused_moe.py:407-421, 473-475` | gather |
| T10 | transfer | B tile of the block's expert | `vllm/model_executor/layers/fused_moe/fused_moe.py:476-480, 540-542` | out_cta, ab_async |
| T11 | reduce | K loop in BLOCK_SIZE_K steps, tl.dot into f32 accumulator | `vllm/model_executor/layers/fused_moe/fused_moe.py:515-578` | k_part, mma, acc_store |
| T12 | transfer | masks: token_mask and K residue on loads; token_mask and N on stores | `vllm/model_executor/layers/fused_moe/fused_moe.py:419-421, 534-542, 618-621` | a_guard, out_guard |
| T13 | compute | routed weight multiply in f32, cast, scattered store | `vllm/model_executor/layers/fused_moe/fused_moe.py:604-621` | epi |
| T14 | lowering | smem layout, swizzle, ldmatrix / MMA, cp.async pipeline, registers (by Triton) | `vllm/model_executor/layers/fused_moe/fused_moe.py:521-578` | ab_async, ab_stages, mma bind the intent; realization: none (ux-03) |
| T15 | schedule | config dispatch (tuned JSON per device, default by M) | `vllm/model_executor/layers/fused_moe/fused_moe.py:1130-1160, 1335-1455` | none (ux-04) |

## 3. Refinement chain

| Step | Template | |
| --- | --- | --- |
| align | runtime_work_descriptors (device, pad_to_block, padded grid) | M |
| gather | indexed_gather (row) | M |
| out_cta | output_tiling | M |
| raster | block_rasterization (GROUP_SIZE_M) | M |
| k_part | reduction_partition | M |
| mma | compute_instruction_selection (by compiler) | M |
| acc_store | physical_storage | M |
| ab_async | async_transfer (by compiler) | M |
| ab_stages | multi_buffer_rotation (num_stages, by compiler) | M |
| a_guard | boundary_predication (token_mask) | M |
| out_guard | boundary_predication (token_mask) | M |
| epi | operator_fusion (routed weight) | M |

Match rate 12/12. 4 unmatched transactions, none of them chain steps.

## 4. Least certain judgements

1. **Counting compiler-realized steps as matched.** mma, ab_async and ab_stages are bound from the
   kernel's hints (tl.dot, num_stages) and the known behaviour of Triton's pipeliner, not from code
   the author wrote. They are matched in the sense that the pool can express them, but their
   bindings are unverified until compiled IR or SASS is attached (ux-03).
2. **align as one step.** The metadata kernel contains its own reduce (histogram), scan and
   scatter. They were folded into runtime_work_descriptors (its program), with the histogram and
   scan recorded as unmatched transactions because they are real transactions that the pool cannot
   describe. Alternatively the metadata kernel could be a second Reference with its own chain.
3. **No Template for the atomic histogram.** Recorded as a Gap only. It is the first atomic-based
   combine in the batch; one occurrence was judged too few to fix a Template's variables.
