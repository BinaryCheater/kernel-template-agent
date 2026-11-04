# Device attention relatives and query residual score correction

Sources below are relative to the retained [revision](inputs/sources.json).
The new [query_center_residual_scores@1](delta/query_center_residual_scores@1/template.yaml)
and its [proof](delta/query_center_residual_scores@1/proof.yaml) capture the
SM120 query preprocessing/score correction separately from key centering.

## Query centering needs a key-dependent correction

| Source decision | Candidate field / rule | Binding and boundary |
|---|---|---|
| `flashinfer/nvfp4_attention_sm120.py:120-148`: center K across tokens, pad to128; center Q per128-query block or per head | softmax_key_centering@1 then `center_scope` | K translation gives a row constant. Q translation instead removes mu_Q dot K_j, which changes with key j. The two steps cannot share one compensation assumption. Padding can influence Q means; the ideal residual identity holds for any common center, while native casts remain an approximation. |
| Same file :150-170: FP32 matmul of compact means against K, after GQA reshape, then return residual Q/K/V/correction | `pre_map`, `pre_numeric`, compact storage | Correction is[B,Hq,Qgroups,Npad], not[B,Hq,Mpad,Npad]. Its GQA reshape maps each query head/block to the proper KV head; correction is computed once for all128 rows sharing a block. FP16 correction output can overflow even if the accumulation is wider. |
| `csrc/nvfp4_attention_sm120/nvfp4_attention_sm120_binding.cu:293-299` | `c_rows`, `group_head_mapping` | Mean-block mode has Mpad/128 correction rows; head mode has one. Shape is part of the binding, not inferred from matching byte count. |
| `include/flashinfer/attention/sm120/nvfp4_attention_sm120/kernel/traits.h:164-167` | Compact shared broadcast | Shared correction row layout uses zero query-row stride. It broadcasts one key row across the Q tile; physical tile generation must match K and residual-Q consumers. |
| `compute/consumer/delta_correction.cuh:35-56`, `qk_gemm.cuh:128-150` under that source directory | `pre_merge`, `initialization_and_scale` | Correction seeds score accumulators; residual block-scaled QK accumulates afterward, then masking applies. Clearing the accumulator for the first residual MMA would silently delete the correction. Native fragment coordinates/instruction rounding remain realization obligations. |

Algebra: (q_i-mu_g) dot K_j + mu_g dot K_j = q_i dot K_j. Apply score scale
to the completed sum and normalize once. Unlike common key translation, the
second term cannot be omitted. If K here was independently centered, the result
is q dot Kcentered; restoration of original-K LSE/logits remains a separate
frame contract. A forward option returning LSE is not evidence that the original
K offset has been restored by this correction alone.

The [logical check](checks/check_query_center_correction.json) uses three
queries with shared mean[8,2], recovers complete dot scores, and distinguishes
missing or inconsistently scaled correction. A130-row query domain has block
counts[128,2]. A correction131072 is finite in FP32 but overflows FP16. It also
checks the independent key-center LSE shift. This does not execute the NVFP4
encoder or native zero-stride TMA/fragment broadcast. Source head dimensions
are64/128, targets120/121; absent CUTLASS/CCCL dependencies remain untouched.

## Balanced GQA: kernel-owned plan and online prefix decode

Route: `flashinfer.decode.prepare_balanced_batch_decode_with_kv_cache` ->
`experimental/balanced_gqa_decode/cake_backend.py` -> its generated CUDA.
Representative generated file is
`csrc/cake_balanced_gqa_decode/cake_balanced_gqa_decode_36ecd3362289b95340d6_kernel.cu`
within that package. It serves ratio8, head128, Page16 BF16 on SM100/103;
packed MTP rows combine draft tokens and heads with the original causal cutoff
L-(q_len-1-query_index).

At :1380-1464 one scheduler warp reads device lengths into shared state,
computes ceil(length/256) block-pair counts and global min/max/sum, then derives
chunk length from resident-grid/work counts and source heuristics. :1784-1864
constructs full-chunk and remainder-bucket totals. No host length mirror or
full per-work descriptor array is required by this selected source.

At :1905 onward a ticket chooses a bucket, local item and retained per-bucket
cursor, then scans32-request windows to invert cumulative work counts. Bind
online_group_work_decode@2 with warp_prefix_32 and no two-window shape cache
to the per-bucket monotone prefix lookup: counts are immutable for this launch,
zero-count requests are skipped, and a later ticket cannot move behind the
same bucket's retained prefix. Per-bucket count construction/chunk policy is
an upstream source binding; the generic rule does not invent it.

The global ticket/counter mechanism shares persistent_tile_scheduler@1's
atomic-fetch idea, but exact token decode, partial slot allocation and
self-resetting launch completion need their native binding. Completed split
states use the existing common-frame algebra. Source :562-626 publishes
partials and handles the two-chunk inline merge versus later completion paths;
completion must mean published states, not merely claimed tickets. Do not call
that epoch_tagged_completion@1's fixed counter encoding without a matching map.
Changing seq_lens between captured invocations requires re-deriving counts;
changing them during a live invocation invalidates all retained prefixes.

This gives concrete online-prefix reuse rather than a new rule for each of
four length buckets. Native ticket reset, partial release/acquire and persistent
progress remain explicit implementation obligations.

## Remaining target/model menus: distinct applicability

| Source group | Observed distinction | Disposition |
|---|---|---|
| `experimental/sm110_gqa_decode/README.md:61-80` and backend routing | Exact SM110a head128 FP16; short CUDA-core route and six tcgen05/TMA/TMEM routes,64/32 key tiles and different ring counts; explicit split menus | Existing tiling, online reduction, multi-buffer/per-slot concepts apply. Select actual route/threads/rings from the record, not the public GQA name. Shape/capacity routing is host metadata policy, not measured proof. Native representative map remains open. |
| `experimental/sm110_xqa/README.md` and `flashinfer/sm110_xqa.py` | Head512 tree speculative decode, independent K/V page tables, mask restricted to final draft block; packed queries keep maximum-Q mask stride | Existing indexed-gather/mask/grouped-row identities need distinct K/V table and draft-index bindings. Tree support is not a plain causal tail or one shared page table. Exact native even/odd TMEM/pair schedules remain separate representatives. |
| `experimental/minimax_h3_varlen_attention/README.md:10-40` | Packed THD noncausal head128, BF16, FP8 and two NVFP4 PV modes | BF16/FP8 reuse existing online/frame and descale rules. NVFP4 QK uses group16 scales; FP8 PV has unit-scale P, while FP4 PV uses a score-tile anchor for multiple K16 probability scales. That P quantizer/anchor and V-transpose grouping remain a distinct numerical/layout binding; do not treat all PV modes as the same gain parameter. |

These initial menu observations are developed in the [SM110/MiniMax follow-up](trace-sm110-and-minimax-probabilities.md), which supplies native representatives and the probability-anchor candidate. Remaining native realization gaps are named there. Reconcile remaining attention families before advancing to GEMM and the other full-library inventory groups.
