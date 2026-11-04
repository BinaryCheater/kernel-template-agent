# Softmax, context-parallel state and vendor attention boundaries

Paths below are under `transformer_engine/common/` in the retained source.
Most attention computation is delegated to cuDNN; local kernels implement
normalization, state correction and data movement around it. Do not attribute
an unavailable vendor device schedule to the visible graph builder.

| Source and concrete binding | Existing rule or retained gap | Useful difference |
|---|---|---|
| `fused_softmax/scaled_masked_softmax.cu:113–123,273–340` | `cross_lane_reduction@1`, register-retained row, fused scale/mask/exp/divide | Logical subgroup width is min(next-power-of-two row length,32); small rows batch two per subgroup. Vector width follows per-lane iterations. Explicit masked elements use finite -10000; a row-max equality guard emits zero for an all-sentinel row. This is not exact -infinity masking for every possible unmasked logit range. |
| Same source backward and causal siblings | Row derivative `scale*P*(dY-sum(P*dY))`; causal bounds and tail ownership | Upper-triangle masking and bottom-right aligned causal masking differ: aligned valid columns end at row+cols-rows. They share reduction/tiling machinery, not one interchangeable mask coordinate map. RTC/static menus specialize subgroup size. |
| `fused_attn/fused_attn.cpp:468–510` and backend-selection branches | Runtime dispatch contract | F16 arbitrary-sequence or FP8 backend selected from configuration, device/runtime compatibility and graph support; unsupported configurations return a rejection reason. The name `flash_attn.cu` is not evidence of an in-repository full FlashAttention MMA implementation. |
| `fused_attn/fused_attn_f16_arbitrary_seqlen.cu:173–298` | External cuDNN SDPA graph with Q/K/V strides, causal/window alignment, ragged/padded lengths, bias, dropout and sink | `operator_fusion@1`, online state and indexed pages are conceptual matches only. cuDNN chooses its device algebra/layout/pipeline; that realization is unavailable in this tree. Sink is a denominator-only term, and dropout seed/offset is an explicit graph input. |
| `fused_attn/fused_attn_fp8.cu:286–334` | External scaled SDPA with paired Q/K/V descales, probability/output scales or MXFP8 block scales | Per-tensor graph exposes amax_s/amax_o for scale recipes; MXFP8 differs in graph inputs and which amax is visible. Quantized intermediates and saved backward stats are part of the graph contract; do not replace them with an unquantized exact-softmax claim. |
| `fused_attn/context_parallel.cu:35–43,281–385` | `split_combine_kernel@1` rescaled-merge algebra and `online_rescaled_reduction@2` frame invariant | Merge natural LSE with max+log1p(exp(min-max)); correct old/new normalized outputs by exp(LSE_part-LSE_total). This is a separate correction launch after completed step outputs. Saved old LSE must precede its update; half-sequence and packed/ragged maps must identify the same row. |
| Same file `:105–144,147–151` and gradient corrections | Segmented work decode, inverse permutation and partial add/copy ownership | Context rank owns a front chunk and a mirrored back chunk of each sequence. Boundaries must divide by 2*world_size. This balances causal work by permutation; it is not softmax pruning. Backward step partials use separately selected copy/add/zero policies. |
| `fused_attn/kv_cache.cu:56–112,196–285` | Page/row addressing and segmented packing; inverse direction uses disjoint scatter ownership | New token i appends at logical position cached_len-new_len+i, translated through page_table and page offset. THD/BSHD/SBHD source maps differ. Cache mutation requires valid pages and nonconflicting destination ownership; source reindex path has its own alias/ordering contract. |
| `fused_attn/flash_attn.cu:476–516,610–663` | `staged_relayout@1`, vectorized access and explicit transfer completion | BSHD/SBHD→BHSD, optional D padding, multiple tensor slots. SM100 TMA path requires supported type and aligned inner dimension; vectorized or shared-transpose fallback handles other layouts. Data-axis permutation and padding precede the external consumer. |

For context size 2 and one sequence of 16 tokens, each rank's local sequence
has 8 tokens: rank 0 selects [0,4) and [12,16), rank 1 selects [4,8) and [8,12).
The source's second-half index already includes the local half offset, giving
the mirrored chunk. Each original token appears once; sequence-local offsets
and cumulative global storage origins must remain separate.

For partial attention masses 1 and 3 with normalized outputs a and b, the
corrected output is a/4+3b/4. Adding outputs directly or averaging them would
discard their normalization masses. This is an existing rescaled-merge
instance, not a new attention rule. Fully empty states and source zero-times-
nonfinite guards need their declared numerical policy.

No new Template is needed for these visible attention wrappers. Native vendor
selection, exceptional mask ranges, RNG replay and cache aliasing remain named
bindings; they do not justify a source audit of every graph configuration.
