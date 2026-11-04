# FlashAttention: source-design pass complete

[Inventory](inventory.md) accounts for the full retained NVIDIA library scope.

| Kernel need | Rule | Useful choice |
|---|---|---|
| Pack three alternating-size pipeline operands | [alternating_size_ring_packing@1](delta/alternating_size_ring_packing@1/template.yaml) | Save shared memory while protecting all overlapping live ranges |
| Reorient a skinny decode attention chain | [transpose_attention_chain@1](delta/transpose_attention_chain@1/template.yaml) | Put tokens on the large MMA axis and retain per-head token normalization |
| Pack grouped heads into query rows | [grouped_head_row_packing@1](delta/grouped_head_row_packing@1/template.yaml) | Preserve original token/head identity in masks and metadata |
| Skip empty pipeline work | [empty_work_pipeline_projection@1](delta/empty_work_pipeline_projection@1/template.yaml) | Keep control-only events balanced or leave all omitted channel states unchanged |
| Start a contraction on a ready operand prefix | [prefix_ready_contraction@1](delta/prefix_ready_contraction@1/template.yaml) | Join prefix readiness with accumulator correction, then wait separately for tail |
| Handle empty ordinary attention support, with or without a finite sink | [empty_support_normalization@1](delta/empty_support_normalization@1/template.yaml) | Avoid undefined normalization and count sink once across split states |
| Reuse a rounded output as a more accurate training checkpoint | [rounding_residual_checkpoint@1](delta/rounding_residual_checkpoint@1/template.yaml) | Save the rounded arithmetic residual and reconstruct in the accumulator format |
| Avoid a probability-domain softmax derivative row reduction | [softmax_gradient_row_checkpoint@1](delta/softmax_gradient_row_checkpoint@1/template.yaml) | Form dot(O,dO), subtract incoming natural-LSE gradient, and reuse the center per tile |

| Add gathered-row gradients back to original storage | [indexed_additive_scatter@1](delta/indexed_additive_scatter@1/template.yaml) | Keep duplicate multiplicity, predicate invalid indices and initialize once |
| Retain forward exponent tiles for a later pass | [shifted_weight_checkpoint@1](delta/shifted_weight_checkpoint@1/template.yaml) | Save each tile's actual reference and decode with final row LSE |
| Bound a multi-kernel transient | [consumer_complete_scratch_chunking@1](delta/consumer_complete_scratch_chunking@1/template.yaml) | Drain every consumer before reuse and preserve global mask coordinates |
| Replay a random mask without retaining its full tensor | [counter_indexed_mask_replay@1](delta/counter_indexed_mask_replay@1/template.yaml) | Save an invocation descriptor and preserve exact counter/component identities |
| Carry a mask with a nonnegative payload | [sign_embedded_predicate@1](delta/sign_embedded_predicate@1/template.yaml) | Use the sign, with explicit signed-zero decoding or zero-insensitive consumers |
| Trade saved MLP intermediates for backward work | [affine_activation_rematerialization@1](delta/affine_activation_rematerialization@1/template.yaml) | Retain both values, the affine value, or immutable affine inputs |
| Keep cross-entropy logits sharded by vocabulary | [vocabulary_partitioned_loss@1](delta/vocabulary_partitioned_loss@1/template.yaml) | Merge row statistics and add the global normalizer/penalty once |

One composition: use online_rescaled_reduction@2 for positive-mass streamed
state, per_slot_handshake@1 plus multi_buffer_rotation@2 for two KV slots, and
cross_iteration_software_pipelining@1 for QK(next)/PV(previous). The empty-row
rule supplies the final zero-support branch. Complete-frame correction must
apply only to old contributions, after the right async PV completes or through
an earlier saved scale tied to the next consumed P generation.

Evidence: [SM90 source trace](trace-cute-sm90-forward.md), [logical check](checks/check_empty_support.py)
and its [observations](checks/check_empty_support.json). The check covers eight
ordinary-support/sink/gain cases, masked prefixes and a duplicate-sink witness.
It is not GPU execution or numerical certification. Remaining library families,
backend mapping and native sync/precision evidence are open. [SM100 representative](trace-cute-sm100-forward.md) records the role graph,
common-gain binding and prefix frontier. Fifteen proposed pairs pass
[schema/pair/expression checks](inputs/schema-check.json). The [forward variants](trace-cute-forward-variants.md) account for SM80/SM120,
twoCTA, S ping-pong, storage and math differences, including the hybrid-exp2
masked-zero premise. [SwapAB decode](trace-swapab-decode.md) adds the transposed
chain. The [geometry check](checks/check_alternating_storage.json) verifies
packed ranges and exposes a cross-slot lifetime counterexample. The [scheduler/indexing](trace-scheduler-and-indexing.md),
[sparse support](trace-sparsity.md) and [MLA forward](trace-mla-forward.md) traces
now account for those distinct choices and name their mapping/protocol gaps.
The [backward trace](trace-cute-backward.md) binds probability recomputation and
gradient ownership across SM80/SM90/SM100/SM120, with two checkpoint rules. Its
[logical check](checks/check_backward_checkpoints.json) separates arithmetic
residual storage from a lossless codec and checks ordinary/sink derivatives
with optional incoming LSE gradient. [Specialized backward and combine](trace-specialized-backward-and-combine.md)
accounts for head256 output ownership, sparse MLA's two V derivative paths,
gather GEMMs, persistent exponent tiles, collision scatter and bounded dS scratch.
The [mapping check](checks/check_sparse_backward_mapping.json) covers separating
index/frame/chunk witnesses. [CUDA FA2](trace-cuda-fa2.md) and
[CUDA FA3](trace-cuda-fa3.md) now account for those families' independent choices.
The [dropout check](checks/check_dropout_replay.json) retains integer-generator,
threshold, signed-zero and derivative witnesses. [Auxiliary operators](trace-auxiliary-operators.md)
account for fused dense/MLP, CUDA and Triton normalization, rotary, cross entropy
and distributed wrappers. [Legacy Triton attention](trace-legacy-triton-attention.md)
accounts for older state representations and ownership choices. The
[auxiliary check](checks/check_auxiliary_choices.json) covers the three MLP
retention boundaries and eight partitioned-loss cases with gradient differences.

Remaining extraction gaps are explicit: general sparse-list classification,
output-ownership/recomputation graph selection, old normalized-accumulator and
LSE-reference recurrences, paired FP8 operand/output permutations, mutating-cache
publication, subset write ownership, and normalization/AUX-specific reconstruction.
Native inline waits, cluster/fragment layouts, stochastic/provider behavior,
numerical accuracy and performance remain implementation evidence obligations.
These gaps do not imply missing source families or independent admission.
Advance to FlashInfer. CUTLASS remains user-paused.
