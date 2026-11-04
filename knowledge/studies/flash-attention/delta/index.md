# Current pool delta

| Need | Candidate | Useful choice |
|---|---|---|
| Begin MMA before the operand tail is ready | [prefix_ready_contraction@1](prefix_ready_contraction@1/template.yaml), [outline](prefix_ready_contraction@1/proof.yaml) | Prefix/suffix publication plus corrected-accumulator readiness |
| Finalize masked/sparse rows that may have no ordinary keys | [empty_support_normalization@1](empty_support_normalization@1/template.yaml), [outline](empty_support_normalization@1/proof.yaml) | Guard empty mass and give finite denominator-only sinks one logical owner |

This complements the positive-final-mass contract of online_rescaled_reduction@2;
it does not alter that version's finite-domain recurrence or claim generic NaN
repair. [SM90 trace](../trace-cute-sm90-forward.md) gives a dense causal witness
and reuse of existing pipeline/layout rules. No Store admission is performed.

[SM100 representative](../trace-cute-sm100-forward.md) binds the prefix rule to
a one-CTA TMEM P publication. Native inline-wait ordering and the two-CTA extension
remain open; generic fragment-parity publication is not claimed as its realization.

| Additional need | Candidate | Useful choice |
|---|---|---|
| Reduce three-slot storage for alternating operand sizes | [alternating_size_ring_packing@1](alternating_size_ring_packing@1/template.yaml), [outline](alternating_size_ring_packing@1/proof.yaml) | Uniform or phase-shifted packing, with every overlapping occupancy's lifetime protected |

[Forward variants](../trace-cute-forward-variants.md) record the two-CTA and
S-ping-pong distinctions, packed storage geometry, hybrid exp2 mask invariant,
and native ordering gaps. The geometry check establishes address bounds and a
cross-slot counterexample, not safe native source scheduling.

| Additional need | Candidate | Useful choice |
|---|---|---|
| Fill the large MMA dimension in single-position grouped decode | [transpose_attention_chain@1](transpose_attention_chain@1/template.yaml), [outline](transpose_attention_chain@1/proof.yaml) | Transpose both contractions and preserve per-head token normalization |

[SwapAB trace](../trace-swapab-decode.md) binds the rewrite and records two-stream
frame correction, head padding and one-shot barrier tails.

| Additional need | Candidate | Useful choice |
|---|---|---|
| Tile grouped query heads as sequence rows | [grouped_head_row_packing@1](grouped_head_row_packing@1/template.yaml), [outline](grouped_head_row_packing@1/proof.yaml) | Split/fold head coordinates and remap every semantic consumer |
| Skip arithmetic on empty pipeline work safely | [empty_work_pipeline_projection@1](empty_work_pipeline_projection@1/template.yaml), [outline](empty_work_pipeline_projection@1/proof.yaml) | Retain required control handshakes or leave every omitted data state unchanged |

[Scheduler/indexing](../trace-scheduler-and-indexing.md), [sparse support](../trace-sparsity.md)
and [MLA forward](../trace-mla-forward.md) record source bindings. General sparse-list
classification, hardware claim realization and multi-frontier MLA readiness
remain named extraction/binding gaps. Arithmetic residual checkpoints are
covered below after following their backward consumer.

| Additional need | Candidate | Useful choice |
|---|---|---|
| Preserve more producer precision in a low-format training checkpoint | [rounding_residual_checkpoint@1](rounding_residual_checkpoint@1/template.yaml), [outline](rounding_residual_checkpoint@1/proof.yaml) | Rounded-only or rounded-plus-residual storage, with accumulator-format reconstruction |
| Obtain a softmax derivative row center without reducing P*dP over keys | [softmax_gradient_row_checkpoint@1](softmax_gradient_row_checkpoint@1/template.yaml), [outline](softmax_gradient_row_checkpoint@1/proof.yaml) | dot(O,dO), optional natural-LSE gradient and denominator-only sink derivative |

The [backward trace](../trace-cute-backward.md) and
[checkpoint/derivative observations](../checks/check_backward_checkpoints.json)
bind both rules. Ordered bulk-add ranks/completion remain native binding gaps.
The next pass resolves indexed scatter, saved weight decoding and scratch chunking.

| Additional need | Candidate | Useful choice |
|---|---|---|
| Accumulate original-row gradients after a gather | [indexed_additive_scatter@1](indexed_additive_scatter@1/template.yaml), [outline](indexed_additive_scatter@1/proof.yaml) | Scalar or four-element FP32 atomic adds with duplicate multiplicity |
| Decode saved exponent tiles without recomputing scores | [shifted_weight_checkpoint@1](shifted_weight_checkpoint@1/template.yaml), [outline](shifted_weight_checkpoint@1/proof.yaml) | Actual per-tile reference paired with final row log-normalizer |
| Reuse bounded scratch across several consumer kernels | [consumer_complete_scratch_chunking@1](consumer_complete_scratch_chunking@1/template.yaml), [outline](consumer_complete_scratch_chunking@1/proof.yaml) | Chunk rows while retaining global semantic coordinates and all consumers |

[Specialized backward/combine](../trace-specialized-backward-and-combine.md)
binds these choices and names remaining native mapping, ordered bulk-add and
output-ownership/recomputation graph gaps. [Mapping observations](../checks/check_sparse_backward_mapping.json)
retain duplicate/invalid index, reference, rounding, causal chunk and tail witnesses.

| Additional need | Candidate | Useful choice |
|---|---|---|
| Replay a random mask from a compact checkpoint | [counter_indexed_mask_replay@1](counter_indexed_mask_replay@1/template.yaml), [outline](counter_indexed_mask_replay@1/proof.yaml) | Fixed integer generator, invocation descriptor and original coordinate/component map |
| Store a predicate with a nonnegative payload | [sign_embedded_predicate@1](sign_embedded_predicate@1/template.yaml), [outline](sign_embedded_predicate@1/proof.yaml) | Sign-bit decoding or comparison with explicit zero-insensitivity |

[CUDA FA2](../trace-cuda-fa2.md) binds the two independent choices, including
dropout's centered derivative and discrete threshold. [Logical observations](../checks/check_dropout_replay.json)
cover source integer replay and separating witnesses, not native lane lowering.
[CUDA FA3](../trace-cuda-fa3.md) records concrete reuse and the paired FP8
operand/output permutation gap alongside deterministic and scheduler differences.

| Additional need | Candidate | Useful choice |
|---|---|---|
| Move an MLP's retained training boundary earlier | [affine_activation_rematerialization@1](affine_activation_rematerialization@1/template.yaml), [outline](affine_activation_rematerialization@1/proof.yaml) | Retain affine+activation, affine only, or affine inputs with explicit replay precision |
| Compute a vocabulary-sharded loss without gathering logits | [vocabulary_partitioned_loss@1](vocabulary_partitioned_loss@1/template.yaml), [outline](vocabulary_partitioned_loss@1/proof.yaml) | Combine local LSE and additive target/smoothing statistics, then finalize once |

[Auxiliary trace](../trace-auxiliary-operators.md) binds both choices and records
the ReLU bitmask counterexample; [observations](../checks/check_auxiliary_choices.json)
cover full-value replay, changed weights and smoothed/regularized partition losses.
Fifteen pairs remain provisional; no Store admission is performed.
