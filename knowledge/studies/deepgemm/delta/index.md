# Proposed delta — DeepGEMM traced mechanisms

The library study remains active. Nine proposed rules are available in the traced slices:
[block_scaled_partial_accumulation@1](block_scaled_partial_accumulation@1/template.yaml),
with its [outline](block_scaled_partial_accumulation@1/proof.yaml). It introduces the
raw-to-scaled partial accumulator. `fused_dequantization@1` applies scales before MMA or
per-channel in the epilogue and cannot express the intervening K-partial promotion;
`reduction_partition@1` expresses the partition but lacks scale ownership and promotion.
No old identity is replaced. Assurance remains declared.

Reuse: `reduction_partition@1` (seed), `persistent_tile_scheduler@1`,
`warp_specialization@1`, `register_rebalancing@1`, `per_slot_handshake@1`,
`cluster_multicast@1`, `staged_relayout@1`, `boundary_predication@1` (reference candidates),
and seed `multi_buffer_rotation@2` explain parts of the SM90 trace. Exact rule paths and
reference versions are pinned in inputs/pool-snapshot.json. Reference candidates remain
provisional. Grouped layout mapping does not yet have a complete parameter binding:
`runtime_work_descriptors@1` describes emitted arrays, whereas this scheduler reads existing
metadata in each persistent CTA. Keep that distinction open instead of inventing a binding.

The chain is intentionally partial: partition → scaled promotion, with metadata and
scheduler edges recorded in trace.md. The pool is unchanged; no admission is performed.

[power_of_two_scale_encoding@1](power_of_two_scale_encoding@1/template.yaml) and its
[outline](power_of_two_scale_encoding@1/proof.yaml) isolate exact metadata encoding from
packing and rounding arbitrary scales. The nearest `operand_precision_conversion@1`
changes contraction operands and has no UE8M0 metadata or endpoint domain; this proposal
introduces a separate exponent-scale code. [tensor_axis_permutation@1](tensor_axis_permutation@1/template.yaml)
and its [outline](tensor_axis_permutation@1/proof.yaml) introduce a zero-copy strided view;
output tiling's index_map does not express allocation sharing or permuted strides.

The two CUDA Realisations in realisations/ bind the post-partial SM90 algorithm and the
SM100 scale-layout encoder. SM100's scaled MMA is a separate instruction path; it does
not implement the SM90 post-partial numerical policy. Existing storage rules describe
subregion lifetime when bound to individual intervals; no new aliasing rule is proposed.
Packing across ragged per-group word boundaries remains an explicit domain-extension gap.

[shared_index_union@1](shared_index_union@1/template.yaml) and its
[outline](shared_index_union@1/proof.yaml) preserve two consumers' independent slot maps
while sharing their sorted union. `indexed_gather@1` addresses payload movement and
`runtime_work_descriptors@1` addresses task planning; neither has this incidence object.
The [attention partial chain](attention-chain.yaml) composes union descriptors with a
shared gather after physical coordinate translation.

[nonnegative_scale_hoisting@1](nonnegative_scale_hoisting@1/template.yaml) and its
[outline](nonnegative_scale_hoisting@1/proof.yaml) expose placement after ReLU or after a
weighted local sum. Its nonnegative/shared-scale premise distinguishes it from unrestricted
contraction scale factoring. The source binds after_reduction on SM90; other placements are
algebraic freedoms conditional on the numerical contract. These two rules have CUDA
Realisations and explicit valid/invalid premise witnesses. No existing identity is replaced.

[epoch_tagged_completion@1](epoch_tagged_completion@1/template.yaml) and its
[outline](epoch_tagged_completion@1/proof.yaml) join separately stored producer partials
using a launch epoch and exact arrival count. Per-slot handshakes protect ring-slot reuse;
this rule distinguishes stale completion across graph replays and requires all users to
retire before the epoch advances. It does not prescribe numerical partial-combine order.

[row_column_rescaling@1](row_column_rescaling@1/template.yaml) and its
[outline](row_column_rescaling@1/proof.yaml) preserve the requested finite 4-by-4
coefficient policy, including its different first row stage and epsilon placement.
A generic reduction or pointwise map does not specify these alternating dependent stages.
The iteration count is a workload boundary rather than a free convergence knob.

Both have CUDA Realisations, bringing the total to six. The [MHC partial chain](mhc-chain.yaml)
connects completion to coefficient rescaling through explicitly external combination,
RMS and affine edges. Publication visibility, resident-grid progress and fast arithmetic
remain device obligations in [the source trace](../trace-mhc.md).

[sorted_run_topk@1](sorted_run_topk@1/template.yaml) and its
[outline](sorted_run_topk@1/proof.yaml) introduce locally sorted candidate runs and
per-run consumption cursors. Generic reductions do not encode the ordered k-winner
sequence, identity tie breaking or winner-only cursor advance. Its CUDA Realisation
binds four candidates per lane/run. The [gate partial chain](gate-chain.yaml) composes
raw-logit reduction and selection, leaving activation, publication and routing external.
There are now eight candidate pairs, seven Realisations and four partial Chains.

[fragment_parity_publication@1](fragment_parity_publication@1/template.yaml) and its
[outline](fragment_parity_publication@1/proof.yaml) expose partial consumer readiness
through per-fragment parity bits. Whole-slot handshakes and all-producer completion
joins do not encode its fragment/read-group incidence. Its CUDA Realisation binds
BF16 one-bit or FP8 two-bit K-read groups, with ring lifetime and progress obligations
kept independent. The [MoE partial chain](moe-chain.yaml) records BF16 K64 consumption.
Current totals: nine candidate pairs, eight Realisations and five partial Chains.
MegaMoE remains partially traced; see [the continuation](../trace-moe.md).
