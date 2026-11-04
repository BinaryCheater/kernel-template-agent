# Current pool delta

| Need | Candidate | Useful choice |
|---|---|---|
| Reduce wide-value output accumulators without repeating QK | [attention_value_axis_partition@1](attention_value_axis_partition@1/template.yaml), [outline](attention_value_axis_partition@1/proof.yaml) | Switch from key-shard/full-feature partials to complete-key/disjoint-feature outputs through a shared common-frame weight tile |
| Reuse decoded packed operands across fragments | [decoded_tile_materialization@1](decoded_tile_materialization@1/template.yaml), [outline](decoded_tile_materialization@1/proof.yaml) | Keep fragment-local decode or materialize one shared decoded tile with identical scale/cast order |
| Combine two independent CTA work families | [two_family_cta_pool@1](two_family_cta_pool@1/template.yaml), [outline](two_family_cta_pool@1/proof.yaml) | Prefer by local-domain ratio, claim a unique family ticket and fall back once |
| Preassign ragged tasks to balanced immutable CTA lists | [cost_partitioned_worklists@1](cost_partitioned_worklists@1/template.yaml), [outline](cost_partitioned_worklists@1/proof.yaml) | Separate estimated-cost assignment from exact descriptor coverage and per-CTA prefix traversal |
| Quantize larger probability codes without changing row mass | [numerator_weight_gain@1](numerator_weight_gain@1/template.yaml), [outline](numerator_weight_gain@1/proof.yaml) | Select unit/power-of-two/format-max gain for P, then apply one inverse gain at output |
| Supply P to two disjoint value-feature owners | [score_replication_or_exchange@1](score_replication_or_exchange@1/template.yaml), [outline](score_replication_or_exchange@1/proof.yaml) | Repeat full QK/P in both owners or shard scores by keys and exchange P; keep mass multiplicity correct |
| Reduce exact top-k to a bounded candidate set | [count_certified_topk_filter@1](count_certified_topk_filter@1/template.yaml), [outline](count_certified_topk_filter@1/proof.yaml) | Certify the full threshold count or fall back to unchanged full-domain selection |
| Resolve exact score/ID boundaries | [radix_boundary_topk@1](radix_boundary_topk@1/template.yaml), [outline](radix_boundary_topk@1/proof.yaml) | Locate the score rank with prefix histograms, then select only the required largest tied IDs |
| Distribute final merge rows across clustered peers | [row_owned_cluster_state_combine@1](row_owned_cluster_state_combine@1/template.yaml), [outline](row_owned_cluster_state_combine@1/proof.yaml) | Row-packed remote raw-state partials, unique row owners and publication/retirement cluster joins |
| Avoid torn records when append destinations repeat | [first_occurrence_record_scatter@1](first_occurrence_record_scatter@1/template.yaml), [outline](first_occurrence_record_scatter@1/proof.yaml) | First-input ownership by uniform earlier scan or reset/atomic-min/write passes; whole-record completion precedes readers |
| Compact sparse support rows while retaining semantics | [stable_sparse_support_compaction@1](stable_sparse_support_compaction@1/template.yaml), [outline](stable_sparse_support_compaction@1/proof.yaml) | Preserve original-slot validity, duplicate occurrences and shared forward/backward row generations |
| Share sparse sets through a common union | [grouped_sparse_union_masks@1](grouped_sparse_union_masks@1/template.yaml), [outline](grouped_sparse_union_masks@1/proof.yaml) | Bitmap logical-key union and per-row masks preserve support through shared payloads |
| Change key range before quantization without changing ideal row probabilities | [softmax_key_centering@1](softmax_key_centering@1/template.yaml), [outline](softmax_key_centering@1/proof.yaml) | One common key mean, local encoding scales and explicit original-frame logit/LSE compensation |
| Map persistent contiguous CTA intervals to group partials | [contiguous_group_tile_walk@1](contiguous_group_tile_walk@1/template.yaml), [outline](contiguous_group_tile_walk@1/proof.yaml) | Exact inverse owner and derived variable segment slots replace a fixed split count or task counter |
| Encode centered query residuals without dropping mean scores | [query_center_residual_scores@1](query_center_residual_scores@1/template.yaml), [outline](query_center_residual_scores@1/proof.yaml) | Compact FP32 key-dependent correction per head/query block, combined before normalization |
| Share low-bit probability scales across K16 groups | [tile_anchor_probability_encoding@1](tile_anchor_probability_encoding@1/template.yaml), [outline](tile_anchor_probability_encoding@1/proof.yaml) | Local score anchors generate E2M1 codes and repeated E4M3 scales while mass stays unrounded |
| Restore tile-centered V | [tile_value_center_restoration@1](tile_value_center_restoration@1/template.yaml), [outline](tile_value_center_restoration@1/proof.yaml) | Keep pending cast-P tile masses in the online frame and restore16 means per matrix group |
| Replay selected value quantization residuals | [selected_value_residual_replay@1](selected_value_residual_replay@1/template.yaml), [outline](selected_value_residual_replay@1/proof.yaml) | Copied key scores drive extra PV terms while only original rows normalize the result |
| Reuse a fixed partial-state capacity across live counts | [runtime_partition_capacity@1](runtime_partition_capacity@1/template.yaml), [outline](runtime_partition_capacity@1/proof.yaml) | Separate useful aligned partitions, capacity strides and collective participation |
| Refine sparse blocks across cache-page boundaries | [semantic_block_page_fragments@1](semantic_block_page_fragments@1/template.yaml), [outline](semantic_block_page_fragments@1/proof.yaml) | Map each gcd-aligned fragment to a physical page/subpage locator, with explicit sentinel decode |
| Preserve streaming pair identity across packed call boundaries | [stream_pair_boundary_state@1](stream_pair_boundary_state@1/template.yaml), [outline](stream_pair_boundary_state@1/proof.yaml) | Stable stream IDs and absolute even tags, with old-carry readers before new-tail commit |
| Contract a scaled weight high code and signed low code separately | [residual_code_contraction@1](residual_code_contraction@1/template.yaml), [outline](residual_code_contraction@1/proof.yaml) | FP32 component-result addition precedes one decoder scale, while unquantized denominator mass stays separate |
| Separate multicast payload and instruction-atom barrier ownership | [atom_owned_cluster_multicast@1](atom_owned_cluster_multicast@1/template.yaml), [outline](atom_owned_cluster_multicast@1/proof.yaml) | Independent payload/full-owner/empty-release maps and fallback-aware atom arrival counts |
| Stage several arithmetic steps of metadata in one slot | [multi_step_metadata_staging@1](multi_step_metadata_staging@1/template.yaml), [outline](multi_step_metadata_staging@1/proof.yaml) | Group transfers and handshakes, preserve individual scale steps and retire all window slices before reuse |
| Choose runtime or prepared low-rank correction scales | [low_rank_scale_placement@1](low_rank_scale_placement@1/template.yaml), [outline](low_rank_scale_placement@1/proof.yaml) | Independent scales versus one BF16 factor fold and common alpha, with coherent parameter snapshot |
| Merge independent FP32 partial slots reproducibly | [ordered_workspace_combine@1](ordered_workspace_combine@1/template.yaml), [outline](ordered_workspace_combine@1/proof.yaml) | Dependency-separated ascending split merge, with exact range-hole predicates and one final cast |
| Quantize complete MX groups with a floor-exponent scale | [floor_anchor_group_quantization@1](floor_anchor_group_quantization@1/template.yaml), [outline](floor_anchor_group_quantization@1/proof.yaml) | Format-fixed anchor256/4, complete32-element maxima, explicit clipping and normal scale/reciprocal range |
| Share tiny statistics across CTAs that continue disjoint output shards | [replicated_cluster_partials@1](replicated_cluster_partials@1/template.yaml), [outline](replicated_cluster_partials@1/proof.yaml) | Replicate each warp tuple to every peer, match transaction bytes and keep dependent phases in separate banks |
| Compress independently consumed variable-length boolean masks | [segmented_bit_packing@1](segmented_bit_packing@1/template.yaml), [outline](segmented_bit_packing@1/proof.yaml) | Reset at segment byte origins, preserve original lengths and order, and zero-fill short tails |
| Supply original feature pairs to a rotation | [paired_transform_supply@1](paired_transform_supply@1/template.yaml), [outline](paired_transform_supply@1/proof.yaml) | Local co-ownership, duplicate loads or converged lane exchange; preserve pairing, input casts and alias read ordering |
| Sample a threshold support without materializing its sorted list | [adaptive_support_rejection@1](adaptive_support_rejection@1/template.yaml), [outline](adaptive_support_rejection@1/proof.yaml) | Certified superset pruning, original full-row membership, exact surviving mass and fresh conditional proposals |
| Reuse a draft prefix while preserving a target token law | [draft_target_residual_correction@1](draft_target_residual_correction@1/template.yaml), [outline](draft_target_residual_correction@1/proof.yaml) | Target/draft acceptance, first-reject positive residual, target bonus and separate diagnostic counts |
| Find a top-p cutoff by cumulative mass rather than count | [mass_rank_radix_boundary@1](mass_rank_radix_boundary@1/template.yaml), [outline](mass_rank_radix_boundary@1/proof.yaml) | Radix mass/count histograms, residual target, guarded compaction/rescan and inclusive boundary cohort |
| Sum positive equal-exponent FP32 histogram entries reproducibly | [shared_exponent_integer_sum@1](shared_exponent_integer_sum@1/template.yaml), [outline](shared_exponent_integer_sum@1/proof.yaml) | Exact fraction/count atomics, hidden-bit carries and one declared toward-zero reconstruction |
| Build reproducible prefixes from mixed-exponent row probabilities | [row_relative_fixed_point_mass@1](row_relative_fixed_point_mass@1/template.yaml), [outline](row_relative_fixed_point_mass@1/proof.yaml) | Align significands to row-maximum unit, quantify downward mass loss and keep integer scans separate from floating/RNG consumers |
| Mix cluster-cooperative and independent CTA rows in one launch | [cluster_granularity_switch@1](cluster_granularity_switch@1/template.yaml), [outline](cluster_granularity_switch@1/proof.yaml) | Stable request classes, exact row expansion, uniform cooperative branches and peer-free short tails |
| Commit only an accepted speculative recurrent prefix | [deferred_prefix_state_commit@1](deferred_prefix_state_commit@1/template.yaml), [outline](deferred_prefix_state_commit@1/proof.yaml) | Freeze the original checkpoint, retain snapshots or complete update inputs, and materialize one consecutive prefix once |
| Turn short recurrent delta feedback into contractions | [chunked_delta_feedback@1](chunked_delta_feedback@1/template.yaml), [outline](chunked_delta_feedback@1/proof.yaml) | Solve the causal weighted feedback system, then form value residuals or state-independent query coefficients |
| Preserve distinct key-channel decays through delta feedback | [chunked_delta_feedback@2](chunked_delta_feedback@2/template.yaml) | Keep cumulative channel frames inside pairwise dots; equal-channel gates recover version1 |
| Increase parallel work before an ordered recurrent chain | [state_independent_chunk_factors@1](state_independent_chunk_factors@1/template.yaml) | Prepare Q/K/gate-only W/P factors across chunks, publish all records, then consume original values and state in order |
| Chunk several ordered state updates per real token | [grouped_state_substeps@1](grouped_state_substeps@1/template.yaml) | Preserve one decay, last-update readout and token-prefix units even when64-update chunks cut inside a token |
| Split a forgetting state chain with an explicit error budget | [bounded_history_warmup@1](bounded_history_warmup@1/template.yaml) | Admit zero-seed warm-up cuts under state/transport/readout bounds; failed cuts extend the previous owner |
| Keep recent accepted recurrent updates without dense state rewrites | [low_rank_state_journal@1](low_rank_state_journal@1/template.yaml) | Project checkpoint plus low-rank correction history, append drafts beyond live readers and periodically fold accepted history |
| Supply entry states to parallel recurrent pieces | [affine_piece_prefix_state@2](affine_piece_prefix_state@2/template.yaml) | Choose repeated prefix replay or one shared ordered fixup; load exclusive entries from inclusive boundary records |
| Reserve ragged chunk records without another prefix scan | [prefix_bounded_ragged_slots@1](prefix_bounded_ragged_slots@1/template.yaml) | Derive safe noncompact starts from token prefixes and sequence indices; keep capacity gaps unread |
| Carry recurrent state through packed sequences sharing physical blocks | [chunked_decayed_outer_product_state@2](chunked_decayed_outer_product_state@2/template.yaml) | Clip both pairwise axes, rebase gates at actual sequence subranges and retain independent sequence state |
| Narrow a checkpoint with a declared random-carry law | [random_discarded_bit_rounding@2](random_discarded_bit_rounding@2/template.yaml) | Preserve FP16/BF16 exponent recipe, paired random fields and logical RNG epoch; distinguish normal mean from subnormal preparation |
| Reuse recurrent prefixes through paged boundary states | [paged_recurrent_boundary_state@1](paged_recurrent_boundary_state@1/template.yaml) | Share completed prefix pages read-only, continue private tails and prepare absolute checkpoint/read/write rows once per immutable step |
| Keep independent state rows on matching memory and execution domains | [locality_matched_state_shards@1](locality_matched_state_shards@1/template.yaml) | Shard V with complete K, preserve padded physical strides and join every domain before publishing state |
| Replay large recurrent training tapes from smaller saved boundaries | [boundary_seeded_tape_replay@1](boundary_seeded_tape_replay@1/template.yaml) | Keep full tapes, entry states only, or no forward artifacts; preserve reverse adjoint exchange in all modes |
| Resolve future recurrent gradients across rank fragments | [reverse_affine_boundary_transport@1](reverse_affine_boundary_transport@1/template.yaml) | Head-owner gather/chain/scatter or bounded document-neighbor hops, with explicit saved/resent map orientation |
| Backpropagate a normalized row shared by multiple consumer heads | [shared_normalization_gradient_merge@1](shared_normalization_gradient_merge@1/template.yaml) | Group-owned or materialized FP32 head sums, then raw/statistic pullback and one final BF16 cast |
| Reuse checkpoint replay work within its backward lifetime | [checkpoint_recompute_liveness@1](checkpoint_recompute_liveness@1/template.yaml) | Separate proved-dead output elision from bounded tape handoff, with last-saver-only automatic retention |
| Convolve rank fragments with short multi-source histories | [document_stencil_halo_pullback@1](document_stencil_halo_pullback@1/template.yaml) | Absolute document-row routing, raw FP32 dpre return and per-operand versus channel-packed exchange |
| Finalize per-CTA partials inside or after the producer grid | [partial_grid_join_placement@1](partial_grid_join_placement@1/template.yaml) | Cooperative join or following reducer, with actual CTA budget, tile-specific live rows and numeric partition identity |
| Reuse a dynamic token queue across ordered invocations | [self_resetting_partition_queue@1](self_resetting_partition_queue@1/template.yaml) | Per-partition two-lookahead tickets and last-claimer reset, with output completion kept separate |

[Prefill source binding](../trace-prefill-value-partition.md) distinguishes the
derived source dispatch from the abstract ownership choice, and records FP8
gain/layout differences outside this version. The
[logical check](../checks/check_classic_attention.json) covers the all-key
feature contraction and streamed correction; native exchange/MMA evidence is
still an obligation. Classic decode and cascade add concrete reuse instances,
not duplicate candidates. No Store admission is performed.

[Quantized prefill](../trace-quantized-prefill.md) and
[POD/Hopper](../trace-pod-and-hopper.md) bind the two additional candidates.
The [second logical check](../checks/check_repack_and_pool.json) covers decoder
coordinates/rounding and small serialized ticket preferences. All twenty-six pairs
pass [schema/pair/expression validation](../inputs/schema-check.json).
[Hopper worklists/gain](../trace-hopper-worklists-and-gain.md) resolves the two
abstract gaps as separate candidates. Their [logical check](../checks/check_worklists_and_gain.json)
covers descriptor coverage/replication and gained-weight rounding. Native
publication, per-task retirement, MMA permutations and numeric error bounds
remain implementation obligations.

[Classic MLA](../trace-mla-score-supply.md) adds the two-owner score-supply
candidate; its [logical check](../checks/check_score_supply.json) binds rounded
P mass and complete-key feature outputs through a reference change. This does
not widen attention_value_axis_partition@1's four-owner domain or identify
rounded-P mass with numerator_weight_gain@1's pre-cast denominator.

[DSA selection](../trace-exact-indexer-selection.md) adds independent count
certification and radix boundary selection. The
[logical check](../checks/check_exact_selection.json) validates finite-score
membership and tie witnesses. Resident overlap, coarse-region decode and native
phase-publication protocols remain named realization work.

[Sliding-window MLA](../trace-sliding-window-mla.md) binds the M8S3 cluster rule
and compares its payload/lifetimes with global and natural-fragment relatives.

[FP8/cache writers](../trace-fp8-correction-and-cache-writes.md) binds record
election and preserves numeric/reference and cache-format boundaries.

[Sparse training/support](../trace-sparse-support-and-training.md) binds the
stable compaction candidate and existing checkpoint/recomputation/scatter rules.

[Union/quantized variants](../trace-group-union-and-quantized-variants.md) binds
logical versus physical union order and the explicit set-semantics premise.

[Key centering](../trace-key-centering.md) distinguishes operand translation
from online reference changes and gained-probability state.

[QSA consumers/walk](../trace-qsa-scale-gate-and-tile-walk.md) binds the
contiguous assignment, scale scopes and required gate-rounding checkpoint.

[Query correction/device routes](../trace-device-attention-and-query-centering.md)
keeps query-mean recovery distinct from discarded key-row shifts.

[SM110/MiniMax](../trace-sm110-and-minimax-probabilities.md) binds independent tree page maps and extracts local-anchor P encoding with original mass.


[VC compensation](../trace-vc-value-compensation.md) binds deferred tile means and selected numerator-only repair rows, including relocated original tails.

[Runtime splits/fragments](../trace-runtime-splits-and-fragments.md) supplies typed live-prefix, fixed-topology and block/page refinement bindings.

[B12x stream compression](../trace-b12x-stream-compression.md) binds pair carry phases and the pooled BF16 checkpoint.

[B12x dual-P/selection](../trace-b12x-residual-p-and-selection.md) binds signed high/low operand products and stable threshold ties.

[GEMM scope/cluster ownership](../trace-gemm-scope-and-cluster-ownership.md) supplies atom-owned multicast and concrete scale/segment reuse.

[Grouped promotion/fused output](../trace-grouped-promotion-and-fused-output.md) supplies scale-window and preserved output-cast bindings.

[SVDQuant/W4A16](../trace-svdquant-and-w4a16.md) binds scale-fold numerics, smoothing channels and native finite decoder equivalence.

[Routed GEMM/ordered wgrad](../trace-routed-gemm-and-ordered-wgrad.md) binds route/scale identities and deterministic linear partial reduction.

[MX floor-output quantization](../trace-mx-floor-output-quantization.md) separates scale selection, exponent encoding and zero/subnormal boundaries.

[Standalone quantization/norm](../trace-standalone-quantization-and-norm.md) supplies cluster tuple, scale-layout and rounded-statistic boundaries.

[Quantizer recipes/segmented masks](../trace-quantization-recipes-and-segmented-masks.md) binds NVFP4 error-selection reuse and the new byte-segment contract.

[Rotary/norm/cache](../trace-rotary-norm-and-cache.md) binds pairing, statistics/checkpoints and distinct positional/cache coordinates.

[Adaptive sampling](../trace-adaptive-sampling.md) supplies the conditional-law proof, boundary cohort contract and exact probability-tree checks.

[Speculative correction](../trace-speculative-chain-correction.md) binds probability/history, counter and endpoint boundaries; [Gumbel/composition](../trace-gumbel-and-sampler-composition.md) records finite-noise and support-order differences.

[Weighted radix/exact selection](../trace-weighted-radix-and-selection.md) supplies weighted crossing and concrete exact-selector reuse boundaries.

[Shared-exponent sum/logit pipeline](../trace-shared-exponent-and-logit-pipeline.md) supplies the deterministic sum codec and typed-fusion/softmax dispositions.

[Fixed-point sparse sampling](../trace-fixed-point-sparse-sampling.md) binds the mixed-exponent codec and shows FP64-prefix/output-numerator boundaries.

[GVR/granularity](../trace-gvr-and-cluster-granularity.md) supplies the hybrid row map and count-crossing/escape dispositions.

[Deferred recurrent state](../trace-deferred-recurrent-state.md) records frozen verification, snapshot/tape retention, accepted-prefix identity and one-use state publication. The exact check covers80 prefixes and440 tracking cases.

[Chunked delta feedback](../trace-chunked-delta-feedback.md) binds residual-first Hopper prefill and coefficient-first frozen MTP output. Exact checks cover96 recurrence/chunk cases,48 padded extents and48 two8-row weighted solves. Recursive full inverse versus weighted solve remains an explicit existing-rule extension.

[Prepared recurrent factors](../trace-prepared-recurrent-factors.md) binds state-independent W/P preparation, whole-workspace publication and the serial64-value-row state chain. Checks cover32 recurrence comparisons and labeled decay/W/pairwise coordinates, including the obsolete-header map counterexample.

[GDP substeps](../trace-gdp-substeps.md) binds direct n2 decode and SM90 expanded64 prefill. Checks compare56 recurrences,1960 accepted prefixes and source readout/causal coordinates across within-token chunk boundaries.

[Bounded history warm-up](../trace-bounded-history-warmup.md) binds the merged GDP split and exposes gate-only applicability gaps. Exact checks cover180 state/360 readout bounds,255 active-cut partitions and large-state/amplifying-key counterexamples.

[Low-rank state journal](../trace-low-rank-state-journal.md) binds virtual projections, accepted-history fold,32-slot reader-safe append and once-per-step shared cursors. Checks cover476 algebra cases and all legal cursor/threshold combinations.

[CP affine fixup/ragged storage](../trace-cp-affine-and-ragged-slots.md) extends the retained matrix-affine prefix rule and extracts conservative slot addressing. Checks cover32 final states,88 piece entries and10878 ragged partitions; native summary/fixup precision remains explicit.

[SSD logical chunks](../trace-ssd-logical-chunks.md) extends the retained complete-chunk rule with clipped packed domains and concrete inter-chunk state propagation. Checks compare162 sequences and shared-block/frame ownership; native dt/IO rounding remains explicit.

[State rounding/random coordinates](../trace-state-rounding-and-random-coordinates.md) extends the pinned BF16 carry rule to the source FP16 recipe and binds UMMA conversion fields/epochs. Exact checks expose subnormal prepared-mean loss and verify load/head/request coordinate regrouping.

[Paged recurrent boundaries](../trace-paged-recurrent-boundaries.md) binds split prefix/checkpoint/final rows and shared metadata generation. Checks cover9652 prefill cases,14080 absolute checkpoints and46 decode boundaries, including unaligned prefixes and private tails.

[Locality state shards](../trace-locality-state-shards.md) binds physical-domain allocations, green-context launch domains and complete fork/join. Checks cover98304 labeled state coordinates and256 FP32 transpose-reduce lane results; target locality/lifetime remains explicit.

[Boundary-seeded tape replay](../trace-boundary-seeded-tape-replay.md) binds full, entry-boundary and no-artifact retention. Exact checks compare51 replay cases and440 polynomial gradient entries; reverse adjoint communication remains required in every mode.

[Reverse affine boundary transport](../trace-reverse-affine-boundary-transport.md) binds head-owner and bounded neighbor paths, saved/resent map orientation and exclusive future seeds. Checks cover48 owner chains,24 short-hop chains and96 exact seed derivatives.

[Shared normalization gradient merge](../trace-shared-normalization-gradient-merge.md) binds grouped TMEM versus materialized partials, fixed group sum and raw/saved-r pullback before one final cast. Checks cover48 exact Jacobian cases and32 finite-stage cases.

[Checkpoint recompute liveness](../trace-checkpoint-recompute-liveness.md) separates output deadness, task-bound tape handoff and peak-lifetime policy. Logical checks cover384 replay schedules and illustrative lifetime/eviction cases; native torch frame integration remains explicit.

[Document stencil halos](../trace-document-stencil-halo-pullback.md) adds absolute-token history/transpose routing and channel-packed exchange. Checks cover510 fragmented stencils and11294 transpose edges. [Activated gate suffix pullback](../trace-gdn-gate-prefix-pullback.md) records existing affine/ownership rule bindings with195 exact length/seed cases.

[Partial-grid join placement](../trace-partial-grid-join-placement.md) binds cooperative versus following-kernel completion, actual grid geometry and live partial rows. Checks cover292 layouts and19080 exact token partitions, with a cross-partition rounding counterexample.

[Self-resetting partition queue](../trace-self-resetting-partition-queue.md) binds the zero-history forward ticket/retirement protocol. Checks explore29448 abstract states and distinguish counter reuse from output completion.

## User-directed convergence checkpoint

Study partial and paused. The [study checkpoint](../study.md) records remaining families and the exact resume point. [rematerialized_activation_gradient_recipe@1](rematerialized_activation_gradient_recipe@1/template.yaml) and its [outline](rematerialized_activation_gradient_recipe@1/proof.yaml) are preserved as the57th draft pair, schema-validated but without a completed source trace or focused stage check. The preceding56 entries remain indexed proposals; this draft is not closure evidence.
