# FlashInfer: current rule menu

Study partial; paused by user direction. [Study checkpoint](study.md) records
the exact resume point and materially unfinished families. The
[inventory](inventory.md) retains the full library scope.

| Kernel need | Rule | Useful choice |
|---|---|---|
| Reduce a wide attention output's per-owner accumulator | [attention_value_axis_partition@1](delta/attention_value_axis_partition@1/template.yaml) | Share common-frame P and give each of four owners all keys for one feature slice |
| Repeatedly decode the same packed KV tile | [decoded_tile_materialization@1](delta/decoded_tile_materialization@1/template.yaml) | Decode per fragment or publish one shared FP16/BF16 tile while preserving scale/cast order |
| Colocate two independent CTA work families | [two_family_cta_pool@1](delta/two_family_cta_pool@1/template.yaml) | Domain-local preference, unique family ticket and one fallback per CTA |
| Plan unequal independent tasks before launch | [cost_partitioned_worklists@1](delta/cost_partitioned_worklists@1/template.yaml) | Greedily assign complete descriptors to least-loaded CTA lists, then traverse prefix ranges |
| Preserve more narrow probability weights | [numerator_weight_gain@1](delta/numerator_weight_gain@1/template.yaml) | Gain only P before conversion, retain original mass/LSE and compensate output once |
| Keep two value owners while reducing probability exchange | [score_replication_or_exchange@1](delta/score_replication_or_exchange@1/template.yaml) | Duplicate complete scores locally or shard keys and exchange common-frame P |
| Filter candidates before exact top-k | [count_certified_topk_filter@1](delta/count_certified_topk_filter@1/template.yaml) | Accept a complete threshold set only when its full count fits and contains enough winners; otherwise rescan exactly |
| Select a large exact top-k without sorting all scores | [radix_boundary_topk@1](delta/radix_boundary_topk@1/template.yaml) | Carry residual rank through score digits, then choose larger original IDs among boundary ties |
| Merge split attention inside a three-CTA cluster | [row_owned_cluster_state_combine@1](delta/row_owned_cluster_state_combine@1/template.yaml) | Give peers disjoint final rows; retain local partials and read raw peer states through DSM mailboxes |
| Append complete records with duplicate destinations | [first_occurrence_record_scatter@1](delta/first_occurrence_record_scatter@1/template.yaml) | Elect lowest input ID by prior scan or ordered atomic-min passes, then write every field from that winner |
| Skip holes in sparse support without changing occurrences | [stable_sparse_support_compaction@1](delta/stable_sparse_support_compaction@1/template.yaml) | Apply original-slot cuts first, stably pack valid IDs/metadata and share the same rows/counts with backward |
| Share overlapping sparse set payloads across queries | [grouped_sparse_union_masks@1](delta/grouped_sparse_union_masks@1/template.yaml) | Build one logical union and row membership masks; keep each row normalization separate |
| Remove a common key offset before low-precision encoding | [softmax_key_centering@1](delta/softmax_key_centering@1/template.yaml) | Subtract one shared key mean; preserve ideal softmax and restore original logit/LSE offsets when requested |
| Walk a grouped tile rectangle with a fixed resident grid | [contiguous_group_tile_walk@1](delta/contiguous_group_tile_walk@1/template.yaml) | Split the flattened line into contiguous CTA intervals and derive each group's partial slots |
| Encode low-bit probabilities using a local score anchor | [tile_anchor_probability_encoding@1](delta/tile_anchor_probability_encoding@1/template.yaml) | Share one scale across K16 groups while retaining unrounded denominator mass |
| Center query blocks before low-precision encoding | [query_center_residual_scores@1](delta/query_center_residual_scores@1/template.yaml) | Encode residual Q and broadcast one FP32 mean-times-K correction row before softmax |
| Restore centered narrow V without a full-width offset in every value | [tile_value_center_restoration@1](delta/tile_value_center_restoration@1/template.yaml) | Carry cast-P tile masses through reference updates and restore16 means with a final tail flush |
| Correct a few high-error V rows | [selected_value_residual_replay@1](delta/selected_value_residual_replay@1/template.yaml) | Replay copied keys with value residuals, adding numerator terms while preserving original-support mass |
| Change useful split count without changing prepared storage/topology | [runtime_partition_capacity@1](delta/runtime_partition_capacity@1/template.yaml) | Use capacity for strides/participants and the useful prefix for work and safe partial loads |
| Select sparse blocks across independent cache pages | [semantic_block_page_fragments@1](delta/semantic_block_page_fragments@1/template.yaml) | Resolve gcd-aligned fragments independently, before launch or through a direct index view |
| Keep pairwise compression invariant across request chunking | [stream_pair_boundary_state@1](delta/stream_pair_boundary_state@1/template.yaml) | Match absolute even-position carry and finish its readers before committing the next tail |
| Improve narrow P with a signed residual product | [residual_code_contraction@1](delta/residual_code_contraction@1/template.yaml) | Run high/low products separately, add FP32 results and decode scale once; keep original mass |
| Multicast across several two-CTA instruction atoms | [atom_owned_cluster_multicast@1](delta/atom_owned_cluster_multicast@1/template.yaml) | Derive payload destinations, full-barrier owners and empty release/counts independently for the active shape |
| Amortize contiguous scale transfers across future K steps | [multi_step_metadata_staging@1](delta/multi_step_metadata_staging@1/template.yaml) | Publish one G-step metadata window, retain per-step scales and release only after every slice reader finishes |
| Reuse a low-rank adapter with one common final scale | [low_rank_scale_placement@1](delta/low_rank_scale_placement@1/template.yaml) | Keep runtime scales independent or pre-fold their ratio into BF16 up weights with explicit refold/error policy |
| Keep split reduction reproducible while producers finish freely | [ordered_workspace_combine@1](delta/ordered_workspace_combine@1/template.yaml) | Store exclusive FP32 partials and merge useful slot IDs ascending, preserving holes and one final cast |
| Match an MX floor-anchor output contract | [floor_anchor_group_quantization@1](delta/floor_anchor_group_quantization@1/template.yaml) | Combine all32 values, use the format-fixed floor anchor, retain admitted clipping and normal reciprocal range |
| Let row-sharded CTAs continue after a tiny shared statistic | [replicated_cluster_partials@1](delta/replicated_cluster_partials@1/template.yaml) | All-to-all sender tuples, local complete folds and distinct phase banks with cluster retirement |
| Byte-pack short or empty boolean segments independently | [segmented_bit_packing@1](delta/segmented_bit_packing@1/template.yaml) | Separate byte prefixes and big/little bit positions with zero tails and original lengths |
| Rotate paired features without losing original partners | [paired_transform_supply@1](delta/paired_transform_supply@1/template.yaml) | Choose local vectors, repeated loads or a warp exchange, with explicit pair map, checkpoints and read-before-overwrite ordering |
| Sample top-k/top-p value cohorts without sorting the full row | [adaptive_support_rejection@1](delta/adaptive_support_rejection@1/template.yaml) | Keep target weights in each shrinking superset, update original surviving mass and use fresh categorical proposals |
| Correct a proposed draft chain to a target distribution | [draft_target_residual_correction@1](delta/draft_target_residual_correction@1/template.yaml) | Emit the consecutive accepted prefix, then positive target-minus-draft correction or target bonus; keep diagnostic counts separate |
| Find a probability-mass boundary without sorting every value | [mass_rank_radix_boundary@1](delta/mass_rank_radix_boundary@1/template.yaml) | Refine mass rank by digits, track counts separately, rescan on capacity overflow and retain all boundary ties |
| Make same-exponent histogram mass independent of atomic order | [shared_exponent_integer_sum@1](delta/shared_exponent_integer_sum@1/template.yaml) | Accumulate fraction bits and counts, then reconstruct once with explicit truncation, range and publication |
| Scan mixed-exponent probability masses reproducibly | [row_relative_fixed_point_mass@1](delta/row_relative_fixed_point_mass@1/template.yaml) | Quantize downward in a row-relative unit, preserve ordered integer prefixes and bind dropped mass plus downstream FP64/RNG policy |
| Cooperate on long rows while packing short rows per CTA | [cluster_granularity_switch@1](delta/cluster_granularity_switch@1/template.yaml) | One fixed cluster grid, prepared request classes and independent short bodies that permit partial tails |
| Commit only an accepted speculative recurrent prefix | [deferred_prefix_state_commit@1](delta/deferred_prefix_state_commit@1/template.yaml) | Freeze the original checkpoint, retain snapshots or complete update inputs, and materialize one consecutive prefix once |
| Turn short recurrent delta feedback into contractions | [chunked_delta_feedback@1](delta/chunked_delta_feedback@1/template.yaml) | Solve the causal weighted feedback system, then form value residuals or state-independent query coefficients |
| Preserve distinct key-channel decays through delta feedback | [chunked_delta_feedback@2](delta/chunked_delta_feedback@2/template.yaml) | Keep cumulative channel frames inside pairwise dots; equal-channel gates recover version1 |
| Increase parallel work before an ordered recurrent chain | [state_independent_chunk_factors@1](delta/state_independent_chunk_factors@1/template.yaml) | Prepare Q/K/gate-only W/P factors across chunks, publish all records, then consume original values and state in order |
| Chunk several ordered state updates per real token | [grouped_state_substeps@1](delta/grouped_state_substeps@1/template.yaml) | Preserve one decay, last-update readout and token-prefix units even when64-update chunks cut inside a token |
| Split a forgetting state chain with an explicit error budget | [bounded_history_warmup@1](delta/bounded_history_warmup@1/template.yaml) | Admit zero-seed warm-up cuts under state/transport/readout bounds; failed cuts extend the previous owner |
| Keep recent accepted recurrent updates without dense state rewrites | [low_rank_state_journal@1](delta/low_rank_state_journal@1/template.yaml) | Project checkpoint plus low-rank correction history, append drafts beyond live readers and periodically fold accepted history |
| Supply entry states to parallel recurrent pieces | [affine_piece_prefix_state@2](delta/affine_piece_prefix_state@2/template.yaml) | Choose repeated prefix replay or one shared ordered fixup; load exclusive entries from inclusive boundary records |
| Reserve ragged chunk records without another prefix scan | [prefix_bounded_ragged_slots@1](delta/prefix_bounded_ragged_slots@1/template.yaml) | Derive safe noncompact starts from token prefixes and sequence indices; keep capacity gaps unread |
| Carry recurrent state through packed sequences sharing physical blocks | [chunked_decayed_outer_product_state@2](delta/chunked_decayed_outer_product_state@2/template.yaml) | Clip both pairwise axes, rebase gates at actual sequence subranges and retain independent sequence state |
| Narrow a checkpoint with a declared random-carry law | [random_discarded_bit_rounding@2](delta/random_discarded_bit_rounding@2/template.yaml) | Preserve FP16/BF16 exponent recipe, paired random fields and logical RNG epoch; distinguish normal mean from subnormal preparation |
| Reuse recurrent prefixes through paged boundary states | [paged_recurrent_boundary_state@1](delta/paged_recurrent_boundary_state@1/template.yaml) | Share completed prefix pages read-only, continue private tails and prepare absolute checkpoint/read/write rows once per immutable step |
| Keep independent state rows on matching memory and execution domains | [locality_matched_state_shards@1](delta/locality_matched_state_shards@1/template.yaml) | Shard V with complete K, preserve padded physical strides and join every domain before publishing state |
| Replay large recurrent training tapes from smaller saved boundaries | [boundary_seeded_tape_replay@1](delta/boundary_seeded_tape_replay@1/template.yaml) | Keep full tapes, entry states only, or no forward artifacts; preserve reverse adjoint exchange in all modes |
| Resolve future recurrent gradients across rank fragments | [reverse_affine_boundary_transport@1](delta/reverse_affine_boundary_transport@1/template.yaml) | Head-owner gather/chain/scatter or bounded document-neighbor hops, with explicit saved/resent map orientation |
| Backpropagate a normalized row shared by multiple consumer heads | [shared_normalization_gradient_merge@1](delta/shared_normalization_gradient_merge@1/template.yaml) | Group-owned or materialized FP32 head sums, then raw/statistic pullback and one final BF16 cast |
| Reuse checkpoint replay work within its backward lifetime | [checkpoint_recompute_liveness@1](delta/checkpoint_recompute_liveness@1/template.yaml) | Separate proved-dead output elision from bounded tape handoff, with last-saver-only automatic retention |
| Convolve rank fragments with short multi-source histories | [document_stencil_halo_pullback@1](delta/document_stencil_halo_pullback@1/template.yaml) | Absolute document-row routing, raw FP32 dpre return and per-operand versus channel-packed exchange |
| Finalize per-CTA partials inside or after the producer grid | [partial_grid_join_placement@1](delta/partial_grid_join_placement@1/template.yaml) | Cooperative join or following reducer, with actual CTA budget, tile-specific live rows and numeric partition identity |
| Reuse a dynamic token queue across ordered invocations | [self_resetting_partition_queue@1](delta/self_resetting_partition_queue@1/template.yaml) | Per-partition two-lookahead tickets and last-claimer reset, with output completion kept separate |

One composition: indexed_gather@1 prefetches a horizon of page offsets;
boundary_predication@1 leaves masked K slots untouched but zero-fills V;
cross_lane_reduction@1 forms feature dot products; online_rescaled_reduction@2
and smem_partial_combine@1 form local/CTA attention states; split_combine_kernel@1
merges normalized output/base2-LSE partials. Empty sentinels need their explicit
representation mapping before empty_support_normalization@1 is applied.

[Classic decode](trace-classic-paged-decode.md) binds that composition at FP16,
D128,GQA4 and records page/offset/table generations. [Prefill ownership](trace-prefill-value-partition.md)
adds the new candidate. [Cascade state composition](trace-cascade-state-composition.md)
reuses the existing merge algebra with same-query/support and representation
premises. The [logical check](checks/check_classic_attention.json) covers page
coordinates, offset horizon refresh, complete-key feature ownership and streamed
reference correction. Forty-nine pairs pass [schema/pair/expression checks](inputs/schema-check.json).
Native layout, async publication and numerical accuracy remain obligations.

[Quantized prefill](trace-quantized-prefill.md) binds decoder placement and scale
sites. [POD and Hopper](trace-pod-and-hopper.md) binds the two-family pool,
records pipeline reuse and the source questions developed in the two new rules. Their [logical check](checks/check_repack_and_pool.json)
covers decoder coordinates, a scale-underflow witness and small serialized
preference sequences; it does not execute native GPU instructions.

[Hopper worklists and gain](trace-hopper-worklists-and-gain.md) binds both new
rules and compares their fields with the closest older versions. The
[logical witness](checks/check_worklists_and_gain.json) covers full task identity,
empty worker ranges, head replication, gained-weight rounding and original-mass
normalization across a reference change. Native list publication, pipeline
retirement, probability permutations and quantitative error bounds remain open.

[Classic MLA](trace-mla-score-supply.md) binds two feature owners, rounded-P
mass, latent reuse and the new score-supply choice. Its
[logical check](checks/check_score_supply.json) distinguishes duplicate replica
denominators and incomplete key reads. [Score-indexing scope](trace-score-indexing-scope.md)
records scale/ranking contracts and local/global key coordinates.

[Exact indexer selection](trace-exact-indexer-selection.md) supplies the two new
rules. Its [logical check](checks/check_exact_selection.json) covers 13,104
threshold cases and 72 radix cases, including overflow, underfull candidates,
crowded ties and signed-zero ranking. The older sorted_run_topk@1 is not silently
rebound to larger k or the opposite tie order.

[Hopper MLA roles](trace-mla-hopper-roles.md) binds shared decoded staging,
Q/KV channel counts, mixed load/PV responsibilities and no-KV control flow.
[Selector protocols](trace-selector-protocols.md) distinguishes score-block
publication, row barriers and invocation parity; the existing fixed epoch
encoding is not reused as a different barrier protocol.

[Sliding-window MLA](trace-sliding-window-mla.md) accounts for its six frozen
tactics, raw versus normalized partials and row-owned cluster combine. The
[logical witness](checks/check_cluster_row_combine.json) checks uneven row
ownership, mailbox coordinates and state weighting.

[CuTe/cuTile/cache trace](trace-mla-cute-cutile-cache.md) now binds flattened
rows, split capacity, transposed cuTile contractions and footer/inline scale
coordinates. Rubin mixed-cluster execution and SM120 cache-consumer mappings
remain explicit gaps.

[FP8 correction/cache writes](trace-fp8-correction-and-cache-writes.md) binds
serial row-state inheritance, lag-range limits, zero-row invalid gathers and
complete-record election. Its [logical check](checks/check_record_election.json)
covers duplicate maps, claim order and stale owner state.

[Rubin/mixed-cache consumers](trace-rubin-fallback-and-mixed-cache.md) binds
fallback coordinates, conditional raw-ready phases and decode/regroup/requantize
chains. The [logical check](checks/check_mixed_cache_channels.json) distinguishes
a wrong global phase and a lossless-reformat interpretation.

[Sparse support/training](trace-sparse-support-and-training.md) adds stable
compaction and binds checkpoint/recompute/scatter reuse for DSA and packed GQA.
Its [logical check](checks/check_support_compaction.json) preserves original
slot cuts, duplicate support mass, segment offsets and gradient multiplicities.

[Union masks/quantized relatives](trace-group-union-and-quantized-variants.md)
adds the shared-set candidate, deterministic score-contract dispositions and
cuDNN/QSA boundaries. Its [logical check](checks/check_group_union_masks.json)
covers page-order identity, row masks, staging reuse and set/occurrence semantics.

[Key centering](trace-key-centering.md) binds the dense quantized recipe and
separates common center, local scale, common probability gain and original LSE
frames. Its [logical check](checks/check_key_centering.json) covers row shifts,
sinks, per-block-center misuse and a bounded INT8 rounding witness.

[QSA scale/gate/walk](trace-qsa-scale-gate-and-tile-walk.md) binds quantized
consumer scales, preserves the BF16-before-gate checkpoint and adds the
contiguous walk candidate. Its [logical check](checks/check_contiguous_tile_walk.json)
covers1620 partition cases and a gate-rounding counterexample.

[Device attention/query centering](trace-device-attention-and-query-centering.md)
extracts mandatory query correction and binds balanced GQA online-prefix reuse.
Its [logical check](checks/check_query_center_correction.json) covers missing
correction, block tails, scale placement and FP16 correction overflow.

[SM110/MiniMax](trace-sm110-and-minimax-probabilities.md) binds tree masks, separate K/V page maps and paired output ownership, and extracts local-anchor low-bit P encoding. Its [logical check](checks/check_probability_anchor.json) distinguishes scale-cast order, original mass and scale underflow.

[VC compensation](trace-vc-value-compensation.md) adds grouped value means and selected residual replay. The [logical check](checks/check_value_compensation.json) covers reference updates, deferred tails, original/repair mass roles and residual generations.

[Runtime splits/fragments](trace-runtime-splits-and-fragments.md) adds fixed-capacity runtime partitions and semantic-block/page refinement. Its [logical check](checks/check_runtime_capacity_and_fragments.json) covers aligned ranges, inactive scratch, nonadjacent pages, current-block policy and sentinel decoding.

[B12x stream compression](trace-b12x-stream-compression.md) adds tagged pair carry and read-before-commit ordering. Its [logical check](checks/check_stream_pair_state.json) covers2040 chunkings, stale tags, early commit and pooled BF16 rounding.

[B12x dual-P/selection](trace-b12x-residual-p-and-selection.md) extracts separate signed component contractions and binds stable tie-column emission. Its [logical check](checks/check_residual_code_contraction.json) covers signed low codes, mass conventions, underflow and contraction order.

[GEMM scope/cluster ownership](trace-gemm-scope-and-cluster-ownership.md) adds a [design inventory](gemm-inventory.md), ragged scale/segment bindings and atom-owned multicast. Its [logical check](checks/check_atom_multicast.json) covers46 source-shape issuer cases, mask separation and fallback.

[Grouped promotion/fused output](trace-grouped-promotion-and-fused-output.md) adds metadata windows and binds register-owned epilogues plus complete-group quantization checkpoints. Its [logical check](checks/check_metadata_windows_and_fused_casts.json) distinguishes per-step scale identity, premature reuse, ring generations and BF16 cast stages.

[SVDQuant/W4A16](trace-svdquant-and-w4a16.md) adds low-rank scale placement and binds original-down versus quantized-residual channels. Its [logical check](checks/check_low_rank_scale_placement.json) covers fold error, stale factors, alpha0, bias/cast order and1905 finite decoder cases.

[Routed GEMM/ordered wgrad](trace-routed-gemm-and-ordered-wgrad.md) extracts fixed-order FP32 workspace merging and binds route/scale identity. Its [logical check](checks/check_ordered_workspace_and_routes.json) covers88 partitions, completion permutations, scattered live slots and the MX floor/ceiling distinction.

[MX floor-output quantization](trace-mx-floor-output-quantization.md) extracts the distinct scale recipe. Its [logical check](checks/check_floor_anchor_quantization.json) covers2490 normal cases, clipped code differences, complete-group maxima and reciprocal underflow exclusions.

[Remaining GEMM representatives](trace-gemm-remaining-designs.md) bind router and alpha/beta reuse and record masked mixed-format, TMEM and two-channel boundaries.

[Standalone quantization/norm](trace-standalone-quantization-and-norm.md) adds replicated cluster partials and binds ceiling versus floor scale, metadata coordinates and residual checkpoints. Its [logical check](checks/check_replicated_partials_and_norm.json) covers20 cluster shapes,1360 tuple deliveries,512 labeled scale coordinates and cancellation/zero-byte witnesses.

[Quantizer recipes/segmented masks](trace-quantization-recipes-and-segmented-masks.md) adds byte-aligned boolean segments and binds existing NVFP4 candidate-error selection,8x4 metadata and gate ordering. Its [logical check](checks/check_segmented_bit_packing.json) covers2048 bit-pattern, segment and tail cases.

[Rotary/norm/cache](trace-rotary-norm-and-cache.md) adds pair supply and binds across-head normalization, BF16/FP8 recipe differences and distinct position/page identities. Its [logical check](checks/check_paired_transform_supply.json) covers248 pair/suffix cases,448 exchange coordinates and ordering/checkpoint counterexamples.

[Adaptive sampling](trace-adaptive-sampling.md) adds support-preserving rejection. Its [exact-rational check](checks/check_adaptive_support_rejection.json) covers5172 laws and tie, stale-mass and renormalized-certificate counterexamples.

[Speculative correction](trace-speculative-chain-correction.md) adds draft/target coupling. Its [exact-rational check](checks/check_draft_target_residual.json) covers387 distribution pairs and a conditional chain; source finite-endpoint/zero-residual behavior remains a named boundary. [Gumbel/composition](trace-gumbel-and-sampler-composition.md) records counter grouping, compact-support tie differences and same-uniform ordering.

[Weighted radix/exact selection](trace-weighted-radix-and-selection.md) extracts the mass boundary and binds fixed/varlen selector reuse. Its [logical check](checks/check_mass_rank_radix.json) covers13440 sorted-reference cases and32 low-mantissa cases; native sum codec and tier maps stay explicit.

[Shared-exponent sum/logit pipeline](trace-shared-exponent-and-logit-pipeline.md) adds the integer histogram codec, plus softmax and typed-fusion reuse. Its [rational-oracle check](checks/check_shared_exponent_sum.json) covers750 groups and45 large-count cases; reproducibility is distinguished from nearest-even rounding.

[Fixed-point sparse sampling](trace-fixed-point-sparse-sampling.md) adds row-relative masses. Its [exact-unit check](checks/check_row_relative_mass.json) covers336 cases and separates integer exactness from FP64 cutoff and original-numerator output policies.

[GVR/granularity](trace-gvr-and-cluster-granularity.md) adds cooperative/independent cluster packing and records GVR2 rung, count, staging and key-space escapes. Its [mapping check](checks/check_cluster_granularity_switch.json) covers3048 cases.

Next: recurrent, MoE, model-block and distributed families. Remaining native roles/contracts in the [sampling inventory](sampling-inventory.md) stay in final reconciliation. Remaining [rotary/cache differences](rotary-cache-inventory.md) stay in final reconciliation. Remaining [quantization/norm variants](quantization-norm-inventory.md) stay in reconciliation. Attention/training
reconciliation and native ExpCast/orthogonal/queue gaps remain visible.
The starting pool includes the fifteen FlashAttention proposals; reuse actual
fields and boundaries rather than mechanism names alone.

[Deferred recurrent state](trace-deferred-recurrent-state.md) records frozen verification, snapshot/tape retention, accepted-prefix identity and one-use state publication. The exact check covers80 prefixes and440 tracking cases.

[Chunked delta feedback](trace-chunked-delta-feedback.md) binds residual-first Hopper prefill and coefficient-first frozen MTP output. Exact checks cover96 recurrence/chunk cases,48 padded extents and48 two8-row weighted solves. Recursive full inverse versus weighted solve remains an explicit existing-rule extension.

[Prepared recurrent factors](trace-prepared-recurrent-factors.md) binds state-independent W/P preparation, whole-workspace publication and the serial64-value-row state chain. Checks cover32 recurrence comparisons and labeled decay/W/pairwise coordinates, including the obsolete-header map counterexample.

[GDP substeps](trace-gdp-substeps.md) binds direct n2 decode and SM90 expanded64 prefill. Checks compare56 recurrences,1960 accepted prefixes and source readout/causal coordinates across within-token chunk boundaries.

[Bounded history warm-up](trace-bounded-history-warmup.md) binds the merged GDP split and exposes gate-only applicability gaps. Exact checks cover180 state/360 readout bounds,255 active-cut partitions and large-state/amplifying-key counterexamples.

[Low-rank state journal](trace-low-rank-state-journal.md) binds virtual projections, accepted-history fold,32-slot reader-safe append and once-per-step shared cursors. Checks cover476 algebra cases and all legal cursor/threshold combinations.

[CP affine fixup/ragged storage](trace-cp-affine-and-ragged-slots.md) extends the retained matrix-affine prefix rule and extracts conservative slot addressing. Checks cover32 final states,88 piece entries and10878 ragged partitions; native summary/fixup precision remains explicit.

[SSD logical chunks](trace-ssd-logical-chunks.md) extends the retained complete-chunk rule with clipped packed domains and concrete inter-chunk state propagation. Checks compare162 sequences and shared-block/frame ownership; native dt/IO rounding remains explicit.

[State rounding/random coordinates](trace-state-rounding-and-random-coordinates.md) extends the pinned BF16 carry rule to the source FP16 recipe and binds UMMA conversion fields/epochs. Exact checks expose subnormal prepared-mean loss and verify load/head/request coordinate regrouping.

[Paged recurrent boundaries](trace-paged-recurrent-boundaries.md) binds split prefix/checkpoint/final rows and shared metadata generation. Checks cover9652 prefill cases,14080 absolute checkpoints and46 decode boundaries, including unaligned prefixes and private tails.

[Locality state shards](trace-locality-state-shards.md) binds physical-domain allocations, green-context launch domains and complete fork/join. Checks cover98304 labeled state coordinates and256 FP32 transpose-reduce lane results; target locality/lifetime remains explicit.

[Boundary-seeded tape replay](trace-boundary-seeded-tape-replay.md) binds full, entry-boundary and no-artifact retention. Exact checks compare51 replay cases and440 polynomial gradient entries; reverse adjoint communication remains required in every mode.

[Reverse affine boundary transport](trace-reverse-affine-boundary-transport.md) binds head-owner and bounded neighbor paths, saved/resent map orientation and exclusive future seeds. Checks cover48 owner chains,24 short-hop chains and96 exact seed derivatives.

[Shared normalization gradient merge](trace-shared-normalization-gradient-merge.md) binds grouped TMEM versus materialized partials, fixed group sum and raw/saved-r pullback before one final cast. Checks cover48 exact Jacobian cases and32 finite-stage cases.

[Checkpoint recompute liveness](trace-checkpoint-recompute-liveness.md) separates output deadness, task-bound tape handoff and peak-lifetime policy. Logical checks cover384 replay schedules and illustrative lifetime/eviction cases; native torch frame integration remains explicit.

[Document stencil halos](trace-document-stencil-halo-pullback.md) adds absolute-token history/transpose routing and channel-packed exchange. Checks cover510 fragmented stencils and11294 transpose edges. [Activated gate suffix pullback](trace-gdn-gate-prefix-pullback.md) records existing affine/ownership rule bindings with195 exact length/seed cases.

[Partial-grid join placement](trace-partial-grid-join-placement.md) binds cooperative versus following-kernel completion, actual grid geometry and live partial rows. Checks cover292 layouts and19080 exact token partitions, with a cross-partition rounding counterexample.

[Self-resetting partition queue](trace-self-resetting-partition-queue.md) binds the zero-history forward ticket/retirement protocol. Checks explore29448 abstract states and distinguish counter reuse from output completion.

Preserved checkpoint draft: [rematerialized_activation_gradient_recipe@1](delta/rematerialized_activation_gradient_recipe@1/template.yaml) and [outline](delta/rematerialized_activation_gradient_recipe@1/proof.yaml). Schema-validated only; source trace and focused stage check unfinished. It is not part of the56 indexed extraction menu entries.
