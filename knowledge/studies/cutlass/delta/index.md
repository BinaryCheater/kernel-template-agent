# CUTLASS pool delta — active

Twenty-three provisional Template/ProofOutline pairs across twenty-one mechanism names are proposed against the original pool
and its recorded [reconciled view](../inputs/pool-reconciled-snapshot.json).

| Choice and pair | Concrete source binding | Implementation boundary |
|---|---|---|
| [cache_prefetch@1](cache_prefetch@1/template.yaml), [outline](cache_prefetch@1/proof.yaml) | Dense A/B or block-scaled A/B/SFA/SFB hint streams; explicit distance or default derived from AB stages. T5/D4/S2 is the distinguishing index witness. | Immutable valid sources and value-neutral hint backend. No shared slot is allocated; different backend/tail addressing needs its own guarded binding. |
| [cross_buffer_operand_window@1](cross_buffer_operand_window@1/template.yaml), [outline](cross_buffer_operand_window@1/proof.yaml) | SM103 W96,C256, crossing windows2/5 and last chunk uses2/5/7. | W<=C; source C256 is hard-coded. A different chunk extent requires descriptor/layout/offset/retirement changes, not a source knob override. |
| [staged_async_output_store@1](staged_async_output_store@1/template.yaml), [outline](staged_async_output_store@1/proof.yaml) | TMA epilogues derive stages=num_c_stage, commit one group per subtile and wait S−1/0 source-read groups. | Shared generations survive source reads; destination observation requires write completion. Direct stores and a new clipping/completion path are separate bindings/variants. |
| [launch_cancellation_work_reuse@1](launch_cancellation_work_reuse@1/template.yaml), [outline](launch_cancellation_work_reuse@1/proof.yaml) | Dynamic dense GEMM, one CLC response slot,224 readers for one CTA or416 for two. Reader count is derived from actual roles. | Exclusive unstarted-group cancellation and response lifetime; coordinates only after success and no request after an observed failure. Pinned unconditional query IR needs guarded lowering or correction. |
| [scatter_reduce_output_shards@1](scatter_reduce_output_shards@1/template.yaml), [outline](scatter_reduce_output_shards@1/proof.yaml) | Cluster split-K,128 epilogue threads, equal E/S fragments, sender-specific DSMEM slots and FP32 owner-first addition. | Defined FP32 partials, consistent injective fragment maps, accepted numeric order and byte/acknowledgement generations. Empty K partials need identity initialization or exclusion; other accumulator types need another variant. |
| [online_group_work_decode@1](online_group_work_decode@1/template.yaml), [outline](online_group_work_decode@1/proof.yaml) | Grouped SM100/SM103, counts[1,2,2], two workers,32-lane windows; helper cache_shapes false/true. | Immutable counts/shapes and nonnegative monotone queries, Int32 prefix, L1 M/N counts. Spatial decode remains external; new ordering/lane/batch mapping needs a variant. |
| [rebind_transfer_descriptors@1](rebind_transfer_descriptors@1/template.yaml), [outline](rebind_transfer_descriptors@1/proof.yaml) | Grouped SMEM-shadow path, input publisher tuple4 or2 and C tuple1,128-byte private slots. | Fixed-compatible atoms, immutable keys, exclusive slots, old-reader retirement and release/acquire. Exact transfer-engine retirement is not discharged by the scalar model; direct-global mutation is separate. |
| [output_prefix_work_filter@1](output_prefix_work_filter@1/template.yaml), [outline](output_prefix_work_filter@1/proof.yaml) | Masked blockwise grouped GEMM,128-row cooperative units, optional unfiltered initial work. | Whole-unit prefix intersection, safe unobserved padding, peer/role agreement and validity before metadata. Source terminal lookup needs a guard; partial tiles still store padding. |
| [implicit_patch_contraction@1](implicit_patch_contraction@1/template.yaml), [outline](implicit_patch_contraction@1/proof.yaml) | Dense SM100 convolution, Q/P/Z/batch output order and C/S/R/T reduction order, patch_storage=on_load. | Zero-padding cross-correlation; correct lower/leading padding, positive geometry, safe tails and supported descriptor fields. Full materialization is an algebraic alternative only; grouped channels/other padding need a variant. |
| [byte_arrival_completion_join@1](byte_arrival_completion_join@1/template.yaml), [outline](byte_arrival_completion_join@1/proof.yaml) | Block-scaled convolution full barriers join A/B/SFB TMA bytes and128/256 completed-copy arrivals, with initial counts129/257. | One reserved atomic-registration owner; exact signal incidence and local/remote target generations. Source two-group delay needs adequate slot capacity and short-tail drain. Device atomicity/visibility remains open. |
| [groupwise_low_float_quantization@1](groupwise_low_float_quantization@1/template.yaml), [outline](groupwise_low_float_quantization@1/proof.yaml) | Block-scaled convolution SFD output, NVFP4 group16/E4M3 scale or MXFP8 group32/E8M0 scale, normalization1. | Decode the stored scale before code generation; finite-code/range, zero, reciprocal and identity-after-quantization premises. Native group/lane/conversion behavior and numeric extremes remain open. |
| [rebind_transfer_descriptors@2](rebind_transfer_descriptors@2/template.yaml), [outline](rebind_transfer_descriptors@2/proof.yaml), [migration](rebind_transfer_descriptors@2/migration.md) | Dense grouped AB tuple2/C tuple1; update_site=shared_shadow or direct_global. | Version1 is unchanged and projects to shared_shadow. Direct mode retires readers before first field store, then release/acquire; zero shadow bytes. Native retirement/init/proxy obligations remain. |
| [integer_operand_widening@1](integer_operand_widening@1/template.yaml), [outline](integer_operand_widening@1/proof.yaml) | Batched mixed-input signed4/signed8/unsigned8→FP16/BF16; typed or paired Int4/BF16 eight-value shuffle. | Integer signedness and physical/logical order agree; paired path requires full eight-value chunks and available converter. Scale arithmetic, tails, wide-slot placement and native lane/converter proof are separate. |

| [segmented_axis_work_decode@1](segmented_axis_work_decode@1/template.yaml), [outline](segmented_axis_work_decode@1/proof.yaml) | Group lengths8/0/40/16, CTA_N16, cooperative N peers1/2/4; scalar prefix cursor. | Work counts round per group; retain signed peer tails and collective identity. Int32 bounds, safe inactive loads and group-end masks are required. |
| [ordered_slot_wait_coalescing@1](ordered_slot_wait_coalescing@1/template.yaml), [outline](ordered_slot_wait_coalescing@1/proof.yaml) | Acc-scale transform waits on second wide slot, publishes both; odd tail waits on itself. | Batch2 fits the ring; actual credit completion must be a retired prefix with exact generation. Replaces empty-wait cost only; native ordering remains open. |

| [online_group_work_decode@2](online_group_work_decode@2/template.yaml), [outline](online_group_work_decode@2/proof.yaml), [migration](online_group_work_decode@2/migration.md) | MoE scalar expert/start/end cursor; backward K0 still owns output. | Preserve version1 at search=warp_prefix_32; scalar_forward forbids shape-window caching. Physical decode/short-axis policy remains family glue; source E0 initialization is excluded. |
| [precompute_transfer_descriptors@1](precompute_transfer_descriptors@1/template.yaml), [outline](precompute_transfer_descriptors@1/proof.yaml) | Separate setup creates forward C1 or backward AB2 descriptors per expert,128-byte slots. | Immutable shared key/slot table scales with problems, rather than private workers. Setup/proxy visibility, invocation retirement and applying base offsets once are required. |

| [segmented_aligned_storage@1](segmented_aligned_storage@1/template.yaml), [outline](segmented_aligned_storage@1/proof.yaml) | Scaled MoE: per-expert scale rows align128 or scale columns align4; real data keeps compact token origins. | Per-segment storage prefix and local layout agreement; initialized padding must be unobserved or neutral. Encoded E8M0 fill and live reduction tails remain native obligations. |

| [replicated_partial_reduce@1](replicated_partial_reduce@1/template.yaml), [outline](replicated_partial_reduce@1/proof.yaml) | RMSNorm C2/W4/R1, FP32 table replicated at each CTA;32 incoming bytes per destination. Local C1 is the shared-barrier variant. | Uniform whole-warp partials, injective sender slots, exact completion and collective allocation lifetime. Source phase0 is one-shot; subwarp ownership and repeated generations need another binding. |

| [operand_bitfield_placement@1](operand_bitfield_placement@1/template.yaml), [outline](operand_bitfield_placement@1/proof.yaml) | SM120 mixed FP4/FP8 load produces low E2M1 payload bits; register shift2 moves them to the MMA field. | Preserve code/coordinates and prepare once after each unpack. Native byte/lane map and fragment lifetime remain open; same-type FP4 and the FP8 side do not shift. |

| [interleaved_consumer_streams@1](interleaved_consumer_streams@1/template.yaml), [outline](interleaved_consumer_streams@1/proof.yaml) | Two SM120 consumer groups alternate local persistent items with fixed positive K stages. | Advance physical index/phase across skipped items without release; active-group arrivals only. External order protocol/output lifetime remain required, including four backed order slots despite the source two-slot declaration. |

[Partial composition](chain.yaml) binds cancelled-launch ownership and peer-owned
shards to a source-shaped M128/N128/K256 case with two nonempty K ranges. Input
staging, partition/MMA and fragment mapping are external edges. The original source
CLC wrapper is not admitted merely because this logical Chain validates.

[Reuse audit](../reuse-audit.md) supplies structural retrieval cues and the hardest
old-field/Realisation check. async_transfer@1 fixes global→shared transactions;
multi_buffer_rotation@2 owns destination-slot lookahead and stops at S8;
cache_eviction_hint@1 selects priority; persistent_tile_scheduler@1 has no hardware
cancellation fetch; warp_specialization@1's four-warp consumer-group unit does not
match the one-warp MMA roles. Missing decisions remain a delta or an explicit gap.
No old identity, variable domain or cost model is silently broadened.

[Retained checks](../checks/README.md) validate the twenty-three pairs, executable guards,
bounded source mappings and [reused field bindings](../inputs/reuse-bindings.json).
No Realisation, ProofSummary or Store admission is proposed. End-to-end agent replay
is deferred; no catalog-wide rewrite is required. The whole CUTLASS source inventory
remains active after the block-scaled, dense, grouped and blockwise routes.

Blockwise scaling reuses block_scaled_partial_accumulation@1 with K128, M1/N128
scale ownership. Its old SM90 Realisation is not an SM100 implementation. Contiguous
segmented-row mapping and the ten-stage scale ring remain named gaps; no old domain
is widened. See [the trace](../trace-blockwise-gemm.md).

The convolution source also admits cluster16 CLC cases with5152 readers, outside
launch_cancellation_work_reuse@1's participant cap4096. That is an explicit old-domain
gap, not an accepted binding or silent widening. The new low-float quantizer does
not reuse affine integer quantization or lossless power-of-two metadata encoding.

[MoE partial composition](moe-chain.yaml) connects the immutable AB table and scalar
expert decode to a logical output ring, with external descriptor-relative staging,
MMA and explicit K0 zeros. The source scheduler sentinel/tail state is a recorded
protocol gap; this chain does not claim correct native progress. See the
[dense MoE trace](../trace-moe-dense.md) for evidence and the remaining scaled slice.

[Scaled metadata partial composition](moe-scaled-metadata-chain.yaml) connects SFA
regions to an immutable AB/SF descriptor table; parallel SFB preparation and native
mainloop remain external. The same setup kernel computes padded origins and the
descriptors, so no second launch is charged. See the
[metadata trace](../trace-moe-scaled-metadata.md).

The [scaled MoE device trace](../trace-moe-scaled-device.md) adds no candidate.
physical_storage@1 supplies only its per-region lifetime inequality; the source's
phase-selected compressed layout, alternating traversal and release threshold are
explicit glue. operator_fusion@1 scale_and_cast supplies the local epilogue, not
permission to commute globals through the dot. Source K768 geometry and numerical
factoring counterexamples constrain later implementation bindings.

[Row-reduction composition](row-reduction-chain.yaml) joins warp partials,
replicated row scalars, an admitted reciprocal-root approximation and weighted
output fusion. Source vector coverage and neutral padding are external guards;
[the trace](../trace-row-reduction.md) retains the missing-column and coarse-vector
counterexamples. This is a partial proposed composition with native obligations.

[GeForce schedule composition](geforce-schedule-chain.yaml) connects striped stage
ownership to fresh FP4 register placement. It is conditional on native/layout and
order-storage backing; the [schedule trace](../trace-geforce-schedules.md) records
those obligations plus K0, cooperative output drain and reverse mixed alignment.

[GeForce layout/host closure](../trace-geforce-layouts.md) adds no candidate.
Global and shared scale maps are separately checked; native coordinate transport
remains a binding obligation. The scalar order-storage model places four slots
inside alignment padding for stages1..16; it does not establish compiled offsets
or native progress. Both GeForce blockscaled kernel entries are source-extracted.

[online_rescaled_reduction@2](online_rescaled_reduction@2/template.yaml) and its
[outline](online_rescaled_reduction@2/proof.yaml) add finite lagged references and
a common weight gain with sink/LSE compensation. The
[migration](online_rescaled_reduction@2/migration.md) preserves version1 and records
the narrower positive-mass projection, changed invariant and reopened cost/backend
obligations. [FMHA frame trace](../trace-fmha-frames.md) retains the source sink
counterexample and rejects nonidentity tile pruning.

[bounded_normalized_weight_pruning@1](bounded_normalized_weight_pruning@1/template.yaml)
and its [outline](bounded_normalized_weight_pruning@1/proof.yaml) certify absolute
normalized-output and optional natural-LSE error from retained/dropped mass and
value ranges. [FMHA pruning trace](../trace-fmha-pruning.md) separates source
threshold proposals, absent cumulative certificates and host/device reference
mismatch; native skip retirement remains conditional.
