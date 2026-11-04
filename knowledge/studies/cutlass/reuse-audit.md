# Retrieval cues, implementation boundaries and hardest reuse

This audit applies the updated extraction skill to the current CUTLASS candidates
and reused decisions. It adds source bindings and bounded evidence; end-to-end agent
replay is deferred. The [pinned pool](inputs/pool-snapshot.json) remains the identity
baseline. No catalog-wide rewrite or independent admission is part of this audit.

## Current candidates

| Structural retrieval cue | Invariant and supported source boundary | Consequential misuse and evidence |
|---|---|---|
| Future global operand tiles are addressable while actual shared stages are occupied | [cache_prefetch@1](delta/cache_prefetch@1/template.yaml) preserves demand-copy indices/waits. Dense A/B and block-scaled A/B/SFA/SFB expose the distance knob; source default derives it from stages. | D may exceed shared stages because hints have no shared destination. Invalid addresses/mutable lifetimes violate scope. checks/prefetch_schedule.py covers the T5/D4/S2 scheduling witness. A different cache-hint backend needs its own guarded implementation. |
| Consumer instruction windows do not divide producer chunks | [cross_buffer_operand_window@1](delta/cross_buffer_operand_window@1/template.yaml) preserves logical coordinates through current/next chunks. Pinned SM103 is W96,C256; chunk size is hard-coded. | Changing C is an implementation variant: update TMA chunk layout, unrolled offsets, descriptors and release frontiers. W>C is outside this version. checks/circular_windows.py extracts all eight source pairs and last uses2/5/7. |
| Register outputs must feed a shared-source async output engine | [staged_async_output_store@1](delta/staged_async_output_store@1/template.yaml) keeps each shared generation live through source reads and observes outputs only after destination writes. Source stage count is derived from layout/budget and one group captures one subtile. | A read-only wait cannot authorize output observation. A new clipping path or copy-group scheme needs its own binding/variant. checks/sm103_layout_output.py preserves a weaker-wait overwrite witness; exact device visibility remains a target obligation. |
| A full grid names work but active cooperative groups finish at different times | [launch_cancellation_work_reuse@1](delta/launch_cancellation_work_reuse@1/template.yaml) transfers one cancelled launch identity exclusively. Single response slot; reader count224 or416 in one-/two-CTA examples. | Check success before querying coordinates and stop after observed failure. Pinned clc.py builds three coordinate-query IR operations unconditionally: guarded lowering or a corrected implementation is required before reuse. checks/dense_scheduling.py retains this gap and bounded ownership states. |
| K peers hold matching partial output fragments and can divide final ownership | [scatter_reduce_output_shards@1](delta/scatter_reduce_output_shards@1/template.yaml) combines one defined FP32 contribution per peer and stores each fragment position once. Source uses128 threads and owner-first order. | Empty K partitions need identity values or exclusion; FP16/Int32 accumulator behavior needs another numerical variant. checks/dense_scheduling.py verifies mailbox incidence and retains the T5/S4 empty-partition witness. |
| Increasing persistent indices cross variable-size problems | [online_group_work_decode@1](delta/online_group_work_decode@1/template.yaml) preserves immutable prefix inversion with a retained monotone cursor;32-lane windows and optional two-window shapes. | Backwards queries cannot reuse a forward cursor. checks/grouped_work_descriptors.py retains the wrong negative local offset; new batch/order/lane mapping needs a variant. |
| Fixed transfer atoms survive changes in problem metadata | [rebind_transfer_descriptors@1](delta/rebind_transfer_descriptors@1/template.yaml) publishes private complete generations after old readers retire, then readers acquire. Input tuples4/2 and C tuple1. | K0 inputs may skip but C cannot. The stale-C witness distinguishes this; exact engine retirement/proxy semantics remain a backend obligation. Direct-global helper mode is outside this binding. |
| Allocated output capacity exceeds the visible per-group row prefix | [output_prefix_work_filter@1](delta/output_prefix_work_filter@1/template.yaml) keeps every visible output once, using whole cooperating-unit starts; optionally bypass initial units. | Partial units still write padding, which must be safe and unobserved. Source terminal metadata lookup needs guarding. checks/blockwise_contracts.py retains per-CTA disagreement and terminal index3 for three groups. |
| Sliding spatial windows should feed a matrix-contraction path | [implicit_patch_contraction@1](delta/implicit_patch_contraction@1/template.yaml) preserves channel-fast tap/output bijections and zero-padding identity; source on_load uses im2col TMA. | Leading padding is lower padding in the pinned helper. checks/conv_contracts.py distinguishes swapped padding/taps and requires positive geometry separately from source field-width bounds. Materialized patches are an algebraic alternative, not an executed source backend. |
| One stage joins transaction bytes and independently completed copy producers | [byte_arrival_completion_join@1](delta/byte_arrival_completion_join@1/template.yaml) waits on both counters, reserving one arrival for atomic byte registration. Source uses128/256 SFA copy arrivals plus one owner. | Early copy arrivals must not complete an unregistered TMA stage. Two-group delayed arrivals need capacity at least min(T,3) in the scalar model and explicit short-tail drain; target generation/visibility remains a backend obligation. |
| Requested output groups need low-float values and shared encoded scales | [groupwise_low_float_quantization@1](delta/groupwise_low_float_quantization@1/template.yaml) uses the decoded stored scale to generate each code; NVFP4 group16 or MXFP8 group32, normalization1. | The rational witness stored scale5/4 gives code3/2, while using raw37/30 gives2. Native group/rounding/reciprocal and numeric extremes remain explicit obligations; arbitrary accepted custom activation cannot satisfy identity by registry membership alone. |

| Stored integer operands must feed a wider floating MMA | [integer_operand_widening@1](delta/integer_operand_widening@1/template.yaml) preserves two's-complement/unsigned integer values and every coordinate; typed or paired eight-value Int4/BF16 order. | Raw0 is0, not Marlin offset-binary-8.1056 code/type pairs and128 distinct packets check the logical contract. Native converter availability/lane mapping, scale multiplication and wide storage are separate premises. |

A source binding inside these boundaries adds an instance. New tails, descriptor
mappings, copy completion semantics, accumulator types or weakened premises require
an explicit variant/revision. A retrieval cue alone is never an executable scope
extension. Template scope/constraints remain the applicability guards.

## Hardest decisions previously marked reused or considered for reuse

| Decision | Exact old field or implementation | Verdict after the hardest-reuse check |
|---|---|---|
| Actual AB TMA input readiness | [async_transfer@1](../../seed/templates/async_transfer.yaml): mechanism=controller,issue_point=after_slot_released,complete_point=barrier_wait,granularity_bytes=one AB operand tile; obligations readiness/no_overwrite_in_flight | Same global→shared mapping at the abstract interface. PipelineTmaUmma's actual arrivals/completion are source-bound implementation obligations. This old record cannot supply shared→global output-store lifetime; that decision is captured by staged_async_output_store@1. |
| A producer reuses shared generation slots P ahead | [multi_buffer_rotation@2](../../seed/templates/multi_buffer_rotation.yaml): S,P,beta,late,private and c_lifetime; bounded S4/P2/beta0/late0/private0 matches the manual S−2 schedule | The pinned version is2; prior current-study prose naming1 was corrected. Its declared S range ends at8. The S9 scalar replay is outside that old scope, and derived source stages>8 need another supported instance/version. Compiler scheduling and UMMA completion do not follow from its slot inequality alone. |
| Early release while another accumulator region is still being read | [physical_storage@1](../../seed/templates/physical_storage.yaml): pre_lifetime p.last_read+p.delta<=p.next_write | Apply the inequality per intersecting storage region. It does not choose the source's reverse/forward subtile order or derive the region partition. Those remain explicit source glue and a traversal-choice gap; there is no whole-buffer lifetime reuse claim. |
| Rectangular full/tail output tiling | [output_tiling@1](../../seed/references/cutlass_ampere_tf32_gemm/templates/output_tiling.yaml): tile_m,tile_n,executor,index_map plus coverage_exact_once | Matches rectangular intermediate tiles and their domain. Full-tile-only direct output, descriptor-clipped output and vector-predicated output have distinct implementation guards. Final split-K output ownership comes from the new shard rule; tiling alone does not supply it. |
| Fixed TMA/MMA/epilogue warp roles | [warp_specialization@1](../../seed/references/flash_attention_3_fwd/templates/warp_specialization.yaml): consumer_groups counts four-warp groups and its resource formula assumes that unit | The source MMA role is one warp with four separate epilogue warps, so its role/resource mapping cannot be recovered by setting consumer_groups1. Retain a role-granularity gap rather than claiming this name alone supplies the six-/seven-warp design. |
| Work assignment through hardware cancellation | [persistent_tile_scheduler@1](../../seed/references/flash_attention_3_fwd/templates/persistent_tile_scheduler.yaml): fetch domain static_stride/atomic_counter; grid denotes physical CTAs | It supplies a static single-CTA stride instance, not CLC cancellation or cooperative cluster launch costs. CLC ownership is captured by the new rule; group-lifted static scheduling/swizzle remains explicit glue/gap. |
| Online inversion from a linear work index to a current problem | runtime_work_descriptors@1: planner_site selects host/device_kernel and builds descriptor arrays | It does not supply a stateful inverse-prefix cursor inside the persistent kernel. online_group_work_decode@1 captures that missing decision rather than treating device_kernel as online scheduling. |
| Rebind TMA addresses/extents/strides while preserving fixed atoms | async_transfer@1 describes transfer issue/completion; cluster_multicast@1 describes receiver ownership | Neither old rule owns mutable descriptor generation or publication. rebind_transfer_descriptors@1 owns those choices, with engine-specific retirement still open. |
| Grouped SM103 active-K operand windows and K0 output | cross_buffer_operand_window@1: W96,C256 current/next addressing; staged_async_output_store@1: per-output source ring | All eight AST pairs and releases2/5/7 match the existing window rule. Rebinding is a new independent rule; explicit K0 zero registers establish defined output before output-ring reuse. Accumulator generations skip K0 while C generations advance, so the stage clocks cannot be equated. |
| K128 raw partial reset and post-scale promotion | [block_scaled_partial_accumulation@1](../deepgemm/delta/block_scaled_partial_accumulation@1/template.yaml): partial_k128, gran_k_a/b128, gran_m1, gran_n128, reset_and_promote | Three blockwise entries preserve this mapping. Scope requires admitted numerical reassociation and initialized tails. The existing SM90 Realisation does not supply SM100 tcgen05 code; source shape/scale partition remains a new implementation binding with open device obligations. |
| Whole output-unit skipping rather than masked K contributions | masked_tile_pruning@1: mask_kind/peeled reduction steps and all-identity skipped contributions | It refines reduction tiles and cannot supply skipped output ownership/pipeline sequence. output_prefix_work_filter@1 introduces that object and makes padding observability explicit. |
| Contiguous group identity and ten-stage scale ring | output_tiling@1 index_map; multi_buffer_rotation@2 S<=8 | Per-row group lookup/128-alignment remains segmented mapping glue/gap, not supplied by a name match. Scale stages10 exceed the old rotation domain, so resource/lifetime reuse needs another version or instance. |
| Spatial patch coordinates without an index table | indexed_gather@1 object=index_map, granularity page/row, pre_valid indices_in_bounds; tensor_axis_permutation@1 permutes three existing axes | Neither introduces arithmetic spatial window replication and zero-fill into a new contraction axis. implicit_patch_contraction@1 owns that mapping; old transfer readiness can apply after its patch values/coordinates are established. |
| Optional dense-convolution bias and post-cast activation | operator_fusion@1 consumer_kind=elementwise, intermediate_site=registers, fused_ops=linear_combination | Supplies accumulator+bias fusion only. Its enum does not name arbitrary activation or cast placement; source ordering remains numeric glue/gap. The concrete old binding validates without broadening the enum. |
| Dense-convolution CLC and dual cluster choice | launch_cancellation_work_reuse@1 participants equals32*(1+cluster_size*6); cluster_multicast@1 receiver sharing | Same single-response ownership rule; old clc.py failed-response query gap remains. Runtime preferred/fallback branch selection is external source glue rather than a multicast parameter. |
| Join two independent full-stage readiness channels | per_slot_handshake@1 full_signal domain tma_tx_count/thread_arrive/cp_async_arrive selects one completion kind; arrival_count obligation | It has no field for a joint byte/arrival counter pair or the reserved registration arrival. byte_arrival_completion_join@1 owns that new choice; empty-slot/all-reader lifetime remains independently required. |
| Block-scaled convolution patch values and scale traversal | implicit_patch_contraction@1 patch_identity and coordinate_bijections; source implementation visits S/R/T before C_tile | Logical values/coordinates match; changed execution grouping and per-filter channel padding need this explicit implementation binding and numerical/tail premises. The retained recurrence checks K-side scale ownership, not SFB N-window offsets or target layouts. |
| Low-float output scale generation rather than integer range quantization | groupwise_affine_quantization@1: unsigned integer code_bits/zero_point and min/max-derived affine range | That old program cannot implement amax-based E2M1/E4M3 codes with E4M3/E8M0 scale metadata. groupwise_low_float_quantization@1 introduces the requested numerical policy without adding fake integer bindings. |
| Upward-round an arbitrary positive amax scale to E8M0 | power_of_two_scale_encoding@1 c_exact_domain/pre_exact require scales_positive_normal_pow2, exponents-126..127 | That old version preserves already-defined scale values and explicitly excludes arbitrary upward rounding and live code0. The new quantizer owns scale generation and its zero/minimum-scale policy; no old scope is broadened. |
| Additional convolution CLC response readers | launch_cancellation_work_reuse@1 participants domain1..4096 | One/two-CTA blockscaled instances352/672 or384/736 fit the domain. Source cluster16 needs5152 without residual and cannot bind the old domain; keep an extension gap. Guarded failed-response queries remain separately required. |
| SFB metadata N64/N192 windows | Source instance nested ((2,2),rest), strides((x,x),3x), odd-tile64-channel offset | Logical windows match the earlier blockscaled instance. cross_buffer_operand_window@1 W<=C does not select a N192/C128 metadata reshape, so this remains a retained physical-layout binding/abstraction gap rather than a false old-rule match. |
| Public shape caching for dense grouped GEMM | online_group_work_decode@1 variable cache_shapes=True, same prefix cursor and immutable metadata | Same mapping and premises as the earlier helper binding; new constructor option adds an instance. Native two-window cache allocation/lowering is still open. |
| Direct-global descriptor field updates | rebind_transfer_descriptors@1 program explicitly stages shared shadows | Version1 cannot bind a zero-shadow in-place update. Version2 adds update_site, preserves all old projected fields/obligations and requires retirement before first direct store. Both dense K branches participate in initialization; no missing-K0-partner source defect is claimed. |

| Mixed-input signed integer conversion | fused_dequantization@1 exact_conversion explicitly produces q-8; its group_size menu and per-element scale policy are fixed | method=cvt_instructions or zero_point=none cannot change that obligation. integer_operand_widening@1 supplies signed4/signed8/unsigned8 exact widening with separate scaling; no old numeric contract is silently changed. |
| Eight-value paired shuffle for A | offline_operand_prepack@1 fixes Marlin tile/interleave choices and requires a constant operand | The mixed-input host permutation[0,2,1,3,4,6,5,7] and potentially dynamic A are not supplied by that old enum/premise. The new conversion rule consumes matching storage order; producer cost/layout preparation remains separate. |

| Scalar segment-local N work and peer tails | online_group_work_decode@1 program fixes warp32 prefix windows and leaves spatial decode external | It cannot select separate-rounded scalar boundary traversal with physical origins and signed distances. segmented_axis_work_decode@1 owns that new mapping; terminal metadata validity and inactive-peer safety stay explicit. |
| Acc-scale mixed-input partial promotion | block_scaled_partial_accumulation@1 partial_k256, gran_k_a256, gran_k_b512, gran_m1, gran_n64; precondition numeric_contract_allows_partial_scaling | B is logical constant1 over K512/N64, not an allocated scale tensor. Same reset/promote algebra, conditional on accepted rounding changes; BF16 witness differs by1/128. Old SM90 Realisation does not implement the SM100 path. |
| Group-boundary hybrid output | staged_async_output_store@1 per-output source ring; boundary_predication@1 runtime_length/fill=none | Reuse only the selected full-TMA subsequence and predicated tail primitive. New family variant owns disjoint domains/coverage and advances the ring only for TMA subtiles; no blanket all-output clock reuse. |
| Skip first empty wait of a transform pair | per_slot_handshake@1 full/empty lifetime and arrival_count | It does not imply prefix-ordered asynchronous retirement or pair-level acquisition. ordered_slot_wait_coalescing@1 replaces only the empty-wait component, preserves individual publication and requires exact native prefix completion. |

| MoE scalar expert cursor | online_group_work_decode@1 program fixes32-lane prefix windows | Version2 adds search=scalar_forward and forbids its shape-window cache; all old bindings project with search=warp_prefix_32. Source E>=1 and per-scenario count construction are family guards/glue. |
| Shared immutable transfer descriptor table | rebind_transfer_descriptors@2 object=per_worker_descriptor_generation, storage scope=cta, program caches a mutable problem key | Setting update_site=direct_global cannot select shared per-expert immutable ownership or a separate setup launch. precompute_transfer_descriptors@1 owns those choices and publication/invocation lifetime. runtime_work_descriptors@1 instead emits per-work ownership arrays. |
| Per-expert shortest-axis traversal | block_rasterization@1 order is compile-bound; group tunes interleave and operand_reads uses a launch-map estimate | Runtime per-expert axis switching and cooperative decode are not supplied by an enum label. Exact source permutation is replayed, but the chooser and any L2 benefit/cost stay explicit gaps. |
| Backward MoE update of existing C | first_writer_atomic_combine@1 requires2..10 partials and a designated initializing overwrite | bulk_reduce_add is only its update engine, not permission to preserve an arbitrary existing C. MoE casts each partial to output dtype before the reduction store; retain a separate arithmetic variant/implementation gap. |

| Scaled MoE separate scale origins | segmented_axis_work_decode@1 preserves original physical segment prefixes; runtime_work_descriptors@1 generates output-work ownership | Neither creates padded payload storage with a distinct prefix. segmented_aligned_storage@1 owns that map, fixed consumer alignment and padding policy. Marlin offline_operand_prepack@1 also cannot select this32_4_4 per-expert scale layout. |
| Scaled immutable AB/SF tuple | precompute_transfer_descriptors@1 descriptors_per_problem domain1..5; injective128-byte key/slot storage and immutable metadata | Tuple4 is an instance. Original immutable ends/host scale layout are setup inputs; final padded origins are derived inside setup before dependent SF construction. AB can be constructed first. Conditional layout/proxy/zero-extent premises remain; the rule does not establish prefix or neutral padding by name. |

| Scaled MoE compressed accumulator lifetime | physical_storage@1 pre_lifetime p.last_read+p.delta<=p.next_write, applied only to each intersecting region | Same inequality as the earlier overlap instance. The source phase/stride/reverse-order/release mapping remains explicit glue and native completion remains open. K768/N256/MXFP8 yields SF288, stride-32 and unreachable release9, despite accepted shape predicates; not an admitted old binding. |
| NVFP4 final global scale and cast | operator_fusion@1 consumer_kind=elementwise, intermediate_site=registers, fused_ops=scale_and_cast | Supplies the operation on an already-defined accumulator. It does not prove input-scale factoring through a reduction; FP32 counterexample survives BF16. A finite computed alpha is required for the source K0 zero path; presence of global A requires paired global B. |

| Full row scalar at every CTA | scatter_reduce_output_shards@1 divides output fragments into peer-owned final shards; smem_partial_combine@1 fixes serial/tree/CUB local implementation and cost | Neither supplies sender-indexed all-to-all scalar replication plus a repeated table fold by every reader warp. replicated_partial_reduce@1 owns that choice; peers1 is the one-shot local shared-table variant. |
| RMSNorm aligned vector witness | vectorized_access@1 vector_bytes16/thread_arrangement=strided_loop/threads128 and c_align contiguous_elems*elem_bytes%vector_bytes==0 | N32768/R1 matches the old access map; N32767 fails its guard. The source's coarse vector predicate does not silently extend the old domain. General multirow TV layout and missing scheduled columns remain explicit source glue/gaps. |
| Source warp butterfly and fast reciprocal root | cross_lane_reduction@1 lanes8/16/32,result_to=all_lanes,combine_op=add; approximate_math_rewrite@1 rewrite=rsqrt_approx/fused_fma=false | Wrapper actually uses xor shuffles within aligned groups. Old numeric associativity/commutativity and tolerance_allows_fast_math fields remain required; scalar incidence does not discharge native precision or approximation error. |
| RMSNorm remote completion has no independent thread-copy arrivals | byte_arrival_completion_join@1 copy_arrivals domain1..4096 | A source byte-only phase0 barrier cannot bind copy_arrivals0 or an invented extra arrival. The new replicated-table rule owns its exact per-destination incidence, while native registration/completion order and progress remain explicit premises. |

| Mixed SM120 FP4 load-to-MMA bit position | integer_operand_widening@1 input_format signed4/8/unsigned8 and compute_format FP16/BF16; retained bitstream_packing@1 program maps contiguous code bits | Neither supplies E2M1 payload relocation from byte bits0..3 to2..5. New operand_bitfield_placement@1 preserves all floating code bits and coordinate identity. Humming's reconciled body is compared only; no dependency or unavailable-old-body equivalence is inferred. |

| GeForce cooperative versus alternating role sequences | warp_specialization@1 producer_warps1/consumer_groups2/epilogue_role=consumers, same_sequence obligation | Counts fit cooperative eight-warp compute roles. Pingpong visits different alternating item sequences, so the same fields do not supply its stage jumps or active-group arrival count. interleaved_consumer_streams@1 owns that mapping, with order arbitration separate. |
| SMEM release before final register MMA | physical_storage@1 pre_lifetime last_read+delta<=next_write | Last shared read precedes release after register capture even while final arithmetic is pending. Register fragment lifetime is a distinct edge. Source cursor checks at S>8 are not evidence for extending multi_buffer_rotation@2. |
| Cooperative multi-stage output tail | staged_async_output_store@1 requires draining final source reads before shared storage reuse/free | No explicit source TMA-store tail is present. S1 acquisition drains each group; S>1 final retirement remains an implementation obligation. Pingpong drains each item's output before order handoff, conditional on valid storage/order backing. |

The proposed rules preserve old identities. No old cost/role model is silently
broadened to fit a new implementation. The source traces and checks remain the
entry point for later scoped replay; agent-level retrieval-to-code experiments are
not required to finish this source extraction.

## GeForce convolution frontend (active device trace)

Retrieve implicit_patch_contraction@1 when spatial windows feed GEMM. Its
`program` fixes Q/P/Z/batch output flattening, C/S/R/T reduction flattening and
lower-side padding. Dense `_setup_conv_tma` selects those logical axis orders;
`_compute_im2col_params` reverses descriptor axes into WHD. This conditional binding
keeps `c_geometry` and `pre_backend`: actual entry predicates accept C0 and dense
negative output axes whose product is positive. The scaled entry admits only
NVFP4 and MXFP8 with whole scale atoms; its shared dispatch helper's broader menu
is not a connected convolution capability. See the
[frontend trace](trace-geforce-conv-frontend.md) for exact fields and guards.
Scale/software-copy coordinates, traversal, CLC/parity ownership and output
lifetimes remain unbound until the device trace is complete. No new rule is added.

## GeForce convolution input coordinates

The old patch rule's `program` supplies the hardest mapping: lower-side padding,
dilated taps and the same input pixel for data and SFA. The
[input trace](trace-geforce-conv-inputs.md) executes both direct and hoisted scalar
expressions. Hoisting/packing itself has no old field binding; signed16 H/W bases
and representable offsets are required, and the accepted H32769 source-menu
counterexample blocks unconditional backend reuse. `byte_arrival_completion_join@1`
binds128 independent SFA copy completions plus one elected TMA registration owner
and A/B/SFB transaction bytes; the source uses immediate per-stage arrivals, not
the SM100 delayed schedule. Native incidence/encoding/progress remain premises.
The two convolution sources explicitly allocate four order slots; the older
standalone GEMM two-slot declaration gap is not copied into these bindings.

## GeForce convolution side streams and outputs

Retrieve interleaved_consumer_streams@1 per fixed-length stream: input K tiles,
bias1 and residual epilogue-subtile count. Its `steps_per_item` and `program`
supply the hardest N96 residual mapping (six subtiles); skipped items advance
without release. Both source order arrays declare four slots. Retrieve
staged_async_output_store@1 for commit/acquire source-read waits plus the explicit
tail before peer epilogue handoff; source-read completion is not output observation.
The [output trace](trace-geforce-conv-outputs.md) records exact fields and checks.
Dense FP16 accumulation casts bias to FP16 despite FP32 comments, so numerical
admission remains conditional. Host layouts/resource policy and native semantics
are not supplied by these old stream/output fields.

## GeForce convolution closure

The [layout/host trace](trace-geforce-conv-layouts.md) completes both entries.
Host reference expands stored scales and crops each filter position's padded
span; random1/2 scales are fixtures, not output quantization, so no
`groupwise_low_float_quantization@1` binding is claimed. Actual SF N rounds128;
narrow custom-helper domains are not connected host branches. All13 size-probe
fields match the runtime allocation. Stage maximization is a capacity heuristic,
not a universal optimum. Existing patch/transfer/stream/output choices remain
conditional on native layouts, numeric policy, fixed-width ranges and CLC progress.
No candidate name/version or old proof domain changes.

## GeForce dense entry (active correction)

Retrieve persistent_tile_scheduler@1 with `fetch=static_stride` for independent
fixed-shape work, and staged_async_output_store@1 for output buffers. The
[entry trace](trace-geforce-dense.md) matches the hardest output edge: a post-acquire
math barrier plus per-tile source tail before buffer reuse. No alternating-group
rule is needed. The actual selected MmaF16BF16Op rejects the front gate's8-bit
inputs. Source stage arithmetic does not guard negative results, and exact native
StMatrix/TMA layouts remain open. Old names do not supply those missing bindings.

## GeForce dense source closure

The static-stride old rule's logical worker index maps to grid.z, not literal
blockIdx.x: create/grid/advance and M-first coordinate decode replay144 domains.
The old grid domain1..4096 and independent-output premise remain. The
[correction trace](trace-geforce-dense.md#scheduler-layout-and-host-correction--source-closure)
separates output store instruction from its geometric StMatrix atom; non16-bit
native typed partition is not established. Exact epilogue tiles and host fixture
values are recovered. No old rule/proof domain changes or new candidates.

## FMHA forward decisions (active)

Retrieve online_rescaled_reduction@1 for exact running-max state and
approximate_math_rewrite@1 for folded exp2. The
[decision trace](trace-fmha-contracts.md) identifies a missing finite-reference
selection policy: freezing the reference can preserve normalization but violates
the old `invariant` requiring the true prefix maximum. masked_tile_pruning@1's
`pre_neutral` cannot supply threshold-based removal of nonzero P/PV work. The
source's per-element maximum bound omits total dropped mass and value range.
No old proof is broadened, no new identity is committed before recurrence and
protocol reconciliation; both gaps remain explicit. Historical gap files do not
supersede the actual pinned candidate bodies now present in the pool.

## FMHA frame revision

online_rescaled_reduction@2 preserves version1 and its five variable domains,
adding reference policy, lag threshold and weight gain. Its framed invariant owns
common denominator/numerator/sink scaling and LSE log compensation. The
[frame trace](trace-fmha-frames.md) and
[migration](delta/online_rescaled_reduction@2/migration.md) distinguish the finite
positive-mass projection from changed zero-mass/cost/backend obligations. Source
FP8 gain256 plus unscaled sink fails `extra_denominator_terms_share_frame`;
nonzero pruning fails `no_nonidentity_terms_dropped`. No backend assurance or
version1 proof is inherited across either failure.

## FMHA bounded nonidentity pruning

bounded_normalized_weight_pruning@1 introduces a certificate object, unlike
masked_tile_pruning@1's identity-only `pre_neutral`. The hardest new decision is
whether total omission respects the caller's absolute error: B/(K_lower+B), value
diameter and output scaling bound it, not tau alone. The
[pruning trace](trace-fmha-pruning.md) separates exact rational assurance from
source threshold proposals and native protocol obligations. Restrict the original
operator under that certificate, then apply the frame rule to its complete kept
operator; do not set the frame rule's no-dropped-terms premise on the original
operator. Source does not compute cumulative bounds and its host reference does
not freeze maxima like the device. Neither gap is silently reused as assurance.

## FMHA masks and crossed read retirement

physical_storage@1's `pre_lifetime` with level=tmem supplies the last-read-before-
next-write criterion; source S0/P1 and S1/P0 aliases add offset permissions. The
[mask/channel trace](trace-fmha-masks.md) distinguishes this from
per_slot_handshake@1's acquire-empty/release-empty program: the source uses one-way
full notifications and compensating prologue/tail counts, so full_signal alone
does not recover it. masked_tile_pruning@1's `pre_neutral` matches exact masking,
but its one peeled-step/first-phase fields do not encode all three source segments.
A host empty-row guard is not a positive-mass proof. Both solver-interface gaps
and native cross-work reuse remain explicit; no old proof is broadened.
