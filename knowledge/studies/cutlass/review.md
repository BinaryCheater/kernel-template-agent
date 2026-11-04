# CUTLASS study — partial, paused by user

| Observable design situation | Proposed rule | Useful choice |
|---|---|---|
| Reference updates trigger correction or narrow P needs more range | [online_rescaled_reduction@2](delta/online_rescaled_reduction@2/template.yaml) | Select a finite reference/gain frame; compensate all denominator terms and LSE. |
| Small-weight tiles may be omitted under an absolute-error contract | [bounded_normalized_weight_pruning@1](delta/bounded_normalized_weight_pruning@1/template.yaml) | Certify total discarded mass and value range; tau alone is insufficient. |
| Future global tiles are addressable while shared slots are occupied | [cache_prefetch@1](delta/cache_prefetch@1/template.yaml) | L2 distance independently of destination stages. |
| Instruction windows do not divide producer chunks | [cross_buffer_operand_window@1](delta/cross_buffer_operand_window@1/template.yaml) | Current/next addressing; source W96,C256. |
| Async output engine needs a shared source | [staged_async_output_store@1](delta/staged_async_output_store@1/template.yaml) | Ring slots retired after source reads; separate output-write completion. |
| Full-grid work groups finish at different times | [launch_cancellation_work_reuse@1](delta/launch_cancellation_work_reuse@1/template.yaml) | Cancel and take exclusive unstarted work; guard failed responses. |
| Increasing linear work indices cross differently sized problems | [online_group_work_decode@1](delta/online_group_work_decode@1/template.yaml), [@2](delta/online_group_work_decode@2/template.yaml) | Warp-window or scalar retained prefix inversion with optional shape cache; immutable metadata and monotone queries. |
| Metadata needs per-group alignment while data stays compact | [segmented_aligned_storage@1](delta/segmented_aligned_storage@1/template.yaml) | Keep logical and storage prefixes separate; preserve each local layout. |
| Many workers reuse immutable per-problem transfer metadata | [precompute_transfer_descriptors@1](delta/precompute_transfer_descriptors@1/template.yaml) | Build one shared table before execution; encode each base offset once. |
| Fixed-format transfers switch problem addresses/shapes/strides | [rebind_transfer_descriptors@1](delta/rebind_transfer_descriptors@1/template.yaml), [@2](delta/rebind_transfer_descriptors@2/template.yaml) | Private descriptor generations; version2 chooses shared shadows or direct global mutation after reader retirement. |
| Allocated grouped output has only a visible row prefix | [output_prefix_work_filter@1](delta/output_prefix_work_filter@1/template.yaml) | Skip empty cooperative units; keep partial units and require safe, unobserved padding. |
| Spatial cross-correlation windows should feed a GEMM path | [implicit_patch_contraction@1](delta/implicit_patch_contraction@1/template.yaml) | Channel-fast tap mapping, correct leading-side padding and on-load window generation. |
| One input stage mixes engine-counted and thread-completed copies | [byte_arrival_completion_join@1](delta/byte_arrival_completion_join@1/template.yaml) | Join both readiness counters and reserve registration; delayed arrivals need progress-safe slots and tail drain. |
| Requested output is grouped low-float codes plus scale metadata | [groupwise_low_float_quantization@1](delta/groupwise_low_float_quantization@1/template.yaml) | Use decoded stored scales for code generation; explicit finite/zero/identity policies. |
| Integer operand storage is narrower than the floating MMA type | [integer_operand_widening@1](delta/integer_operand_widening@1/template.yaml) | Preserve signedness and coordinate order; pair shuffled storage with its converter. |
| Unequal contiguous groups share a tiled output axis | [segmented_axis_work_decode@1](delta/segmented_axis_work_decode@1/template.yaml) | Round per group, retain physical origins and mask peer tails. |
| Later empty credits certify all earlier readers retired | [ordered_slot_wait_coalescing@1](delta/ordered_slot_wait_coalescing@1/template.yaml) | Wait once per pair, publish each item; preserve exact generations and odd tails. |
| Every CTA needs a complete row scalar before writing its own slice | [replicated_partial_reduce@1](delta/replicated_partial_reduce@1/template.yaml) | Sender-indexed replicated table; exact byte completion and collective scratch lifetime. |
| Unpack and arithmetic instructions expect different floating payload bit positions | [operand_bitfield_placement@1](delta/operand_bitfield_placement@1/template.yaml) | Preserve code and coordinate; prepare each fresh fragment once. |
| Two consumer groups alternate tiles from one input producer | [interleaved_consumer_streams@1](delta/interleaved_consumer_streams@1/template.yaml) | Jump the global stage cursor over unowned items without releasing them. |
| K peers hold matching partial output fragments | [scatter_reduce_output_shards@1](delta/scatter_reduce_output_shards@1/template.yaml) | Distributed final shards with defined FP32 partials and sender-specific mailboxes. |

For a source-shaped M128/N128/K256 FP16/F32 split-K case, CLC assigns a cooperative
group, two nonempty ranges of K64 tiles produce matching FP32 fragments, and each of the two
peers combines and stores half of every128-element per-thread fragment. The
[partial composition](delta/chain.yaml) leaves partition/MMA/fragment implementation
edges explicit. The original CLC coordinate-query wrapper needs guarded lowering;
empty upstream partitions need identity initialization or exclusion.

[Dense trace](trace-dense-gemm.md) accounts for all six Blackwell dense-GEMM entries.
[Physical/output trace](trace-sm103-physical.md) completes the four block-scaled
entry routes. [Forty-one passing commands](checks/README.md) validate the stated
source/AST/schema/interval domains and twenty-three Template/ProofOutline pairs.
[Reuse audit](reuse-audit.md) supplies retrieval cues, supported boundaries and old
field matches or gaps. No independent admission or GPU execution is claimed.

[Grouped trace](trace-grouped-blockscaled.md) accounts for both grouped entries.
The three-problem K768/K0/K1536 witness decodes five output tiles while input work
skips K0; C is rebound and zero output is explicitly produced. The new rules keep
metadata lookup and descriptor publication independent. Exact engine retirement
remains a required backend premise.

[Blockwise trace](trace-blockwise-gemm.md) accounts for three entries. K128 scale
promotion reuses the existing DeepGEMM rule; contiguous groups need aligned row
ownership. Masked output filtering adds one candidate and retains the source
terminal metadata-lookup guard gap.

[Convolution trace](trace-conv.md) accounts for both entries. Its
[block-scaled output trace](trace-conv-blockscaled-output.md) adds the encoded-scale
quantizer, SFB windows, residual ordering and frontend neighbors. It retains
custom-activation identity and cluster16 reader-domain gaps. Numeric extremes and
native fragment/conversion behavior remain implementation obligations.

[Dense grouped trace](trace-grouped-dense.md) adds the direct-global update variant
and a version1 projection. Both K branches complete the source init handshake;
per-group alignment is required despite the last-group-only host guard.

[Mixed-input trace](trace-mixed-input.md) extracts batched integer widening. The
[grouped companion](trace-grouped-mixed-input.md) completes the four-file family:
separately rounded group work, hybrid TMA/SIMT tails and paired empty waits.
Accumulator-side scaling reuses the old partial rule conditionally; a BF16
counterexample rejects unrestricted relocation. Random scales remain a test fixture.

[Dense MoE trace](trace-moe-dense.md) adds scalar expert decode and an immutable
transfer-descriptor table. Backward K0 outputs are explicitly zero; source
sentinel/tail state and unsupported inline setup remain gaps.

[Scaled MoE metadata](trace-moe-scaled-metadata.md) separates real-token and padded
scale origins and checks the host blocked order across chunked prefix boundaries.
Native encoded padding neutrality remains open.

[Scaled device trace](trace-moe-scaled-device.md) completes the five-file MoE
family: overlap lifetime, K0 clocks, final scaling and frontend/reference routes.
No new rule is needed for this slice. Accepted-but-invalid K768 geometry and a
scale-factoring counterexample remain explicit boundaries.

[Row reduction/RMSNorm](trace-row-reduction.md) adds replicated scalar-table
reduction for CTA-local and cluster results. Configuration coverage, whole-vector
bounds, predicate-false fill and native numerical/communication behavior remain
explicit boundaries.

[GeForce dispatch](trace-geforce-dispatch.md) adds code-preserving register payload
placement for mixed FP4/FP8 and retains the unselected K64 helper boundary.

[GeForce schedules](trace-geforce-schedules.md) adds alternating consumer cursors
and register-capture lifetime. Order storage backing, cooperative output drain,
K0 and reverse mixed alignment remain explicit source obligations.

[GeForce layout/host trace](trace-geforce-layouts.md) completes the three-file
blockscaled family. Separate global/shared scale maps and fixture/reference
policies are checked. The order-slot addresses fit alignment padding in the scalar
model; native struct backing remains an obligation, not an observed overflow.

Next: the two GeForce convolution files, then remaining DSL/C++ families.
End-to-end agent replay is deferred; retained bindings and evidence are ready for
scoped later replay.

[GeForce convolution frontend](trace-geforce-conv-frontend.md) adds conditional
reuse of the old patch coordinate rule and checks both actual entry menus. C0,
negative output axes and zero stride remain excluded by the rule even where a
source predicate accepts them. Device schedules, scale maps and output paths are
next; these two entries are not yet counted as fully source-extracted.

[GeForce convolution inputs](trace-geforce-conv-inputs.md) bind patch coordinates
and the129-arrival completion join conditionally. Recomputing coordinates and
hoisting them agree in the tested signed16 range; an accepted large-H geometry
exposes the packed-base boundary. Scale pairing uses each position's padded SF
span. Math/output/host and full scheduler protocol extraction remain active.

[GeForce convolution outputs](trace-geforce-conv-outputs.md) bind striped input,
bias and residual streams separately, including N96's six residual subtiles.
Both paths drain output source reads before sharing D with the peer. The dense
FP16 accumulator branch retains its own conversion policy. Host scale/reference
and shared-layout/resource reconciliation remain before family closure.

[GeForce convolution layouts/host](trace-geforce-conv-layouts.md) completes both
convolution entries: host scale blocking, reference cropping per filter position,
13-field size-probe agreement and stage selection have source dispositions.
The correction pass distinguishes wrapper guards from predicates and selected
SF N128 from helper-only narrow layouts. Native gaps stay explicit. Next: GeForce
dense GEMM; the full CUTLASS assignment remains active.

[GeForce dense entry](trace-geforce-dense.md) compares width gates with the actual
selected f16/bf16 instruction, and records static work assignment plus output
retirement. Oversized output stage requests can leave negative input stages
outside the zero-only guard. Scheduler/output layout and host correction remain.

The [GeForce dense correction](trace-geforce-dense.md#scheduler-layout-and-host-correction--source-closure)
closes its source extraction: grid.z static ownership, exact epilogue coverage,
selected output stores and signed host fixtures are recovered. Native output typed
partition/capacity/precision remain explicit. Next: Blackwell attention's
[12-file inventory](inputs/attention-variants.json), starting with FMHA forward/backward.

[FMHA forward decisions](trace-fmha-contracts.md) separate two missing choices:
freezing a finite softmax reference versus dropping nonzero softmax/PV tiles.
Existing online-max and exact mask-pruning fields cannot supply those policies.
Threshold-only decisions have no global discarded-mass/value-range error bound.
Full recurrence, precision/protocol and backward tracing remain active.

When reference updates or narrow P storage matter, choose
[online_rescaled_reduction@2](delta/online_rescaled_reduction@2/template.yaml):
finite reference lag and common gain preserve one framed state. Sink terms must
share that gain; LSE must remove it. The [frame trace](trace-fmha-frames.md) records
the source gain256/sink mismatch. Version1 remains preserved; threshold-based
nonzero pruning and native protocol/precision are separate open work.

[FMHA pruning](trace-fmha-pruning.md) adds a certificate for approximate restriction:
error is bounded by diameter*B/(K_lower+B), with explicit scale/numerical allowance
and optional natural-LSE bound. Then the complete kept operator can use the frame
rule. Source threshold execution computes no such certificate, and its host mask
reference differs when device reference freezing is enabled. Full native protocol,
masks/empty rows and backward remain active.

[FMHA masks/channels](trace-fmha-masks.md) recovers exact leading/unmasked/trailing
classification, host residual selection and crossed S/P read-retirement events.
The host's empty-row guard is incomplete; two offset boundary events are necessary
in the bounded channel model. Three-segment solver fields and native cross-work
ownership remain open rather than being hidden inside older two-phase rules.

User-directed checkpoint: CUTLASS is partially studied and will not be resumed in this catalog pass unless requested. See [scope checkpoint](study.md#user-directed-checkpoint) for saved work and unread families.
