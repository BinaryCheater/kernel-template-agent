# Routed GEMM, output scale recipes and ordered weight-gradient tails

Sources below refer to the retained [tree](inputs/sources.json). This pass
continues `flashinfer/prims_ts/batched_gemm` and the explicit
`flashinfer/experimental/cake_moe_grouped_det_gemm` training backend.

## Exclusive partials separate completion order from arithmetic order

Retrieval cue: parallel producers may finish in arbitrary order, while a fixed
plan requires a reproducible floating-point sum. New
[ordered_workspace_combine@1](delta/ordered_workspace_combine@1/template.yaml)
and [proof](delta/ordered_workspace_combine@1/proof.yaml) assign exclusive FP32
partial slots and merge them in a second kernel by logical split index.

Route: `flashinfer/grouped_mm/cake.py` -> prepared deterministic package ->
`cake_jit.py:wgrad_bf16_k256` selects generated main
`cake_moe_grouped_det_gemm_ed9a74f36ba13a52183d_kernel.cu` and tail reducer
`cake_moe_grouped_det_gemm_bc774d802e4f0b3ac27c_kernel.cu`. Filenames below
are under the package's `csrc/cake_moe_grouped_det_gemm/`.

| Source choice | Candidate field / obligation | Binding |
|---|---|---|
| Main :674-686 | `split_count`, `range_and_presence` | Nominal split s owns[floor(s*T/S),floor((s+1)*T/S)), T being the group's row-reduction step count. Slots can have holes: T2/S8 has live slots3 and7, not slots0 and1. |
| Main :686-757 | `pre_slots`, `exclusive_ownership` | Tail tile j uses partial slot j*S+s. Multi-contributor useful ranges store FP32 fragments independently; no shared-output atomic update determines their order. |
| Reducer :65-143 | `reproducible_sequence`, `inactive_slots` | Reconstruct expert/tail-tile identity and the same ranges. Loop s ascending, read only hi>lo slots and add FP32 vectors in that order. Completion order cannot change this sequence. |
| Reducer :143 onward; package README numerics | `publication_and_precision` | One final BF16 conversion or FP32 store follows the ordered sum; output leading strides preserve row padding. Reproducibility is scoped to the same partial partition, MMA arithmetic and merge plan. |
| Main :681-737; reducer count guard | Separate singleton/empty branches | A whole-range single contributor writes output directly; reducer writes only for count>1. The new rule's T>=2/S>=2 projection covers actual multi-contributor tails. The shared native utility's unused singleton scratch loads remain outside that projection; no singleton publication guarantee is invented. |

The tail-wave planner chooses which output tiles split and the nominal count;
that policy is independent of the merge identity. Empty groups and direct main
outputs have their own exact-zero/ownership branches. `deterministic=False`
currently selecting the same source programs is not a separate algorithm.

ordered_global_combine@1 serializes producer CTAs using a lock and repeatedly
updates an output buffer. It does not encode exclusive workspace plus a separate
merge launch, and may introduce intermediate low-format rounding.
split_combine_kernel@1 instead merges normalized attention O/LSE states.
Neither is silently rebound to these linear FP32 partials. No claim is made
that changing split count or mainloop shape preserves the same bits.

The [logical check](checks/check_ordered_workspace_and_routes.json) covers88
floor-partition cases, poisoned inactive slots, six producer completion orders,
FP32 cancellation and a BF16-intermediate counterexample. Fixed ascending
partials[2^24,1,-2^24] give0; arrival-order addition can give0 or1. Native
stream publication, MMA association and resource lifetime remain obligations.

## Routed payload and scale indices must agree

Representative resources are `prims_ts/batched_gemm/smem_ab_resources.py` and
`smem_sf_resources.py`.

| Source decision | Existing rule / field | Binding |
|---|---|---|
| AB :731-805 | indexed_gather@1 granularity=row, registers prefetch | Prepare route rows once per output tile, with operand A/B chosen by swapAB and physical2CTA row offset. Reuse a route across K boxes only when the thread/copy mapping assigns the same logical row. |
| AB :915-944 | vectorized_access@1 vector_bytes=16 | cp.async fetches16-byte chunks from routed_row*activation_stride plus the current K-box offset, then writes the declared shared swizzle. Live route IDs must be valid; a logical tile limit is not a general arbitrary-negative route sanitizer. |
| SF :577-634 | indexed_gather@1 plus metadata identity | Scale gather4 caches four row IDs per issue group; the same routed token identity must select its activation and scale row. Padding uses an out-of-bounds TMA coordinate rather than token0. |
| AB :805 onward; SF local-limit helper | boundary_predication@1 binding | TRT-style absolute end-row limits become local counts for the current token tile. Payload/SF stage behavior for padded rows must meet the downstream arithmetic/quantizer neutral-value premise; an omitted LDGSTS copy is not by itself proof of zero-fill. |

These are read mappings: duplicate route rows represent repeated expert inputs
and must not be deduplicated by first_occurrence_record_scatter@1. The latter
resolves duplicate **write destinations**. The logical witness route[2,0,2]
with distinct row scales distinguishes correct paired gather from gathering
payloads while leaving scale rows in their old order.

Task/scale pipelines can cache route rows for different physical durations.
Their common logical route generation must remain immutable until every gather
finishes; matching pointer shapes alone does not guarantee that identity.
Existing atom_owned_cluster_multicast@1 handles cluster masks separately from
this token/scale mapping. Routed LDGSTS and multicast menus have their own
support restrictions, not a freely composable cross-product.

## Low-bit output recipes: group closure and scale choice are separate

`gmem_c_resources.py` is the runtime output resource. In contrast,
`batched_gemm_quant.py` is explicitly test-data initialization and is not used
as evidence of the runtime quantizer.

| Native choice | Existing rule / gap | Disposition |
|---|---|---|
| :2514-2583,2719-2745 | Group reduction/publication obligations | M16 warp maxima publish to shared scratch, a barrier joins them, and a partner warp completes an M32 group. Batched write/read phases can amortize barriers only while scratch survives every reader. A local half-group maximum is insufficient. |
| :3118-3165 NVFP4 | groupwise_low_float_quantization@1 only on its matching finite/positive-scale subset | Compute rounded E4M3 scale from complete amax/6, generate codes from that decoded stored scale, not the ideal pre-round value. Source rounded-scale0 explicitly selects multiplier0; this differs from the older rule's zero reciprocal-clamp sequence, even when all-zero codes agree. Global scale_c placement has its own decoder contract. |
| :3170-3224 MXFP8/MXFP4 | Distinct floor-anchor recipe gap | Scale=trunc_pow2(amax)/256 for E4M3 codes or /4 for E2M1 codes, packed as UE8M0. This is not groupwise_low_float_quantization@1's least power of two above amax/448. MXFP4 is also outside its nvfp4/mxfp8 format menu. |
| Positive normal MX scales | power_of_two_scale_encoding@1 | Once the chosen scale is already a positive normal power of two and its exponent is in range, exponent-byte storage is value-preserving. That rule does not choose the floor-anchor policy or define live zero/subnormal scale behavior. |

At amax500 the floor-anchor MXFP8 scale is1; the ceiling recipe gives2.
Scaled500 then exceeds E4M3's finite448 range in the former recipe. This is a
real numerical-policy choice, not a layout-only reinterpretation. The logical
check records that distinction and full-group reduction, while native saturation,
zero/underflow recipes and packed code/scale coordinates remain source bindings.
A dedicated floor-anchor output quantizer refinement remains extraction work.

## Continuation

Deterministic grouped training now has main/direct/tail ownership and a concrete
ordered-combine candidate. Routed task GEMM has payload/scale map dispositions;
low-bit outputs retain explicit recipe differences. Next extract the floor-anchor
quantizer, reconcile remaining GEMM provider/decoder/menu designs, then advance
through norm/activation, rotary/cache, sampling, recurrent, MoE and distributed
families. The attention/training reconciliation and full-library inventory remain
active; native utility gaps are named without per-entry bug hunting.
