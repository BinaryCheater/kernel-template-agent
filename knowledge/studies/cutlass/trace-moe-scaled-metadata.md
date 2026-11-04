# Scaled MoE: separate data and scale origins

When data groups are concatenated compactly but a consuming scale layout requires
each group to start on an atom boundary, preserve two prefixes. Round each scale
payload segment before taking its storage prefix. A data token offset cannot
select the scale region, even when the total fused extent happens to be aligned.
This is the retrieval cue for
[segmented_aligned_storage@1](delta/segmented_aligned_storage@1/template.yaml).

This slice covers host scale assembly, padded-offset setup and the scaled
extension/descriptor paths in the five-file [MoE inventory](inputs/moe-variants.json).
Paths and lines below refer to the pinned CUTLASS tree in
[the manifest](inputs/sources.json). [The device companion](trace-moe-scaled-device.md) completes mainloop, accumulator
overlap, per-expert global-scale arithmetic and full frontend reconciliation.
The two scaled traces and dense trace close source extraction of the family.

## Concrete metadata witness

Use MXFP8 with E4M3 data, E8M0 scale metadata, sf_vec_size32, hidden256,
intermediate256, four expert token lengths[32,0,160,64] and actual cumulative
ends[32,32,192,256]. The supported CLI forces separate_tensormap_init=true and
consistent_token_padding=false. [The retained binding](inputs/moe-scaled-metadata-binding.json)
records both scenarios. Forward scale rows pad per expert to128, giving padded
ends[128,128,384,512]. Backward scale columns have raw lengths[1,0,5,2] and pad to4,
giving column ends[4,4,12,16]; descriptor offset metadata uses token-equivalent
ends[128,128,384,512], multiplying columns by32.

Forward A scale shape is(512,8); B scale shape is(4,2048). Backward both scale
shapes are(256,16). These are wrapper shapes around the flat blocked data, not
ordinary row-major views to index directly. Reusing the data end32 for scale
expert2 instead of padded end128 selects the wrong region. Separately padded
extents sum to512, while rounding the fused token extent once gives256.

## Source decisions and boundaries

| Decision and source | Actual binding | Rule field or gap |
|---|---|---|
| `torch_scaled_grouped_mm.py:2263–2282,2707–2758` | MXFP8/MXFP4 use scale groups32 with E8M0; NVFP4 groups16 with E4M3 and optional per-expert global scales. Host distributes tokens in multiples of group width, including empty experts. | Workload representation/menu. Raw scales are generated in{1,2} at:2413–2432, not inferred from data amax. No production quantizer is established by those fixtures. Per-expert global-scale arithmetic awaits the device trace. |
| Host shape/assembly:2323–2379,2463–2480 | Forward A segments rows padded128; forward B has a separate fixed scale matrix per expert. Backward A/B segments scale columns padded4, concatenating each expert's individually blocked flat payload. | New segmented_aligned_storage@1 alignment128 or4, fixed by the consumer. Preserve logical/storage origins and segment-local layout. Empty segments consume zero payload; whole-empty tensor reshape behavior is not executed by the scalar checks. |
| Host `to_blocked`:2382–2404 | Pad rows to128, columns to4; view(row_blocks,128,col_blocks,4), permute(0,2,1,3), reshape(-1,4,32,4), transpose(1,2), flatten. | Unique labels recover offset512*((r//128)*col_blocks+c//4)+16*(r%32)+4*((r%128)//32)+c%4. This is a source instance, not reuse of Marlin's fixed offline_operand_prepack@1 interleave or its constant-operand premise. |
| Host fill:2395–2400; `utils/blockscaled_layout.py:40–48,66–86` | Source requests zeros in scale dtype. Atom stride((16,4),(0,1)) broadcasts each scale across its sf group; tile_to_shape order(2,1,3). | New rule requires initialized padding that is unobserved or neutral. E8M0 has no true zero, so a zeros call does not establish the consumed encoding or tail neutrality. Native conversion/fill, clipping and any live padded products remain open. Local atom constants and host permutations are checked; native CuTe tiling is not executed. |
| Caller:616–684,785–800; workspace:175–184 | Scale tensors infer their own padded dimensions; data layouts remain compact. Extra global padded-end array is4×experts bytes only when not aliasing data ends. | Different origin units are explicit family glue. A padded scale column end is multiplied by sf_vec_size before descriptor consumption; unpadded data never receives that origin. Prefix allocation/scan cost is separate from payload storage. |
| Setup:906–947,1007–1059 | One256-thread setup block, two128-thread groups. Group A creates padded ends in128-expert chunks, P128 forward or P=4×sf backward. One-stage pipeline transfers a129-Int32 buffer: carry plus128 ends. | Prefix implementation of the new padding map, not runtime_work_descriptors@1's per-output work arrays. Source also uses four cross-warp totals and two pipeline barriers. Counter visibility/progress require a native binding. |
| Setup:1145–1204 | Round each size, warp inclusive scan steps1/2/4/8/16, cross-warp correction, add carry, publish shared ends, optionally write global ends, retain final total for next chunk. | Chunk carry must equal the preceding complete total. Inactive final-chunk lanes contribute zero but still synchronize. Scalar replay checks expert counts around32 and128 boundaries, including257 experts. |
| `moe_utils.py:714–771` | Group B builds C or AB without waiting for padded ends, then all128 lanes wait; backward SFA/SFB use shared[lane]/shared[lane+1], then every lane releases, including out-of-range experts. | Same [precompute_transfer_descriptors@1](delta/precompute_transfer_descriptors@1/template.yaml) immutable table contract, tuple1 forward or4 backward. Immutable source ends and scale layouts are available before setup; final padded ends are derived inside setup and must be ready before their descriptor use. Two-phase construction is explicit source implementation glue, not a second setup launch. |
| Descriptor constructor:695–707,778–837,844–910 | Backward slots a/b/sfa/sfb per expert. A/B descriptor bases use real token origins; SFA/SFB bases advance MN_padded*padded_token_offset/sf elements and use a fresh per-expert scale layout. | Old descriptor-table variable descriptors_per_problem4 binds within its domain; allocation is512×experts bytes plus4×experts padded-end bytes. Matching key/coordinates, visibility and zero-extent support remain conditional premises. No per-worker mutable descriptor cache is selected. |
| `moe_sched_extension.py:315–429` | Forward A uses true token M origin; SFA uses padded M origin. B/SFB use expert L. Forward C is descriptor-local. Backward AB/SFA/SFB are descriptor-local; C uses expert L. | New padding map composes with the existing data-domain mapping; applying encoded descriptor origins a second time is invalid. A globally reinterpreted backward scale matrix can have a different row-block pitch from the concatenated local matrices. |
| CLI:3859–3864,3884–3889 | Consistent-padding alias and fused initialization are overridden to false/true respectively. | These are not two verified implementations. Direct users of aliasing must prove each segment already meets the required padding and that host storage agrees. Mere alignment of the total does not suffice. Fused setup remains unsupported. |

## Composition and evidence

Host preparation establishes per-expert scale payloads and their local blocked
order. The setup kernel derives padded origins from immutable input ends; its
independent descriptor phase can overlap this prefix calculation. Its dependent
phase waits for the current shared chunk before constructing SFA/SFB descriptors.
The main extension then pairs real-token data origins with padded scale origins
for the same expert. The [partial chain](delta/moe-scaled-metadata-chain.yaml),
[binding](inputs/moe-scaled-metadata-binding.json) and
[checks](checks/moe-scaled-metadata-result.json) retain this partial composition.
No extra setup launch is charged for the prefix computation in the same kernel.
The descriptor-table rule does not establish padding or the scale layout by name.

Checks execute the retained pure shape helpers and padding expression, then replay
the source shuffle steps with simultaneous lane snapshots.512 shape domains,
128 chunked-prefix domains and11808 segment queries agree with independent sums.
Distinct coordinate labels check9216 host-swizzle coordinates and28 descriptor
base expressions. For MN_padded256 and two groups of four scale columns, the local
second row block has offset512; an erroneous globally reinterpreted eight-column
matrix uses1024. Per-expert descriptor rebasing is therefore consequential.

The new rule's executable guards reject a mismatched required alignment, mixed
logical/storage origins, uninitialized padding and unpublished offsets. The old
descriptor-table tuple4 binding passes its constraints with native premises kept
conditional. No Torch/CUDA execution, native descriptor construction, mainloop
validation or performance measurement occurred. The [device companion](trace-moe-scaled-device.md) supplies the remaining
accumulator/global-scale/tail dispositions.
