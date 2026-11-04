# Blackwell blockwise GEMM and its two grouped layouts

This slice accounts for all three entries in
`examples/python/CuTeDSL/cute/blackwell/kernel/blockwise_gemm/`:
`blockwise_gemm.py`, `contiguous_grouped_gemm.py`, and `masked_grouped_gemm.py`.
[Variant bindings](inputs/blockwise-variants.json) retain source paths, hashes and
method spans. Paths and lines below refer to the pinned tree in inputs/sources.json.

## Witness and public boundary

All three frontend predicates accept FP8 E4M3 A/B, FP32 raw/final accumulation,
BF16 C, MMA128×128, one CTA per instruction, cluster1×1, M256/N256/K256/L3.
The actual dtype gates also accept E5M2 inputs and FP32 outputs, despite the narrower
introductory prose. They reject a K257 neighbor on input alignment, and require
K-major A/B. Shape predicates do not validate scale dtype, immutable metadata,
padding observability or all runtime group-table properties; those remain caller
premises. Frontend acceptance alone does not establish a supported backend launch.

The standard entry uses independent batch matrices. The contiguous entry flattens
A/C and A-scales into one M axis, using a per-row group lookup for B and B-scales.
An example with group lengths [128,256,0,128] has512 rows and no rows for group2.
The masked entry retains equal allocated M capacity for every group and separately
stores valid row counts; a row count1 still computes/stores part of a full tile.
It promises valid-prefix results rather than preservation of every padded row.

## Decisions, reuse and missing choices

| Decision and source | Binding or branch | Exact rule ownership and boundary |
|---|---|---|
| Standard _setup_attributes:254–297; grouped counterparts:275 onward | Four K32 MMA instruction extents form K128; scale granularities M1/N128/K128 are fixed. Source asserts exactly one K and N scale per CTA tile. | Reuse [block_scaled_partial_accumulation@1](../deepgemm/delta/block_scaled_partial_accumulation@1/template.yaml): partial_k128, gran_k_a/b128, gran_m1, gran_n128. These scale granularities describe the input representation; they are not free tuning knobs. |
| Standard MMA loop:1383–1429; grouped equivalent loops | Set ACCUMULATE=False for each K128 raw partial, then commit that partial to the accumulator pipeline. | Old rule reset_and_promote owns per-partial reset. The existing DeepGEMM Realisation is SM90 WGMMA and cannot implement this SM100 tcgen05 path. A new source/backend binding is required before device replay. |
| Standard:1539–1632; contiguous:1580 onward,1672–1695; masked:1585 onward,1675–1695 | Zero final registers per output tile; load matching A/B scales, compute scale_a*scale_b, then raw*scale+final once per completed K128 partial. Convert to C only in epilogue. | Old rule program and numerical_policy match exactly. K0 leaves initialized final zero. Exact rational checks distinguish a wrong single-final-scale policy; FP32 fusion/rounding/error admission remains open. |
| Grouped constructor/compiler route: contiguous198–202, masked197, run compile options | Contiguous vector codegen uses target>=13.1 or two-CTA at12.9; masked vector path uses>=13.1. Older branch uses scalar per-element expressions; selected option also controls O2/O3. | Same logical mapping, separate compiler/backend instances. Neither exact-real algebra nor matching source expressions proves identical FP rounding or compiler acceptance. |
| Standard:500–527,876–893,931 onward; grouped corresponding tensor views and copy partition | SFA broadcast along K128 with per-row ownership; SFB broadcast over N128/K128. Scale warp loads compact values, update warps view them in C-coordinate layout. | Old rule scale_owner supplies logical coordinates. Zero-stride views and concrete partition_D mapping are physical implementation edges, not independent quantization choices. Exact lane/descriptor encoding remains open. |
| Standard constructor:191–238; grouped constructor and _compute_stages:2297–2307 | Twelve warps: four scale-update, four epilogue, MMA, TMA, scale loader, scheduler. Raw TMEM occupies384 columns with3 stages for128 rows or6 for64; final active region starts384. Scale stages10, tile-info stages2; AB/C derived from remaining SMEM. | Per-region physical_storage@1 lifetime inequality applies, but it does not choose this partition, handoff or role map. multi_buffer_rotation@2 stops at8, so it cannot supply the source's ten-stage scale ring. Keep the stage-domain and role-granularity gaps explicit. |
| Contiguous:1020–1033,1074–1099,1244–1265,1840–1872 | Read group id at global tile-start row; A/SFA/C use flattened rows and batch0, B/SFB use that group id. | Source glue around output_tiling@1's index_map. Neither online_group_work_decode@1 nor tensor_axis_permutation@1 supplies this per-row lookup/segmented ownership. Executable alignment is the decisive premise; a general segmented-row mapping rule remains a gap. |
| Contiguous geometry gate:2499–2507; create_mask:2634–2658 | Cooperative M footprint must be64 or128; generated group lengths are128-aligned. A lookup at one row then names the whole unit. | Checks reject cluster footprint256 in contiguous while standard/masked permit it. A64-row group followed by128 rows would straddle a128-row unit and choose the wrong B. Supporting unaligned groups needs new mapping/tail handling, not a bound override. |
| Masked scheduler:1006–1045 | Subsequent work is retained iff cluster-first-M-row < valid_rows[group]. All peers use the cluster start, not each CTA's local start. Initial work in each role is unfiltered, e.g.1071–1082. | New [output_prefix_work_filter@1](delta/output_prefix_work_filter@1/template.yaml): unit_rows is the whole cluster footprint, bypass_initial=True. It retains partial units; padding accesses/stores must be safe and unobserved. A per-CTA test can split cooperating peers. |
| Masked scheduler:1012–1022, static_persistent_tile_scheduler.py:563–601,622–640 | Advance work, derive coordinates, then read gidx_mapping[group] without a new validity guard. Non-swizzled terminal index can decode batch=group_count. | Concrete terminal metadata-index gap. Proposed rule requires validity before metadata lookup; the pinned backend needs guarding before it satisfies that premise. Scalar replay retains a3-group terminal read at index3, not a successful device execution. |
| Masked:1882–1935 and other epilogues | Final TMEM handoff to epilogue; shared C ring; PipelineTmaStore commit/acquire after each subtile. Descriptor C capacity remains allocated M, not the per-group prefix. | Reuse staged_async_output_store@1 for the retained sequence. Filtering removes work units, not elements within a boundary tile; padding writes cannot be treated as predicated away. |

## Worked composition and evidence

For the masked M256 example, set valid rows [1,128,129] and128-row units. Keep
unit0 in each group and unit1 only in group2, apart from any bypassed worker initial
unit. Every retained visible output promotes two independently reset K128 raw
partials with its row A-scale and N128 B-scale, publishes final FP32 values, then
stages converted C for async output. A whole-unit skip advances none of those
input/compute/output stage streams. The metadata pipeline must send the same kept
sequence to all five consumer roles. The terminal validity guard is an unresolved
backend edge of this composition.

[Checks](checks/blockwise-contracts-result.json) evaluate1029 dtype triples,
20 exact scale/coordinate cases, and cooperative prefix domains across64/128/256-row
units,1/2/4 peers, zero/full/partial prefixes and worker strides. They preserve
wrong final scaling, peer-local disagreement, unaligned row-group ownership and
terminal metadata counterexamples. The [reused binding](inputs/reuse-bindings.json)
substitutes actual fields of block_scaled_partial_accumulation@1; schema validation
also rejects missing padding/metadata-validity/peer-agreement guards for the new rule.
These checks do not establish generated lane maps, pipeline progress or FP32 error.
