# Standalone quantizers and row statistics

When several CTAs own one row's output columns, they can recompute the complete
statistic independently or exchange small partials before proceeding locally.
FlashInfer contains both. The exchanged-partial design adds
[replicated_cluster_partials@1](delta/replicated_cluster_partials@1/template.yaml)
and its [outline](delta/replicated_cluster_partials@1/proof.yaml). Keep these
scheduling choices separate from the statistic's exact rounded input and the
quantizer's scale recipe.

Source paths below are relative to the retained [tree](inputs/sources.json).
[Auxiliary inventory](quantization-norm-inventory.md) records the remaining
quantization, norm and activation variants without treating every backend as a
new design.

| Source decision and binding | Existing field or new boundary | Evidence |
|---|---|---|
| Grouped MXFP8 cuTile: `quantization/kernels/cutile/mxfp8_grouped_quantize_cutile.py:294-348`:128x128 tile, row groups32, amax/448 then UE8M0 round-positive | `groupwise_low_float_quantization@1(format=mxfp8,group_size=32)` on nonzero normal decoded scales and matching arithmetic/cast policy | This is ceiling scale, not `floor_anchor_group_quantization@1`. `.astype` native code conversion still needs its rounding/saturation binding |
| `:282-291,312-325` and `quantization_cute_dsl_utils.py:159-246`: UE8M0 byte0 forces code multiplier0; exponent construction for other bytes | Existing quantizer only reused on matching nonzero-byte range; live byte0 is a recipe boundary | Positive tiny groups may be erased even though stored byte0 decodes to positive2**-127. A zero-group witness alone would conceal that difference |
| Grouped scale layout`:330-348`: each128x4 metadata tile has physical offset16*(row%32)+4*(row//32)+scale_col | Concrete permutation; same layout coordinates as the inverse masked-GEMM mapping in [GEMM trace](trace-gemm-remaining-designs.md) |512 distinct labeled coordinates round-trip in the logical check. This preserves already-selected scales and does not change a quantizer |
| Prefix scheduling`:87-117,350-436`: precompute exclusive tile offsets; static CTA-stride walk; binary search first group end strictly greater than work ID | Same inverse-prefix ownership as `online_group_work_decode@2`, but its `search` menu has scalar_forward/warp_prefix_32, neither a fresh binary search over a materialized prefix | Counts=ceil(M_g/128)*(K/128), immutable same-call metadata, zero-count groups skipped and Int32 prefix bounds required. Retained-cursor savings are not claimed; binary-prefix search is a named realization/field gap |
| Standalone MXFP4: `quantization_cute_dsl_utils.py:958-990,1050`:amax*rcp.approx.ftz(6), ceiling UE8M0, matching code reciprocal | Same ceiling-group concept, but old `groupwise_low_float_quantization@1.format` excludes MXFP4; approximate reciprocal before exponent selection is a numerical boundary | Do not replace it with the floor anchor4 recipe or assume exact amax/6 at scale boundaries |
| Per-token group quantization: `include/flashinfer/quantization.cuh:100-139`: FP32 scale division, optional power-of-two scale, then clamped8-bit code | Native arithmetic binding must preserve `__fdiv_rn` for ordinary scales; multiplication by reciprocal only on the power-of-two branch | Fast-math compilation does not authorize reciprocal substitution in the ordinary branch; optional float scale metadata is distinct from UE8M0 byte storage |
| Classic RMSNorm: `include/flashinfer/norm.cuh:62-154`: one row/CTA, sum of FP32 squares, warp then shared combine, reload input for normalized map | `cross_lane_reduction@1(lanes=32,combine_op=add)` + `smem_partial_combine@1(topology=serial,combine_op=add,result_to=all_threads)` + local norm scaling | Vector width=gcd(16/sizeof(T),d), making vector tails whole. Epsilon, weight_bias and admitted rsqrt/FP32 tree stay explicit |
| Fused residual RMSNorm`:416-502`: form FP32 h=input+residual, save h in FP32 SMEM, separately save residual as T, normalize FP32 h | `operator_fusion@2(consumer_kind=row_reduce_then_map,intermediate_site=smem,fused_ops=norm_scale,checkpoint=none)` for the FP32 edge, retaining the separate mandatory cast/store consumer | The immediate FP32 h is not the rounded residual checkpoint. Reloading the residual T would change the normalization input |
| LayerNorm`:941-984,1082-1099`: raw E[x²]-E[x]² or centered second pass; shared caching or repeated input loads | `centered_moment_merge@1` describes centered-statistic composition only; native raw-moment option needs its own admitted cancellation policy | It is not evidence that raw and centered methods agree bitwise. USE_SHMEM changes transport/lifetime, not permission to move casts |
| Gated residual norm: `norm/kernels/gated_residual_rmsnorm.py:23-35,221-286`: each output-segment CTA rereads the complete row using one fixed reduction map | Independent column outputs and full local statistic dependency; `cache_eviction_hint@1`/native cache policy is separate from correctness | No cross-CTA allreduce occurs. Fixed reduction association is intentional; source performance comments are not measurements in this study |
| Gated residual next norm: `norm/kernels/gated_residual_next_rmsnorm.py:29-39,42-113,193-245`: H8192,8 CTAs,4 warps each, two dependent reductions | New `replicated_cluster_partials@1(peers=8,warps_per_cta=4,phases=2)` | Each source warp writes16 bytes to every peer at slot4*rank+warp. Receiver expects512 bytes per phase. Every consuming warp folds the complete32-slot table |
| `cute_dsl/add_rmsnorm_fp4quant.py:702-784`: compute sumsq from FP32 h, save residual T, then quantization loads T h from SMEM or published global residual | Fusion with explicit input checkpoints and cluster publication; numerical/metadata quantization refinement remains separate | Statistics use FP32 h but quantization uses rounded T h. In cluster mode, cluster fence/join precedes rereads of slices written by other CTAs |

The eight-CTA dual norm first reduces b² to form r1, then forms/stores the FP32
residual `res=u+v*r1`, and reduces **that residual's** square for r2. Expanding
its square into u²+2uvr1+v²r1² changes the finite-precision operation. The
[logical check](checks/check_replicated_partials_and_norm.json) gives u=100000000,
v=-99999992 and r1=1: residual8 has square64, while expanded FP32 terms cancel
to0. Keep the intermediate checkpoint even when both normalizations are fused.

The same check covers20 legal cluster shapes and1360 tuple deliveries, matching
each destination's bytes and exclusive source slot. A fast peer's next-phase
store can overwrite a slow peer's unread prior partial if both phases share one
bank. The source allocates two banks and two fresh phase0 barriers, followed by
a terminal cluster retirement join. It is not a rolling two-bank protocol:
[reusable_group_partial_allreduce@1](../transformerengine/delta/reusable_group_partial_allreduce@1/template.yaml)
has a different global-counter and retirement mechanism. Likewise
[row_owned_cluster_state_combine@1](delta/row_owned_cluster_state_combine@1/template.yaml)
assigns attention row owners, whereas this rule replicates tiny statistics so
all CTAs continue their disjoint column shards.

The source witness has exactly32 live slots. Inferred smaller peer/warp groups
need explicit neutral inactive lanes; reading all32 lanes from a smaller table
is not supplied by the source. Native DSM transaction ordering, cluster launch
support and collective progress remain realization obligations. Standalone
quantizer zero/special-range policies, additional8x4 layouts and the remaining
specialized norm/activation/pack variants stay in the auxiliary inventory.

A rounded T residual is not the exact FP32 intermediate required by `operator_fusion@2.pre_checkpoint`. Preserve it as a separate cast/store consumer under `pre_external`; do not label it an exact checkpoint of the immediate FP32 norm input.
