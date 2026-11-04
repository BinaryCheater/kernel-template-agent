# SM110 attention ownership and MiniMax probability anchors

All source paths are relative to the retained [revision](inputs/sources.json).
This pass uses one native representative for each distinct design and records
related target/shape menus without per-entry replay.

## SM110 GQA: direct or raw split-state output

`experimental/sm110_gqa_decode/cake_jit.py:108-408` binds seven programs for
exact SM110a. The public contract is **FP16**, head128,32 query heads/8 KV heads,
one query per request. The previous menu's BF16 label was incorrect.

| Source decision | Concrete reuse or gap | Boundary |
|---|---|---|
| `csrc/cake_sm110_gqa_decode_01f1434b7424417bec5c_kernel.cu:70-99,205-240,362-439` within that package | indexed_gather@1 / cross_lane_reduction@1 / boundary_predication@1 concepts | Short CUDA-core route maps CTA to request/KV head; eight feature lanes form each key dot, four query heads reuse K/V. Length1 directly returns V. Otherwise keys outside device length are masked, maxima/mass reduce across key lanes and FP16 P feeds PV. No TMA/TMEM capability is required by this short route. |
| `cake_sm110_gqa_decode_b7f31235adbc7ac497d0_kernel.cu:148-151` | reduction_partition@1 | Split s owns blocks[floor(T*s/S),floor(T*(s+1)/S)). T comes from live device length; capacity chooses the host program, not the live split bounds. Empty splits remain possible. |
| Same file :382-441,464-493 | online_rescaled_reduction@2 frame algebra; raw-state combine | Partial sum is original mass, partial max is scaled base2 reference, partial O is **raw numerator**. Empty state is(-inf,0,0). Merge uses exp2(m_s-M) on both raw O and mass; multiplying raw O by local mass again is wrong. Existing normalized-output/base2-LSE split_combine_kernel@1 is not this representation without an explicit map. |
| Same file :444-459,495-499 | Completion/publication realization gap | Warp publication precedes GPU acq_rel atomic increment; last ticket triggers four row-owning merge warps. Counter resets after the final reader/output barrier. This is not epoch_tagged_completion@1's fixed encoding. Preserve the native cross-warp publication and separate mutable workspace per concurrent invocation. |
| Registry route table and `README.md:61-95` | Compiled tiles/rings and metadata policy | Short, long, N32/N64 and three/four-slot rings are source programs. Accepted split counts are1,2,4,8,10,16. Thresholds route by batch/capacity; they are not measured evidence for an optimizer. |

## SM110 tree: independent page maps and two output halves

Representative source bodies are the shared
`experimental/sm110_xqa/csrc/cake_sm110_xqa_tree_fp8_page128_{tmem_even,tmem_odd,pair}_kernel.cuh`.
The package README:103-178 gives dispatch/geometry; device bodies supply the
following concrete identities.

| Source decision | Rule/field or gap | Binding |
|---|---|---|
| `tmem_even:198-252,513-540` | indexed_gather@1 / boundary_predication@1 bindings | Flattened row=head-local token*GQA+head-in-group; query token=floor(row/GQA). For packed requests, actual Q is adjacent offset difference, while mask stride stays ceil(maxQ/32). Prefix=KVlength-actualQ. Keys before prefix are visible; later keys use bit(key-prefix), with bit1=allow. This is a draft-tree mask, not a causal diagonal. |
| `tmem_even:3178-3187,3205-3214,3241-3250` | Independent gather identities | K page slot=(2*b)*maxPages+floor(key/128); V slot=(2*b+1)*maxPages+floor(key/128). Physical token=page*128+key%128. A shared K/V table would alter the logical operands. |
| `tmem_even:200-202,1158-1227`; `tmem_odd:196,785-795,1454` | Repeated complete scores / online_rescaled_reduction@2 with lag8,gain1 | Two CTAs own output columns[0,256) and[256,512), each computes full scores and full mass. FP16 P is produced **after** FP32 mass accumulation, so score_replication_or_exchange@1's rounded-P-mass premise fails. Its replication idea applies, its exact program does not. K descale multiplies score scale; V descale multiplies final inverse mass. |
| Even/odd prologues and README:110-118 | Cluster multicast and ownership binding | Even Q-tile count uses cluster(2,2,1), Q multicast across output halves and KV multicast across two Q tiles/halves. Odd count uses(2,1,1), preserving the same output halves and full normalization. Native ring/barrier frontiers remain realization obligations. |
| `pair:222-228,1668-1694,1719-1754` | Paired matrix-instruction realization | Cluster rank maps pair_rank=rank%2 and output-half pair_id=floor(rank/2); row_begin=(2*head_tile+pair_rank)*128, output_col=pair_id*256. Even pair leader issues group2 M256 MMAs, spanning two Q tiles; K token halves and V column halves reside in peers. This extends physical cooperation, not sparse-support algebra. |
| README:119-132; pair softmax :821,1226-1253 | Target/menu guard | Pair requires E4M3 KV and even Q-tile count; odd and FP16 route to TMEM. Ratio8 tuned builds alter role counts/prologue only; no automatic native-layout proof follows. |
| README:153-178 | D128 cluster/gmem split alternative | Runtime partition_tokens is positive multiple64. Two/four partitions use DSM clusters; other counts use global raw-state workspace. row_owned_cluster_state_combine@1 is fixed to **three** peers and its row map; bind common-frame algebra only, not that exact candidate to2/4 peers. |

## MiniMax: one score anchor for eight probability scale groups

New [tile_anchor_probability_encoding@1](delta/tile_anchor_probability_encoding@1/template.yaml)
and [proof](delta/tile_anchor_probability_encoding@1/proof.yaml) describe a
quantizer with a separate original-mass denominator. Source package is
`experimental/minimax_h3_varlen_attention`; device filenames below are under
its `csrc/cake_minimax_h3_varlen_attention/`.

`cake_jit.py:694-720` selects FP4-PV on SM100/103 with
`cake_minimax_h3_varlen_attention_1f0285434cb51b016f42_kernel.cu` as the dense
consumer. At :1044-1088 it retains a lag8 running reference and forms a local
score-tile anchor. In scaled base2 coordinates x, local a=max(x), running r:

- lambda=exp2(a-r)/6;
- z_j=6*exp2(x_j-a), zero on masked keys;
- mass increment=lambda*sum(z), before either code or scale cast;
- PV code=E2M1(z), scale=E4M3(lambda), replicated across eight K16 groups.

The source does **not** divide z by the rounded scale. :1090-1238 forms z,
accumulates original mass and casts codes; :1391 and :1690 publish repeated
scale words to low/high scale regions; :1699-1700 updates mass. Thus ideal
z*lambda equals exp2(x-r), but native PV uses code*rounded_scale. A fully
masked span has zero scale/codes/mass; finite positive final row mass is the
normalization boundary. Other divisor16 anchor spans in the candidate are
inferred algebraic alternatives, not claimed native menu entries.

This differs from hierarchical_low_float_scaling@1, whose encoder divides by
the **rounded** local scale and coarse inverse, and from
numerator_weight_gain@1, which uses one fixed gain with a final inverse.
The [logical check](checks/check_probability_anchor.json) covers all four
anchor spans16/32/64/128, a121-key tail, empty spans, reference correction,
scale-cast order and underflow. With one valid key, constant V=1 yields1.03125
under this declared code/scale recipe because its denominator stays1. A tiny
scale may erase all numerator codes' decoded weights while retaining positive
mass. These are properties of the approximation, not an exact convex-average
claim. Native fast exp2, FTZ/FMA and packed scale-coordinate accuracy remain
separate obligations.

### V orientation and FP8-PV relative

The FP4 quantizer `cake_minimax_h3_varlen_attention_5d47bc81a6048253de1c_kernel.cu:325-366`
first stages V, then groups **16 tokens at one feature**, not16 features at
one token. It chooses max(abs(V))/6 floored at2^-9, casts to E4M3, decodes the
rounded scale and uses its reciprocal for V codes. :389-404 places metadata
in separate low/high feature-half arrays with row-quad/row-lane coordinates.
This is a fresh V-transpose quantizer with its own grouping; it is not a
lossless transpose of rowwise quantized V. Exact byte/TMA layout remains a
native realization binding, while the orientation/cast order is source-backed.

FP8-PV consumer `cake_minimax_h3_varlen_attention_3f7f340abbf71c6a17d1_kernel.cu:780-811,1091-1183,1529-1555`
uses lag8 row reference, unit-scale E4M3 P, original FP32 mass and V descale
amax(V)/448 at the epilogue. No tile-anchor scale or gain is implied.
Its quantizer `..._90b49eb7762c49d34e34_kernel.cu:159-198` reduces V amax
before V encoding; Q/K quantization is a separate channel. README:268-350
records the early-QK/PDL alternative: only independent Q/K work may precede
the amax wait, V depends on amax, and attention must wait before reading packed
operands. No claim of publication equivalence follows merely from equal bytes.

Host packed-layout planning and tail-wave splitting reuse partition and
common-frame merge concepts; planner thresholds are source policy. BF16/FP8
variants, plan cache lifetime and split FP16 partial checkpoints stay explicit.
Next reconcile remaining attention/training/model representatives and named
native schedule gaps, then proceed through GEMM and the remaining inventory.
