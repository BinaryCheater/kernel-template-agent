# Prefill: switch PV from key partials to feature owners

When the value output is wide, keeping all features in every KV warp can make
register occupancy expensive. The classic prefill path can keep QK ownership
along keys while changing PV ownership to features. This is independent of
switching the whole attention chain's matrix orientation. Paths below are
relative to the [retained source](inputs/sources.json).

## Common core and observed difference

flashinfer/prefill.py routes fa2/fa3/provider backends into JIT config, csrc
instantiations and include/flashinfer/attention/prefill.cuh or hopper/. The
classic fa2 body uses register MMA, tiled Q/K/V, guarded tails, grouped query
row views, and online rescaled normalization. Existing output_tiling@2,
compute_instruction_selection@1, grouped_head_row_packing@1, boundary_predication@1
and online_rescaled_reduction@2 explain those constituent choices after the
actual layout/numeric bindings are supplied. This trace isolates a new
ownership choice; quantization and Hopper differences remain open inventory work.

| Decision / source in include/flashinfer/attention/prefill.cuh | Candidate / field or gap | Concrete consequence |
|---|---|---|
| :116-132,:366-379: shape-derived warp and VO split | [attention_value_axis_partition@1](delta/attention_value_axis_partition@1/template.yaml), value_ownership disjoint_value_features | Query tiles16/32 have one query-warp group and four KV warps. Ordinary16-bit VO width>256 and divisibility enable the split; width512 assigns128 output features per owner. This is a derived compile/dispatch choice, not an exposed free runtime toggle. Full-feature partials remain the alternative abstract organization, subject to a legal backend/resource binding. |
| :3879-3934: score-shard maxima into shared row table | Common-reference obligation; smem_partial_combine@1/cross_lane_reduction@1 for constituent statistics | Four lanes combine each row's local max, four KV owners publish it, then every reader computes the same running max and old-output correction. Local independently normalized probability tiles would not have the required common frame. |
| :3936-4004: each score shard writes P and local row sum | P exchange object and exact row/key map | For a16-token fragment, row=q_base+mma_q*16+lane/4+8*j; columns=warp_kv_base+mma_kv*16+2*(lane%4)+k and that column+8. Every key-shard weight writes once. CTA join publishes P and the row sums before any feature owner consumes the complete tile. |
| :4007-4052,:4774-4824: feature-owned PV | value_ownership disjoint_value_features | Each warp corrects only its old feature numerator, then reads P for every key block and V for its own feature slice. QK is not repeated. Output numerator per owner shrinks fourfold; the shared P tile and per-tile max/sum joins are added. Performance benefit is unmeasured. |
| :4216-4264: final normalize and output coordinate | Unique output-owner obligation | Every feature owner uses the same final d and writes its original query/head/feature coordinates. No cross-owner output-vector merge is required. Packed query row decoding still applies to output stores. |
| :378-385,:5020-5043: stage and capacity coupling | physical_storage@1 plus source binding | P/statistic exchange must fit shared memory with Q/K/V. Equal QK/VO shapes can reuse K and V storage in a sequenced path; V loading follows the final score readers. The new candidate only prices exchange and output numerator, not whole-kernel storage. |

For weights(1,2,3,4) and value element(key,d)=(key+1)*100+d, every output
feature is300+d. All four feature owners must therefore consume all four keys.
Using only the same-numbered key gives100*(owner+1)+d, a different result.
The [logical check](checks/check_classic_attention.py) covers all512 features
and a second key tile whose larger maximum forces old-state correction;
[results](checks/check_classic_attention.json) agree with the complete weighted
sum. Native probability casts, MMA fragment maps and barrier lowering remain
obligations rather than outcomes of this host arithmetic check.

The FP8 variant at :3950-3968 stores exp weights multiplied by256 while d
still sums ungained FP32 weights; :4227-4231 compensates the numerator by1/256.
That is not online_rescaled_reduction@2's common gain on numerator and denominator.
It also uses a different swizzled P/MMA mapping. The new version is intentionally
bound to ordinary FP16/BF16 P and excludes an uncompensated gain. FP4 KV and
FP8-Q/P decode/conversion paths remain separate numerical/layout bindings.

Comparison: scatter_reduce_output_shards@1 sends already-computed output
partials to output owners. This design exchanges the earlier P tile and changes
which owner performs each PV term. tensor_axis_permutation@1 only changes a
stride view; transpose_attention_chain@1 changes the contractions' orientation.
Neither supplies this score-owner to feature-owner exchange. The design review
therefore keeps one candidate with two organizations, common-frame and complete
P-read guards, and explicit backend legality. A fixed width threshold is a
source dispatch boundary, not a proposed performance law for other kernels.
