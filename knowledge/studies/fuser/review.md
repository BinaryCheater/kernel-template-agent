# fuser reusable design menu

Status: representative-design extraction complete; proposals pending independent review.

| Kernel need / choice | Rule |
|---|---|
| When a scale consumer expects small column groups interleaved with row subtiles, split coordinates into digits rather than copying a plain transpose | [blocked_scale_digit_layout@1](delta/blocked_scale_digit_layout@1/template.yaml) |
| When each lane holds several outer-reduction statistics, reduce and scatter vector chunks before writing shared partials | [butterfly_reduce_scatter_chunks@1](delta/butterfly_reduce_scatter_chunks@1/template.yaml) |
| When a cluster reduction needs its result at one CTA or all CTAs, choose destination tables before selecting the final writer | [cluster_partial_destination_table@1](delta/cluster_partial_destination_table@1/template.yaml) |
| When a cross-warp reduction reads same-bank rows from several warp partials, shift each warp row digit cyclically before storing | [cyclic_warp_row_skew@1](delta/cyclic_warp_row_skew@1/template.yaml) |
| When independent CTA partials must be read by one cleanup CTA, separate publication from numerical combination | [designated_segment_completion@1](delta/designated_segment_completion@1/template.yaml) |
| When a caller already supplies a positive coarse encode multiplier, bind its provenance explicitly rather than inventing an amax-derived value | [hierarchical_low_float_scaling@2](delta/hierarchical_low_float_scaling@2/template.yaml) |
| When peer buffers have corresponding offsets and a registered multicast address, replace software fanout stores or peer-load sums with the corresponding multimem address operation | [multimem_peer_address_operation@1](delta/multimem_peer_address_operation@1/template.yaml) |
| When a derived tensor must survive a reduction but its immutable producer inputs occupy fewer live bytes, choose whether to retain the tensor or retain its producer frontier and reconstruct it | [persistent_value_rematerialization@1](delta/persistent_value_rematerialization@1/template.yaml) |
| When a pointwise producer computes values later discarded by a slice, propagate the demanded coordinates backward instead of materializing the full producer output | [pointwise_slice_demand_pushdown@1](delta/pointwise_slice_demand_pushdown@1/template.yaml) |
| When locality requires grouping both tile axes, choose separate M/N section sizes instead of one grouped-axis width | [rectangular_section_rasterization@1](delta/rectangular_section_rasterization@1/template.yaml) |
| When fused inputs and outputs require different contiguous lane directions, choose an explicit cache boundary to exchange a tile through shared memory | [thread_ownership_exchange@1](delta/thread_ownership_exchange@1/template.yaml) |

Grouped NVFP4: find expert by row prefix → use expert-local row → form 16-element scale groups and round stored scale → quantize with that stored scale → write scale through blocked digits and expert padded base. Route ownership, numerical scale pairing and physical metadata are independent contracts.

Evidence: [source-backed design traces](trace-design-families.md), [initial concrete bindings](trace.md), [coverage inventory](inventory.md), [executed witnesses](checks/results.md), [retained sources](inputs/sources.json).

Every new/revised rule has proof.yaml beside template.yaml. Native memory visibility, numerical guarantees, compiler mapping and hardware progress are recorded as specific open obligations; schema validation and semantic witnesses do not claim to discharge them. No GPU experiment or commit was performed.
