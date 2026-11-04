# cccl reusable design menu

Status: representative-design extraction complete; pending independent review.

| Kernel need / choice | Rule |
|---|---|
| When reproducibility must survive changes in reduction partition rather than merely fixing a tree, select an exponent-binned partial-state representation | [binned_floating_accumulator@1](delta/binned_floating_accumulator@1/template.yaml) |
| When boundary candidates may exceed scratch capacity, keep original identities and fall back to replaying the current prefix predicate over original input | [bounded_candidate_materialization@1](delta/bounded_candidate_materialization@1/template.yaml) |
| When parallel set operations encounter duplicate runs crossing tile diagonals, rebalance the cuts through equal-key prefixes before serial intersection/difference/union | [duplicate_balanced_set_partition@1](delta/duplicate_balanced_set_partition@1/template.yaml) |
| When a scan must reset accumulated values at segment starts, lift the value to head-count plus suffix aggregate | [head_reset_scan_state@1](delta/head_reset_scan_state@1/template.yaml) |
| When an in-place adjacent difference reads across CTA boundaries, save just those cross-tile neighbors before launching the output pass | [inplace_neighbor_boundary_checkpoint@1](delta/inplace_neighbor_boundary_checkpoint@1/template.yaml) |
| When merging long sorted runs, partition by output diagonals so each executor receives a bounded amount of merged output even if one run dominates | [merge_path_partition@1](delta/merge_path_partition@1/template.yaml) |
| When looking for the first matching index, use the best published index to stop workers whose future tiles start later | [monotone_first_match_pruning@1](delta/monotone_first_match_pruning@1/template.yaml) |
| When a scan spans CTAs, publish local aggregates so each tile can recover its exclusive prefix without a global barrier | [ordered_prefix_lookback@1](delta/ordered_prefix_lookback@1/template.yaml) |
| When many samples contend on the same output bins, accumulate per-worker tables and flush counts | [privatized_bin_accumulation@1](delta/privatized_bin_accumulation@1/template.yaml) |
| When a radix top-k contract requires a particular original-index tie set, choose any, smallest_index or largest_index and bind the matching ordered boundary emission | [radix_threshold_selection@3](delta/radix_threshold_selection@3/template.yaml) |
| When repeated radix passes outgrow resident shared memory, keep a prefix or suffix window and re-stream the rest through bounded slots | [resident_overflow_repeated_scan@1](delta/resident_overflow_repeated_scan@1/template.yaml) |
| When a batch mixes tiny and large tasks, choose size thresholds and cooperative width rather than giving every task a full CTA | [size_class_cooperative_assignment@1](delta/size_class_cooperative_assignment@1/template.yaml) |
| When integer or transformed scalar keys can be sorted by digits, scatter through bin base plus earlier-tile count plus local rank | [stable_radix_digit_scatter@1](delta/stable_radix_digit_scatter@1/template.yaml) |

Stable radix scatter: global bin base + earlier-tile same-bin prefix + local stable rank gives one output position per original key/payload. Onesweep changes count publication scheduling; it preserves stable digit order. Duplicate-aware set partition and ordered segmented state are separate refinements.

Evidence: [source-backed design traces](trace-design-families.md), [initial concrete bindings](trace.md), [coverage inventory](inventory.md), [executed witnesses](checks/results.md), [retained sources](inputs/sources.json).

Every new/revised rule has proof.yaml beside template.yaml. Native memory visibility, numerical guarantees, compiler mapping and hardware progress are recorded as specific open obligations; schema validation and semantic witnesses do not claim to discharge them. No GPU experiment or commit was performed.

When a warp group is noncontiguous or has a non-power-of-two member count, [sparse_lane_rank_reduction@1](delta/sparse_lane_rank_reduction@1/template.yaml) reduces in logical member rank and preserves ordered associative operands. cudax caller barriers and root/grid tables are recorded in the design trace.
