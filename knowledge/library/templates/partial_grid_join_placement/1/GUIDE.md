# partial_grid_join_placement@1: exclusive partial grid completion boundary

## When to consider

When a kernel produces independent FP32 partials, choose a cooperative in-kernel join or a following completion-ordered reducer. Couple the choice to the actual CTA budget, tile-specific live partial rows and fixed reduction tree; changing the grid is a numerical partition change.

## Computation before and after

Exclusive partial reduction with an explicit in-grid cooperative or following-kernel completion boundary.

Derive actual main grid B=ctas_per_sm*(SM_count-reserved_sms). CTA b owns channel tile t=b mod T and row rank=floor(b/T). Tile t has G_t=ceil((B-t)/T) existing producers; allocate ceil(B/T) capacity rows but only read rows0..G_t-1 for its channels. Partition token work with that G_t and the configured substreams, accumulate FP32 local partials and publish every existing partial, with zero for empty streams. With cooperative_grid, every CTA joins the cooperative grid barrier before the unique final channel/tap reducers read partials in the same kernel. With following_kernel, ordinary producers never wait for other unscheduled CTAs; a later reducer joins whole-producer completion before reads. A supported dependent launch may schedule its independent prologue early but does not remove the completion wait. Reduce live rows by the declared lane-strided FP32 sums and halving tree. Treat a changed B as a changed numerical partition, even if the final mathematical sum is the same. Keep scratch until the final reducer completes and publish final gradients before consumers.

Recover the original computation: Forget partial ownership and the chosen completion boundary after expanding every live partial into its unique token terms. Both organizations recover the complete reduction under their admitted finite-precision partition; holes are not terms and early scheduling is not an arrival.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| join_site | compile / software | ["cooperative_grid", "following_kernel"] | Use an all-resident cooperative producer grid with an in-kernel join/reducer, or an ordinary producer followed by a completion-ordered reducer kernel. |
| reserved_sms | compile / software | {"lo": 0, "hi": 1024} | Subtract this many SM-equivalent CTA budgets from the main grid. Source records bind0 for cooperative and1 for following-kernel mode. This does not reserve or select physical SM IDs. |

## Coupled constraints and scope

FlashInfer Cake deterministic KCP width4 depthwise-convolution backward, BF16 activation/dx and FP32 rank-local dweight. Native cooperative record uses B=2 or3 CTAs per SM with reserved_sms0; separate-launch record uses the same per-SM resource menu and reserved_sms1, with PDL-enabled reduce binding. Fallback versus st8 changes resource and partition details but not the completion choice. This rule does not provide a cross-rank allreduce, claim guaranteed overlap, or permit a hand-built global barrier on an ordinary oversubscribed grid.

Scope predicate: `(reserved_sms < p.sm_count and p.ctas_per_sm >= 1 and p.channel_tiles >= 1) and (p.main_grid_channel_tile_and_token_partition_explicit) and (p.exclusive_fp32_partial_slots_written_including_empty_stream_identities) and (join_site != 'cooperative_grid' or p.cooperative_launch_and_whole_grid_residency_contract_supported) and (join_site != 'following_kernel' or p.reducer_joins_actual_predecessor_completion_before_partial_reads) and (p.grid_budget_channel_coverage_and_workspace_capacity_consistent) and (p.partial_inner_order_and_lane_strided_fixed_merge_tree_admitted) and (p.partial_storage_generation_and_unique_final_reducer_valid) and (p.selected_target_launch_record_and_partial_reducer_implemented)`.

- Precondition `pre_partition` (semantic): `p.main_grid_channel_tile_and_token_partition_explicit` — For each channel tile, CTA b owns tile b mod T and partial row floor(b/T). Its actual producer count is G_t=ceil((B-t)/T), not the global row capacity ceil(B/T). Token substreams use that tile-specific count and cover each live output/reduction term once.
- Precondition `pre_slots` (implementation): `p.exclusive_fp32_partial_slots_written_including_empty_stream_identities` — Every existing CTA/channel/tap partial has one writer and is fully stored before the join. Empty token streams publish zero. Nonexistent CTA rows for a channel tile are capacity holes and never read.
- Precondition `pre_cooperative` (implementation): `join_site != 'cooperative_grid' or p.cooperative_launch_and_whole_grid_residency_contract_supported` — In-kernel global join is a real cooperative launch supported by the selected kernel/resource configuration and grid size. All CTAs execute the grid join, including those with no live token work. Ordinary oversubscribed grids cannot emulate it with a spinning barrier.
- Precondition `pre_separate` (implementation): `join_site != 'following_kernel' or p.reducer_joins_actual_predecessor_completion_before_partial_reads` — A later reducer uses ordinary completion order or a supported programmatic dependent-launch wait before any partial read. Early launch scheduling release alone is not payload visibility. The ordinary producer contains no whole-grid wait.
- Precondition `pre_geometry` (implementation): `p.grid_budget_channel_coverage_and_workspace_capacity_consistent` — Grid B is positive and covers every channel tile; source computes B=ctas_per_sm*(SM_count-reserved_sms). This is a CTA-count budget, not physical SM affinity or guaranteed stream overlap. Producer, reducer, workspace and main_ctas agree on the actual B.
- Precondition `pre_numeric` (implementation): `p.partial_inner_order_and_lane_strided_fixed_merge_tree_admitted` — Native rows merge per lane r=lane+32*k in ascending k, then a fixed halving16/8/4/2/1 tree. Each launch geometry has its own deterministic token partition/partial order. Moving the join while changing B does not promise identical floating-point bits.
- Precondition `pre_lifetime` (implementation): `p.partial_storage_generation_and_unique_final_reducer_valid` — Partial buffers retain their allocation/generation through all reducer reads. Final channel/tap elements have one owner, following computation reads only completed final gradients, and buffers are not concurrently recycled.
- Precondition `pre_backend` (implementation): `p.selected_target_launch_record_and_partial_reducer_implemented` — Source deterministic KCP depthwise-convolution backward on SM100/103/107, FP32 four-tap rank-local gradients. Selected fallback/st8 resources, channel plan and launch attributes are actually supported; graph/capture restrictions and collective readiness remain separate.
- Constraint `c_grid` (implementation): `reserved_sms < p.sm_count and p.ctas_per_sm >= 1 and p.channel_tiles >= 1` — Grid count remains positive and selected kernel supports a positive co-resident CTA budget.

## Local example and misuse

For a local instance of `reduce`, identify the object `exclusive_partial_grid_completion_boundary` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/partial_grid_join_placement@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/partial_grid_join_placement@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Padded FP32 partial rows indexed by CTA/channel tile, with tile-specific live row counts and final unique reduction ownership.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `partition_and_holes` (test): Producer geometry covers each reduction term once; final reducers read all and only written tile-specific partial rows.
- `global_join` (proof): Every reducer read follows a valid cooperative whole-grid join or actual preceding-grid completion, with supported producer progress.
- `numeric_partition` (test): Within each actual grid geometry, inner partial order and final lane/tree order are fixed; cross-geometry comparisons respect their separate rounding contract.
- `last_reader` (test): Partial and final output generations stay owned until their complete readers finish.

Changing grid size changes token stream bounds and partial association; two reproducible forms can produce different dweight bits. Reading capacity holes or treating PDL scheduling release as completion gives incorrect results.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
