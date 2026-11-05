# cluster_granularity_switch@1: cluster cooperative or independent row packing

## When to consider

When a few long rows benefit from cluster cooperation but many short rows do not, classify requests once and use one fixed-width launch with two granularities. Assign all cluster ranks to each cooperative row, then pack one independent row per CTA in short clusters. Preserve request/local-row identity, keep short bodies free of peer dependencies and reprepare request metadata after relevant classification/population changes; surplus grid slots still launch but do no row work.

## Computation before and after

Prepared workload classification switches executor granularity between a full cluster and one CTA while retaining one launch shape.

Prepare a stable request-ID order[long requests,short requests] from key>cooperate_threshold, counts L and S. Launch max_requests*rows_per_request physical clusters of cluster_ctas CTAs. Define A=L*rows_per_request and B=S*rows_per_request; live clusters=A+ceil(B/cluster_ctas). For cluster g<A, request=order[g//rows_per_request], local=g%rows_per_request; all ranks call the cooperative body for row request*rows_per_request+local. Otherwise let j=(g-A)*cluster_ctas+rank. If g is a live short cluster and j<B, request=order[L+j//rows_per_request], local=j%rows_per_request; that CTA alone calls the independent body for its original row. Other short tail ranks and surplus clusters perform no row work. Independent bodies have no cluster collectives/peer reads, so partial short clusters are safe. Cooperative branch choice is cluster-uniform. Keep prepared metadata immutable through all calls and retire both branch roles before reuse; changing classification keys/request population/order/threshold requires reprepare; changing row expansion instead requires a matching consumer and tensor layout.

Recover the original computation: Erase the prepared order and physical grouping. Every original request/local-row pair occurs exactly once as either one cooperative body call or one independent body call, with unchanged row coordinates and the same requested result.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| cluster_ctas | compile / software | [2, 4, 8] | Physical cluster width and number of CTAs that cooperate on one long-class row; short-class rows use one CTA each. |
| cooperate_threshold | compile / software | {"lo": 0, "hi": 2147483647} | Classify request scan key>threshold as cooperative. This is a scheduling threshold; it must not alter the row operation or semantic valid length. |
| rows_per_request | compile / boundary | {"lo": 1, "hi": 32} | Uniform semantic row expansion per request, source next_n. Larger inferred settings require supported body/metadata bindings. |
| max_requests | compile / boundary | {"lo": 1, "hi": 1024} | Prepared table and fixed grid capacity. Source classifier uses one block and at most1024 request slots. |

## Coupled constraints and scope

GVR hybrid source classifies request lengths in scan units, expands next_n rows, and uses physical clusters2/4/8 with cooperative long rows and CTA-local short rows in one fixed-capacity launch. Default key divisor is the workload compression ratio1/4 and default threshold64K scan entries; neither changes actual valid-row bounds. This mapping does not prove the native top-k bodies, permit cluster barriers in partial short clusters, or eliminate launched surplus CTAs. Other request expansions/body types require explicit supported bindings.

Scope predicate: `p.live_requests >= 0 and p.live_requests <= max_requests and p.request_row_grid_and_index_products_fit_int32 and (p.nonnegative_bounded_request_scan_keys and p.scan_key_units_and_generation_explicit) and (p.stable_partition_lists_every_request_once and p.counts_equal_complete_class_populations and p.prepare_threads >= p.live_requests and p.invalid_prepare_lanes_contribute_zero and p.partition_matches_scan_key_threshold) and (p.each_request_has_declared_consecutive_row_ids and p.row_id_equals_request_times_rows_per_request_plus_local_row and p.selected_consumer_expansion_matches_tensor_row_layout) and (p.cooperative_and_independent_bodies_implement_same_row_contract and p.selected_body_shapes_and_targets_supported) and (p.cooperative_branch_uniform_for_all_cluster_ranks and p.all_cooperative_ranks_reach_matching_collectives_and_retire) and (p.independent_body_has_no_cluster_peer_wait_remote_access_or_collective and p.inactive_short_slots_skip_all_output_side_effects) and (p.partition_counts_keys_threshold_and_request_population_same_prepared_generation and p.prepare_completes_before_all_main_consumers) and (p.cluster_launch_supports_selected_width_and_fixed_capacity_grid and p.compiled_combined_resource_budget_matches_launch and p.row_scratch_outputs_private_or_properly_owned) and (p.every_active_row_output_has_one_body_owner and p.all_metadata_and_body_readers_retire_before_reprepare_or_reuse)`.

- Precondition `pre_keys` (implementation): `p.nonnegative_bounded_request_scan_keys and p.scan_key_units_and_generation_explicit` — One bounded nonnegative key per request describes the intended scan/work units. Source uses floor(seq_len/compress_ratio), not raw token length.
- Precondition `pre_classify` (implementation): `p.stable_partition_lists_every_request_once and p.counts_equal_complete_class_populations and p.prepare_threads >= p.live_requests and p.invalid_prepare_lanes_contribute_zero and p.partition_matches_scan_key_threshold` — Prepare a stable partition of request IDs into key>threshold then remaining requests; counters contain complete request counts, not expanded row counts. Preparation threads cover live request slots and inactive scan lanes contribute zero. Class flags equal the declared scan-key>threshold predicate.
- Precondition `pre_expansion` (implementation): `p.each_request_has_declared_consecutive_row_ids and p.row_id_equals_request_times_rows_per_request_plus_local_row and p.selected_consumer_expansion_matches_tensor_row_layout` — Every request expands to the same declared number of semantic rows at unchanged original coordinates. Expansion is a consumer/layout binding; the prepared request partition itself is independent of it.
- Precondition `pre_bodies` (implementation): `p.cooperative_and_independent_bodies_implement_same_row_contract and p.selected_body_shapes_and_targets_supported` — Both bodies implement the requested row operation, including masks, exactness/ties, values and output padding, on supported native bindings.
- Precondition `pre_cluster` (implementation): `p.cooperative_branch_uniform_for_all_cluster_ranks and p.all_cooperative_ranks_reach_matching_collectives_and_retire` — A cooperating row is assigned to all physical ranks of one cluster; its collective participation and retirement are complete.
- Precondition `pre_independent` (implementation): `p.independent_body_has_no_cluster_peer_wait_remote_access_or_collective and p.inactive_short_slots_skip_all_output_side_effects` — Independent mode has CTA-local synchronization only. Tail/inactive sibling CTAs can skip work without leaving a cluster dependency unsatisfied.
- Precondition `pre_metadata` (implementation): `p.partition_counts_keys_threshold_and_request_population_same_prepared_generation and p.prepare_completes_before_all_main_consumers` — Prepared list/counts match request order/population, classification keys and threshold. Reuse across layers requires that generation; a changed row expansion selects a matching consumer/layout but does not inherently require recomputing the request partition.
- Precondition `pre_resources` (implementation): `p.cluster_launch_supports_selected_width_and_fixed_capacity_grid and p.compiled_combined_resource_budget_matches_launch and p.row_scratch_outputs_private_or_properly_owned` — Physical launch cluster width/grid and the compiled combined branch resource allocation are legal; per-row scratch/output ownership prevents cross-row races.
- Precondition `pre_lifetime` (implementation): `p.every_active_row_output_has_one_body_owner and p.all_metadata_and_body_readers_retire_before_reprepare_or_reuse` — Every row is processed exactly once at the body level, and prepared tables/scratch stay alive until all readers retire.
- Constraint `c_requests` (implementation): `p.live_requests >= 0 and p.live_requests <= max_requests and p.request_row_grid_and_index_products_fit_int32` — Live requests fit prepared capacity and all expanded indices/counts fit native arithmetic.

## Local example and misuse

For a local instance of `work_assign`, identify the object `cluster_cooperative_or_independent_row_packing` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/cluster_granularity_switch@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/cluster_granularity_switch@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Stable request permutation, two class counts and body-private row scratch.. Structured effects describe work and resource changes, not timings.

```yaml
transactions:
- id: fixed_capacity_ctas
  pattern: launch
  kind: cta_launch
  count: max_requests * rows_per_request * cluster_ctas
storage:
- id: request_order
  level: dram
  scope: kernel
  bytes: 4 * max_requests
- id: class_counts
  level: dram
  scope: kernel
  bytes: '8'
overlap: []
```

## Obligations and numerical checks

- `row_bijection` (exhaustive): Stable request partition and both expansion formulas cover every live semantic row exactly once without overlap, including nondivisible short tails.
- `collective_domains` (test): Every cooperative cluster has all ranks, while independent/tail branches introduce no cross-CTA wait or remote lifetime dependency.
- `prepared_generation` (test): Counts/order/key units/request population remain one published generation; each consumer binds the correct row expansion and tensor layout.
- `body_equivalence_and_resources` (test): Selected bodies satisfy the same row contract under their own numerical/tie semantics and fit the combined native launch allocation.

Both row bodies must meet the requested numerical/tie contract. Mismatched population/counts or consumer row expansion can omit or misaddress rows; stale scan keys can choose the wrong execution granularity.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
