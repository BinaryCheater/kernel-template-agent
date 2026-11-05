# key_grouped_incidence_reindexing@1: grouped route permutation with inverse

## When to consider

When expert work needs contiguous token routes, group incidences by key and retain an inverse for every original route. Choose count/prefix scatter or sorted-key bounds; keep token multiplicity and route-specific metadata, and require stable order only when its consumer needs it.

## Computation before and after

Key-grouped incidence permutation with token gather and original-route inverse.

For each active original route r with token t(r) and key e(r), assign a distinct destination inside group e's interval. Count-prefix-scatter computes exact group counts and prefixes, then reserves a disjoint local cursor or stable rank for each route. Sorted-key-bounds finds lower_bound(key,e) boundaries and uses the sorting permutation to recover original route identity; subtract only the known inactive negative-prefix length. Write destination_to_token[d]=t(r), route_to_destination[r]=d, and retain any weights/scales by the same original route/token map. Inactive routes receive negative inverse entries. Preserve source_order only with a demonstrated stable implementation. Initialize safe unused gather capacity without increasing any group's live extent. Physical per-group padding and data/scale origins remain separate refinements.

Recover the original computation: For each active original route r, read its unique destination through route_to_destination and recover its original token and associated metadata. This restores every incidence once, including repeated token identities; inactive routes remain excluded.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| construction | compile / software | ["count_prefix_scatter", "sorted_key_bounds"] | Build counts/prefixes and scatter route records, or derive interval boundaries from already sorted keys and invert the original-ID permutation. |
| within_group_order | compile / boundary | ["arbitrary", "source_order"] | Whether consumers require source route order; this determines stable-sort/scatter obligations. |

## Coupled constraints and scope

cutlass_prepare_moe_input.cuh count/prefix/block-local atomic scatter uses construction=count_prefix_scatter,within_group_order=arbitrary, token=r/topk. moe_permute_prepare.cu uses sorted_key_bounds and original reorder IDs, excluding an inactive negative prefix. Atomic order is not a stable source-order guarantee. Separate aligned scale regions use segmented_aligned_storage@1; invalid positive keys are excluded from the sorted-bound source binding.

Scope predicate: `(p.route_ids_unique and p.token_identity_may_repeat and p.group_keys_valid_or_inactive) and (p.group_counts_exact and p.group_intervals_disjoint_and_complete and p.interval_capacity_sufficient) and (within_group_order != 'source_order' or p.group_emission_preserves_original_route_order) and (construction != 'sorted_key_bounds' or (p.sorted_keys_nondecreasing_and_reorder_ids_permutation and p.nonnegative_sorted_keys_below_group_count)) and (p.each_active_route_placed_once and p.inverse_route_map_matches_placement and p.route_auxiliary_metadata_matches_original_identity) and (p.inactive_inverse_entries_negative and p.unused_capacity_safe_and_not_published_live and p.maps_ready_before_consumers)`.

- Precondition `pre_records` (semantic): `p.route_ids_unique and p.token_identity_may_repeat and p.group_keys_valid_or_inactive` — Each incidence/route has one original identity even when several routes share a token; keys identify valid groups or inactive records.
- Precondition `pre_intervals` (implementation): `p.group_counts_exact and p.group_intervals_disjoint_and_complete and p.interval_capacity_sufficient` — Counts and offsets allocate one nonoverlapping interval for every active route; any independent padding is supplied by its own declared storage map.
- Precondition `pre_order` (implementation): `within_group_order != 'source_order' or p.group_emission_preserves_original_route_order` — An ordered request needs stable input sorting or a stable within-group scatter; atomic cursor order alone cannot satisfy it.
- Precondition `pre_sorted` (implementation): `construction != 'sorted_key_bounds' or (p.sorted_keys_nondecreasing_and_reorder_ids_permutation and p.nonnegative_sorted_keys_below_group_count)` — Sorted-bound construction has a complete original-ID permutation; negative records form only an inactive prefix, with no out-of-range positive IDs.
- Precondition `pre_maps` (semantic): `p.each_active_route_placed_once and p.inverse_route_map_matches_placement and p.route_auxiliary_metadata_matches_original_identity` — Destination→token and original-route→destination maps agree; weights and scales retain the same route/token generation.
- Precondition `pre_inactive` (implementation): `p.inactive_inverse_entries_negative and p.unused_capacity_safe_and_not_published_live and p.maps_ready_before_consumers` — Inactive routes never turn into live nonnegative addresses; full-capacity gathers have safe tail entries and readers wait for complete maps.

## Local example and misuse

For a local instance of `map`, identify the object `grouped_route_permutation_with_inverse` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/key_grouped_incidence_reindexing@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/key_grouped_incidence_reindexing@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Counts/offsets, one destination→token map and one route→destination map.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `incidence_bijection` (proof): Active original route IDs and packed live destinations are a bijection within their declared groups.
- `identity_and_order` (test): Token identity, auxiliary metadata and any required source order survive forward/inverse mapping.
- `inactive_capacity` (test): Inactive routes and safe gather tails cannot publish extra work or become live addresses.

Permutation alone changes no per-route values. Dropping duplicate token incidences, mismatching weights/scales or folding output routes in a different floating-point order changes results.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
