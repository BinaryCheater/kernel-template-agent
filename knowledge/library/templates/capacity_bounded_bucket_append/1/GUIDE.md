# capacity_bounded_bucket_append@1: capacity bounded bucket append

## When to consider

When grouping points into bounded voxels, separate bucket identity assignment from point reservation, and expose the retained extent rather than the unsaturated reservation counter. Three points in one bucket with capacity2 reserve0,1,2: two payloads exist and count must clamp3 to2. The third point can still report that bucket ID; it is not a retained-slot identifier.

## Computation before and after

When grouping points into bounded voxels, separate bucket identity assignment from point reservation, and expose the retained extent rather than the unsaturated reservation counter.

First hash each valid discretized point coordinate. Enumerate occupied table entries; atomically reserve a bucket ID and materialize coordinates only for IDs<bucket_capacity. For each point whose bucket ID fits, atomically fetch-add its point count; write features only when old_count<items_per_bucket. After append completes, clamp each visible count to items_per_bucket. Optionally fill remaining initialized padding with the mean of retained points, using the clamped count. Point-to-bucket IDs name the assigned bucket even when that point lost the per-bucket reservation.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| bucket_capacity | compile / software | {"lo": 1, "hi": 2147483647} | Maximum retained buckets. |
| items_per_bucket | compile / software | {"lo": 1, "hi": 2147483647} | Maximum stored points in each retained bucket. |

## Coupled constraints and scope

Generated NVIDIA point-to-voxel CUDA, arbitrary-order bounded output. Spatial floor transform and key hash are separate bindings. Empty mean-fill buckets are excluded because assigned buckets arise from at least one valid point. Resource effects: One hash record per distinct valid coordinate, two reservation phases, bounded bucket_capacity*items_per_bucket feature storage; raw counters can exceed capacity until clamp.

Scope predicate: `p.bucket_keys_valid and p.hash_deduplicates_keys and p.unordered_subset_admitted and p.counters_initialized and p.phases_ordered and p.counter_no_overflow`.

- Precondition `pre_0` (semantic): `p.bucket_keys_valid and p.hash_deduplicates_keys` — Each valid point has one discretized bucket key and the table contains one entry per key.
- Precondition `pre_1` (semantic): `p.unordered_subset_admitted and p.counters_initialized` — Caller accepts arbitrary retained buckets/points on overflow; counters start at zero.
- Precondition `pre_2` (semantic): `p.phases_ordered and p.counter_no_overflow` — Hash build, bucket assignment, append and final clamp are ordered, and reservation counters do not wrap.

## Local example and misuse

For a local instance of `map`, identify the object `capacity_bounded_bucket_append` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/spconv/delta/capacity_bounded_bucket_append@1/template.yaml`. Original outline: `knowledge/studies/spconv/delta/capacity_bounded_bucket_append@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One hash record per distinct valid coordinate, two reservation phases, bounded bucket_capacity*items_per_bucket feature storage; raw counters can exceed capacity until clamp.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `bounds` (proof): No rejected bucket or point reservation writes outside retained storage.
- `retained_extent` (proof): Visible count equals number of initialized payload rows, not number of attempted reservations.
- `subset_contract` (proof): Every stored point belongs to its bucket; overflow output is an admitted schedule-dependent subset, not stable input order.
- `phase_lifetime` (proof): All writers finish before clamp/mean fill or consumers inspect count and payload.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
