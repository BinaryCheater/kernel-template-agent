# record_field_axis_storage@1: record field axis storage

## When to consider

When neighboring workers access the same field of independent records, choose field-major rather than record-major storage and rewrite every producer/consumer address consistently. RasterizeCSR.cuh:25-145 selects [batch][stateField][pixel], offset=(batch*D+field)*pixels+pixel. physical_storage@1 supplies occupancy/lifetime only, not axis order.

## Computation before and after

When neighboring workers access the same field of independent records, choose field-major rather than record-major storage and rewrite every producer/consumer address consistently.

Select record_major offset(r,f)=r*D+f or field_major offset(r,f)=f*N+r for N records and D fields. For batch b add b*N*D. Allocate N*D scalar slots, store and load every logical (b,r,f) through the same map. The maps are bijections and preserve values; if converting existing storage, copy through distinct storage or use a separately proved in-place permutation. Field-major places adjacent record IDs contiguously for a fixed field. It changes addresses, not mathematical record identity or reduction ordering.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| axis_order | compile / software | ["record_major", "field_major"] | Choose record-contiguous fields or field-contiguous record coordinates. |

## Coupled constraints and scope

Uniform scalar field arrays, not heterogeneous structs or bitpacked records. Layout change and actual transaction coalescing are separate: bind worker-to-record order and target transaction rules. Resource effects: Same scalar capacity N*D, different adjacency/transaction pattern.

Scope predicate: `p.record_count > 0 and p.field_count > 0 and p.uniform_scalar_field_width and p.all_producers_consumers_share_axis_map and p.all_live_indices_in_bounds and p.buffer_lifetime_and_capacity_valid`.

- Precondition `pre_0` (semantic): `p.record_count > 0 and p.field_count > 0 and p.uniform_scalar_field_width` — A rectangular record-by-field logical array with uniform scalar width.
- Precondition `pre_1` (semantic): `p.all_producers_consumers_share_axis_map and p.all_live_indices_in_bounds and p.buffer_lifetime_and_capacity_valid` — Changing layout is visible to every reader/writer and retained through deferred backward.

## Local example and misuse

For a local instance of `map`, identify the object `record_field_axis_storage` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/gsplat/delta/record_field_axis_storage@1/template.yaml`. Original outline: `knowledge/studies/gsplat/delta/record_field_axis_storage@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Same scalar capacity N*D, different adjacency/transaction pattern.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `bijection` (proof): Each layout maps all N*D logical coordinates onto unique offsets0..N*D-1.
- `consumer_identity` (proof): All complete payload fields and delayed readers use the selected map and generation.
- `materialization` (proof): Source buffer is retained until conversion readers finish; no unproved in-place overlap.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
