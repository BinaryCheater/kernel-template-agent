# cached_sparse_row_segmented_join@1: cached sparse row segmented join

## When to consider

When sparse pairwise metrics repeatedly probe one source row, cache that row and stream coordinate-sorted nonzeros of the other matrix, reducing contributions by destination row. A has column2=value3, B row0 has col2=value4 and row1 has col5=value7. Product inner metric contributes12 to pair(A,0), zero to pair(A,1). Row0 split over chunks must combine its pieces atomically; relabeling a partial as row1 corrupts the pair matrix.

## Computation before and after

When sparse pairwise metrics repeatedly probe one source row, cache that row and stream coordinate-sorted nonzeros of the other matrix, reducing contributions by destination row.

Choose dense shared array initialized to zero or a bounded shared hash map for one A row. Insert its nonzeros and synchronize. Partition B nonzeros into disjoint chunks; each thread probes cached A at B column and applies the metric product. Maintain a partial accumulator for its B row; when row identity changes within a warp, group lanes by row and segment-reduce. One leader atomically combines each completed segment into pairwise output(Arow,Brow). For metrics depending on support union, separately run the reverse direction only on coordinates absent from the first cached support, so intersection is not counted twice.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| chunk_nonzeros | compile / software | {"lo": 1, "hi": 1048576} | B-side nonzeros processed per CTA chunk; source binds a per-thread chunk with block width. |
| cache_capacity | compile / software | {"lo": 1, "hi": 2147483647} | Dense feature slots or hash slots for the cached A row; representation selected by dimensionality and occupancy. |

## Coupled constraints and scope

NVIDIA sparse COO/CSR pairwise distance backend. Dense and hash shared caches have different resource bounds. rev tests cached value==0, so callers requiring structural support distinction must additionally rule out semantically different explicit zeros. Warp participation at tails requires realization verification. Resource effects: Caches one sparse row per CTA, streams B nonzeros with segmented reductions and atomic output updates; dense cache costs dim*sizeof(value), hash cache scales with occupied slots and may split long rows.

Scope predicate: `p.sparse_rows_unique_columns and p.coo_rows_ordered and p.cache_exact_and_capacity_valid and p.absence_zero_semantics and p.segment_combine_atomic_and_numeric_policy`.

- Precondition `pre_0` (semantic): `p.sparse_rows_unique_columns and p.coo_rows_ordered` — Each cached row has unique columns and streamed COO rows group their nonzeros in order.
- Precondition `pre_1` (semantic): `p.cache_exact_and_capacity_valid and p.absence_zero_semantics` — Cache returns exact present values and zero for absence; hash row splits preserve column coverage.
- Precondition `pre_2` (semantic): `p.segment_combine_atomic_and_numeric_policy` — Partial segments across warps/chunks combine atomically under the specified semiring/reassociation contract.

## Local example and misuse

For a local instance of `map`, identify the object `cached_sparse_row_segmented_join` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/cached_sparse_row_segmented_join@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/cached_sparse_row_segmented_join@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Caches one sparse row per CTA, streams B nonzeros with segmented reductions and atomic output updates; dense cache costs dim*sizeof(value), hash cache scales with occupied slots and may split long rows.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `cache` (proof): Every cached column has its source value; absent columns are zero without confusing explicit zeros and structural absence where the metric distinguishes them.
- `segments` (proof): Every streamed contribution belongs to its correct B row and exactly one emitted partial segment.
- `chunk_cover` (proof): Source hash splits and B chunks cover the intended support exactly once.
- `support_union` (proof): Forward/reverse metric passes include required intersection and symmetric-difference terms without duplication.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
