# online_group_work_decode@1: monotone group prefix cursor

## When to consider

When a persistent worker receives increasing linear tile indices across differently sized problems, consider decoding their group online instead of materializing one descriptor per tile. Preserve the inverse of the immutable count prefix and bind cache_shapes only to metadata-load strategy; the cursor still requires monotone queries. The pinned implementation searches32 groups per window for L1 GEMMs, and counts[1,2,2] map index2 to group1/local1 even when group1 has K0; backwards queries or new batch/lane layouts need a variant.

## Computation before and after

Stateful inverse prefix over group work counts with monotone queries.

Retain the matched group and its cumulative start/end. A new index inside that half-open interval reuses the group. Otherwise scan groups forward in32-lane windows, padding out-of-range lanes with zero counts. Compute inclusive prefix ends and count lanes whose end is <= the query; the first larger end identifies its group and query-minus-group-start is its local work index. If no group owns the query, return invalid and do not consume its metadata. Optionally cache two shape windows, refreshing their group identities when advancing the cursor.

Recover the original computation: Enumerate all groups' work units in input order; each valid linear index returns the group and offset at that position in the concatenation.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| cache_shapes | compile / software | [false, true] | Keep two per-lane windows of problem-shape tuples or load shapes on demand; prefix cursor state is retained in both paths. |

## Coupled constraints and scope

Increasing per-worker queries into a finite concatenation of variable-size groups, with32-lane prefix/ballot search and optional two-window shape cache. The pinned grouped GEMM computes M/N cluster counts for L1 problems; K0 still owns output units. Spatial tile decode, descriptor mutation, MMA and output arithmetic are separate. A new query order, wider integer prefix, lane width or multi-batch group implementation needs an explicit variant or guarded binding.

Scope predicate: `p.counts_nonnegative and p.count_table_complete and p.prefix_fits_int32 and p.query_indices_nonnegative and p.queries_monotone and p.problem_metadata_immutable`.

- Precondition `pre_order` (semantic): `p.query_indices_nonnegative and p.queries_monotone` — Each worker's new linear work index never moves behind its retained prefix.
- Precondition `pre_metadata` (semantic): `p.problem_metadata_immutable` — Counts and cached shapes stay valid throughout the worker's queries.
- Constraint `c_counts` (implementation): `p.counts_nonnegative and p.count_table_complete and p.prefix_fits_int32` — Counts describe every group and their cumulative total fits the pinned32-bit arithmetic.

## Local example and misuse

For a local instance of `work_assign`, identify the object `monotone_group_prefix_cursor` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/online_group_work_decode@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/online_group_work_decode@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Prefix cursor and optional per-lane two-window shape cache.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: cached_shapes
  level: reg
  scope: thread
  bytes: 2 * p.shape_bytes
  when: cache_shapes
  note: Logical two-tuple payload; physical register allocation/spilling is unmeasured.
overlap: []
```

## Obligations and numerical checks

- `inverse_prefix` (exhaustive): Every valid query has exactly one owning half-open group interval and the correct local offset even with zero-count groups.
- `cache_identity` (test): Retained prefix and optional shape-cache entries refer to the same immutable group metadata as an independent full search.
- `bounds` (test): Invalid lanes and queries do not read or publish nonexistent group metadata; every window advance makes progress through zero-count groups.

No arithmetic-value change; counts and work ownership must remain exact.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
