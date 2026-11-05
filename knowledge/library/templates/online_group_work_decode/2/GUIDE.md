# online_group_work_decode@2: monotone group prefix cursor

## When to consider

When persistent workers query increasing indices across variable-size problems, consider retaining an inverse-prefix cursor instead of materializing per-tile work descriptors. Choose warp_prefix_32 for the old lane-window mapping, optionally with shape caching, or scalar_forward for one-count-at-a-time traversal. Counts and metadata must be immutable, prefixes must fit Int32, and queries cannot move backwards. The MoE scalar instance skips zero-output forward experts but retains backward K0 output tiles; it requires at least one expert because source initialization reads count0.

## Computation before and after

Stateful inverse prefix with warp32 windows or scalar forward traversal.

Retain the matched group and cumulative start/end. If the new nonnegative monotone query lies inside the interval, reuse it. Otherwise advance until an end exceeds the query or all groups are exhausted. With search=warp_prefix_32, scan32-lane windows padded with zero counts and optionally retain the two shape windows exactly as version1. With search=scalar_forward, inspect one count at a time, adding it to the end and updating the start; skip zero-count groups and return invalid after the last group. A scalar initialization must avoid reading group0 when the group count is zero. Invalid queries never consume group metadata. Physical tile decode and per-group count construction remain separate.

Recover the original computation: Enumerate all groups' work units in input order; each valid linear index returns the group and offset at that position in the concatenation.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| search | compile / software | ["warp_prefix_32", "scalar_forward"] | Choose32-lane prefix windows or a scalar forward scan; the latter retains only group/start/end. |
| cache_shapes | compile / software | [false, true] | Keep two per-lane windows of problem-shape tuples or load shapes on demand; prefix cursor state is retained in both paths. |

## Coupled constraints and scope

Version1 warp32 mapping is preserved by search=warp_prefix_32. scalar_forward supports immutable finite count tables and nonnegative monotone queries, without shape-window caching. The MoE instance computes expert cluster counts lazily, with a nonempty expert table because its source initializes group0 unconditionally. Forward zero-token experts have zero output work; backward zero-token experts retain fixed output work with K0. Physical coordinates, short-axis traversal, descriptor publication and reduction identity are separate obligations.

Scope predicate: `p.counts_nonnegative and p.count_table_complete and p.prefix_fits_int32 and p.query_indices_nonnegative and p.queries_monotone and p.problem_metadata_immutable`.

- Precondition `pre_order` (semantic): `p.query_indices_nonnegative and p.queries_monotone` — Each worker's new linear work index never moves behind its retained prefix.
- Precondition `pre_metadata` (semantic): `p.problem_metadata_immutable` — Counts and cached shapes stay valid throughout the worker's queries.
- Constraint `c_counts` (implementation): `p.counts_nonnegative and p.count_table_complete and p.prefix_fits_int32` — Counts describe every group and their cumulative total fits the pinned32-bit arithmetic.
- Constraint `c_cache` (implementation): `search == 'warp_prefix_32' or not cache_shapes` — The scalar implementation has no two-window per-lane shape cache.

## Local example and misuse

For a local instance of `work_assign`, identify the object `monotone_group_prefix_cursor` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/online_group_work_decode@2/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/online_group_work_decode@2/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

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
