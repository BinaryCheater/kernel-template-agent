# lookahead_prefix_group_decode@1: lookahead prefix group decode

## When to consider

When work counts derive from adjacent prefix boundaries, reserve one subgroup lane for lookahead so a single cooperative load supplies both ends of each group interval.

## Computation before and after

When work counts derive from adjacent prefix boundaries, reserve one subgroup lane for lookahead so a single cooperative load supplies both ends of each group interval.

For W lanes load prefix boundary P[b+lane] for valid boundary indices. Lane l<W-1 gets count ceil((P[b+l+1]-P[b+l])/tile_extent)*other_tiles; last lane contributes0. Inclusive scan counts, ballot interval-end<=query to find the first containing group, then broadcast group start/count. Advance next window by W-1 groups, retaining the shared boundary. Cache matched group/start/end only while queries remain within that half-open interval. Zero-length groups have count0 and never capture valid work. Past-last queries return invalid before consuming metadata.

Recover the original computation: Subgroup lookahead inversion of a concatenation of tiled ragged groups. Source32 lanes handle31 sequence lengths; this differs from loading32 independent group-count tuples.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| lanes | compile / software | {"lo": 2, "hi": 32} | W: source32 lanes,31 groups per window. |

## Coupled constraints and scope

Subgroup lookahead inversion of a concatenation of tiled ragged groups. Source32 lanes handle31 sequence lengths; this differs from loading32 independent group-count tuples.

Scope predicate: `(p.monotone_boundaries and p.nonnegative_monotone_queries and p.prefix_counts_fit and p.metadata_immutable and p.tile_extent_positive)`.

- Precondition `pre0` (semantic): `p.monotone_boundaries and p.nonnegative_monotone_queries and p.prefix_counts_fit and p.metadata_immutable and p.tile_extent_positive` — Valid prefix sequence and monotone queries preserve cached-window validity.

## Local example and misuse

For a local instance of `work_assign`, identify the object `lookahead_prefix_group_decode` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/quack/delta/lookahead_prefix_group_decode@1/template.yaml`. Original outline: `knowledge/studies/quack/delta/lookahead_prefix_group_decode@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `lookahead` (proof): Each boundary difference uses the adjacent boundary and the next window overlaps exactly one boundary.
- `inverse_prefix` (proof): First interval end>query owns query uniquely, skipping zero-count groups.
- `cache` (proof): Cached interval can serve a query only while start<=query<end.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
