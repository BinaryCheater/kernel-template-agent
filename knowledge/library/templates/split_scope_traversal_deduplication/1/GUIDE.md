# split_scope_traversal_deduplication@1: split scope traversal deduplication

## When to consider

When multiple CTAs explore one query graph, separate short-lived local candidate deduplication from query-wide atomic ownership of expanded parents. With desired internal topk=96 and search_width=2, num_cta=max(2,ceil(96/32))=3. Two CTAs discovering node 17 may retain local copies, but only one wins global parent insertion. Replacing the global table with per-CTA tables permits duplicate expansion.

## Computation before and after

When multiple CTAs explore one query graph, separate short-lived local candidate deduplication from query-wide atomic ownership of expanded parents.

Give each CTA a local visited table for candidate duplicates, reset/repopulate it as required by the current frontier. Give the query a separate global traversed table retained across iterations. Walk ranked unused candidates; atomically insert a parent ID into the global table. Only the successful inserter marks and expands that parent; a failed claimant invalidates its local copy and considers another. Merge per-CTA retained outputs after search. Local candidate revisit and globally unique parent expansion have distinct lifetimes.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| local_topk | compile / software | [32] | Fixed multi-CTA retained candidates per CTA in the source. |
| local_hash_bits | compile / software | {"lo": 8, "hi": 14} | Compiled/shared-memory hash capacity exponent, sized from frontier and degree. |
| global_hash_bits | compile / software | {"lo": 11, "hi": 25} | Query-wide traversed-table exponent; planner may impose a higher minimum. |

## Coupled constraints and scope

Approximate NVIDIA graph search: this preserves selected traversal ownership, not exact global nearest-neighbor recall or deterministic exploration order. Search width, retained size, iteration budget and random seeds change explored candidates and are separate algorithm parameters. Resource effects: Per-CTA shared local hash plus per-query global atomic hash and num_cta*32 intermediate results. Local reset saves persistent visitation storage at the cost of repeated inserts.

Scope predicate: `p.valid_graph and p.ids_fit_without_marker_bit and p.hash_capacity_sufficient and p.atomic_insert_linearizable and p.query_tables_disjoint and p.initialization_precedes_search`.

- Precondition `pre_0` (semantic): `p.valid_graph and p.ids_fit_without_marker_bit` — Graph neighbor IDs are valid and leave the high bit available as a used marker.
- Precondition `pre_1` (semantic): `p.hash_capacity_sufficient and p.atomic_insert_linearizable` — Hash insertion terminates and elects exactly one successful new owner per query/key.
- Precondition `pre_2` (semantic): `p.query_tables_disjoint and p.initialization_precedes_search` — Each query has its own initialized global table and local tables have CTA ownership.

## Local example and misuse

For a local instance of `map`, identify the object `split_scope_traversal_deduplication` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/split_scope_traversal_deduplication@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/split_scope_traversal_deduplication@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Per-CTA shared local hash plus per-query global atomic hash and num_cta*32 intermediate results. Local reset saves persistent visitation storage at the cost of repeated inserts.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `ownership` (proof): At most one CTA expands a given parent per query.
- `local_reset` (proof): Reset local visited state only after its users finish and rebuild retained entries before next insertion.
- `identity` (proof): Strip marker bits before dataset access and preserve invalid sentinel distinction.
- `termination` (proof): Finite iteration bounds and sufficient hash space prevent unbounded search/insertion.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
