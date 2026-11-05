# group_aligned_vector_partials@1: group aligned vector partials

## When to consider

When vector loads cross reduction-group boundaries, split each vector into gcd(vector_width,group_width) microblocks so local partials never mix groups.

## Computation before and after

When vector loads cross reduction-group boundaries, split each vector into gcd(vector_width,group_width) microblocks so local partials never mix groups.

For aligned contiguous channel groups of width G and vectors of width V, set d=gcd(G,V). Since every group start and vector start is a multiple of d, d-element microblocks lie within one group. Form V/d independent partial states per loaded vector, then route each by its logical channel/group id to the group reduction. Source V4,G10 gives d2 and two partial states per vector, rather than one sum of4 that could cross a group boundary. Spatial-row and channel blocks subsequently combine states for the same group. Reload inputs after statistics or retain fragments; that storage choice does not change membership. A swizzled scratch layout must preserve the microblock/group coordinate mapping.

Recover the original computation: Contiguous equally sized channel groups and aligned vectors. gcd determines a safe local aggregation unit; backend tail and scratch layouts remain part of the realization.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| vector_width | compile / software | {"lo": 1, "hi": 32} | V: selected load width. |
| group_width | compile / software | {"lo": 1, "hi": 4096} | G: workload channels/group. |

## Coupled constraints and scope

Contiguous equally sized channel groups and aligned vectors. gcd determines a safe local aggregation unit; backend tail and scratch layouts remain part of the realization.

Scope predicate: `(p.group_and_vector_origins_aligned and p.group_membership_by_channel and p.partial_route_preserves_group and p.combine_complete)`.

- Precondition `pre0` (semantic): `p.group_and_vector_origins_aligned and p.group_membership_by_channel and p.partial_route_preserves_group and p.combine_complete` — Aligned groups/vector origins, same-group combine and complete participation.

## Local example and misuse

For a local instance of `reduce`, identify the object `group_aligned_vector_partials` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/apex/delta/group_aligned_vector_partials@1/template.yaml`. Original outline: `knowledge/studies/apex/delta/group_aligned_vector_partials@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `no_cross_group` (proof): All boundaries are multiples of d, so no d-element microblock crosses a group boundary.
- `coverage` (proof): Each channel occurs in exactly one vector microblock and one final group state.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
