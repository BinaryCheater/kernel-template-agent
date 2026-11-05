# sparse_lane_rank_reduction@1: sparse lane rank reduction

## When to consider

When predicate partitioning leaves sparse warp members, reduce over set-bit rank instead of physical neighbors. For lanes {1,4,9}, logical ranks0,1,2 must combine A,B,C in that order. A three-member group needs no padding lane to contribute. Bind first-member or all-member result policy explicitly.

## Computation before and after

A masked warp reduction communicates by logical member rank rather than physical XOR lane distance.

For physical lane l in mask M define rank=popcount(M & lanes_below(l)), n=popcount(M). First reduce each lane local items in order. At strides 1,2,4,...<next_power_of_two(n), fetch the current value of member rank+stride through the masked rank-to-lane shuffle; combine only if that member exists. First member obtains the complete ordered reduction. In all_members mode broadcast from the least physical set bit of M. Do not use physical lane+stride or an XOR mask for a sparse group.

Recover the original computation: Erase masked shuffles; the first member or every member receives the ordered reduction of all original members in increasing logical rank.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| result_to | compile / software | ["first_member", "all_members"] | Expose only the first logical member or broadcast the completed value. |

## Coupled constraints and scope

Nonempty subsets of one NVIDIA warp, including noncontiguous lanes and non-power-of-two member counts. Associativity is required, commutativity is not; this is not cross_lane_reduction@1 fixed-width XOR butterfly.

Scope predicate: `(p.member_mask_nonempty_and_same_for_all_members and p.logical_rank_ascending_physical_lanes) and (p.operator_associative_under_contract and p.every_member_converges)`.

- Precondition `pre0` (semantic): `p.member_mask_nonempty_and_same_for_all_members and p.logical_rank_ascending_physical_lanes` — Participating lanes share one mask and rank by set-bit order.
- Precondition `pre1` (semantic): `p.operator_associative_under_contract and p.every_member_converges` — Ordered grouping is legal and all mask members execute each shuffle.

## Local example and misuse

For a local instance of `reduce`, identify the object `sparse_lane_rank_reduction` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cccl/delta/sparse_lane_rank_reduction@1/template.yaml`. Original outline: `knowledge/studies/cccl/delta/sparse_lane_rank_reduction@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `rank_interval` (proof): After stride s each rank summarizes its ordered suffix interval of at most 2*s members; the first rank eventually covers all members.
- `masked_shuffle` (test): Rank-to-physical-lane mapping and convergence preserve group membership.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
