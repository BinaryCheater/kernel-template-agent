# donor_seeded_centroid_reinitialization@1: donor seeded centroid reinitialization

## When to consider

When clustering leaves empty or undersized groups, reseed their centers near selected donor clusters and rerun assignment instead of treating a center change as moving existing labels. With donor center10,point14,offset0.1,receiver becomes10.4. Its previous count remains stale until prediction. Declaring it balanced merely from the new center is invalid.

## Computation before and after

When clustering leaves empty or undersized groups, reseed their centers near selected donor clusters and rerun assignment instead of treating a center change as moving existing labels.

Identify undersized receiver clusters from average population times the lower tolerance. Either search a donor point from a cluster at least average size or pair sorted-small receivers with sorted-large donors under the upper/lower threshold policy. Set receiver center=donor_center+offset*(donor_point-donor_center). Do not alter labels/counts in the reseed kernel; the next prediction and centroid recomputation establish the new membership. An unsuccessful finite donor search leaves that receiver unchanged.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| donor_selection | compile / software | ["random_above_average", "sorted_extreme_pair"] | Source donor policy; upper tolerance applies to paired selection, while random path chooses at-least-average clusters. |

## Coupled constraints and scope

NVIDIA balanced KMeans reseed heuristic, including random above-average and sorted-extreme donor policies. Lower/upper tolerances and offset affect cluster quality; they do not establish a hard per-cluster capacity invariant. Resource effects: One warp per receiver/pair reads a donor point and writes a center; adds reassign/recompute iterations while avoiding full data repartition inside the reseed kernel.

Scope predicate: `p.labels_and_sizes_same_generation and p.rebalance_policy_defined and p.receivers_unique_and_donors_readable and p.data_finite and p.reassignment_after_reseed and p.finite_iteration_budget`.

- Precondition `pre_0` (semantic): `p.labels_and_sizes_same_generation and p.rebalance_policy_defined` — Cluster sizes match current labels; lower/upper tolerance, donor selection and perturbation offset are explicit.
- Precondition `pre_1` (semantic): `p.receivers_unique_and_donors_readable and p.data_finite` — Each receiver has one writer and donor center/point reads remain valid during the reseed phase.
- Precondition `pre_2` (semantic): `p.reassignment_after_reseed and p.finite_iteration_budget` — Fresh prediction recomputes labels before claiming new balance, within a finite algorithm budget.

## Local example and misuse

For a local instance of `map`, identify the object `donor_seeded_centroid_reinitialization` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/donor_seeded_centroid_reinitialization@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/donor_seeded_centroid_reinitialization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One warp per receiver/pair reads a donor point and writes a center; adds reassign/recompute iterations while avoiding full data repartition inside the reseed kernel.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `generation` (proof): Selection counts and donor labels describe the same assignment generation.
- `write_ownership` (proof): One receiver update does not race a read of its center as a donor in the selected policy.
- `membership` (proof): Center reseeding is never reported as an immediate transfer of assigned points.
- `termination` (proof): Finite donor attempts and clustering iteration cap terminate without a strict-balance guarantee.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
