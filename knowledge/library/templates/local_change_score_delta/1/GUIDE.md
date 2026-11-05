# local_change_score_delta@1: local change score delta

## When to consider

When an objective sums independent local group scores and a candidate changes only a few groups, score only changed groups and subtract their original contributions.

## Computation before and after

When an objective sums independent local group scores and a candidate changes only a few groups, score only changed groups and subtract their original contributions.

Let J=sum_g f(group_g). For a candidate permutation affecting groups S, compute delta=sum_(g in S)(f(new_group_g)-f(old_group_g)); unchanged groups cancel. Source2:4 sparsity uses f of four values = sum of two largest absolute magnitudes, computed as maximum of six pair sums. Swapping one column from each of two stripes changes only those stripes; enumerate16 swaps, restore original stripe values after each candidate, sum deltas over rows. Accumulate into initialized per-candidate outputs with disjoint or complete reduction ownership. This chooses a grouping permutation, not a lossless pruning of arbitrary dense weights.

Recover the original computation: Additive local quality objectives, source unweighted absolute-magnitude2:4 retention score. Objective equivalence does not claim task accuracy or unbiased training.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| changed_groups | compile / software | {"lo": 1, "hi": 1024} | S size: source2 groups for a column swap. |

## Coupled constraints and scope

Additive local quality objectives, source unweighted absolute-magnitude2:4 retention score. Objective equivalence does not claim task accuracy or unbiased training.

Scope predicate: `(p.objective_additively_separable and p.unchanged_groups_identical and p.candidate_state_restored and p.reduction_rounding_admitted)`.

- Precondition `pre0` (semantic): `p.objective_additively_separable and p.unchanged_groups_identical and p.candidate_state_restored and p.reduction_rounding_admitted` — Objective decomposes by groups; candidate evaluation has no state leakage.

## Local example and misuse

For a local instance of `reduce`, identify the object `local_change_score_delta` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/apex/delta/local_change_score_delta@1/template.yaml`. Original outline: `knowledge/studies/apex/delta/local_change_score_delta@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `cancellation` (proof): Every unchanged group contribution cancels from J_new-J_old.
- `candidate_isolation` (proof): Each trial sees original groups except its own declared local change.
- `reduction_complete` (proof): Row contributions appear exactly once in each candidate score.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
