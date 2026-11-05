# certified_distance_refinement_filter@1: certified distance refinement filter

## When to consider

When candidate distances are expensive, reject candidates from a conservative lower bound against a threshold certified by k retained complete scores, and refine only survivors. Partial L2 sum7 with remaining nonnegative terms may be dropped against kth6; partial inner product7 could receive term-5 and become2, so the same test is invalid. An estimate10 with error3 has bound7; using estimate10 itself can incorrectly reject a true distance8 against threshold9.

## Computation before and after

When candidate distances are expensive, reject candidates from a conservative lower bound against a threshold certified by k retained complete scores, and refine only survivors.

For each candidate compute a lower bound L. If L>threshold discard it; discard L==threshold only when the declared equality/tie policy permits; otherwise evaluate more terms or a finer code representation and repeat, finally offering the complete score to top-k. Nonnegative partial squared-distance sums are lower bounds when all remaining terms and rounded accumulation preserve monotonicity. Signed inner-product sums are not, so disable early stopping unless a separate residual bound exists. For a short-code estimate with error radius E, use estimate-E only when E is proved conservative for the exact refinement score under all admitted rounding. For a fixed-radius predicate use its declared cutoff and the requested equality policy, without fabricating k retained candidates.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| terms_per_check | compile / software | {"lo": 1, "hi": 65536} | Score terms or packed chunks evaluated between lower-bound checks; choose within realization resource constraints. |

## Coupled constraints and scope

NVIDIA IVF-PQ partial nonnegative L2 sum and optional multi-stage quantized refinement. The generic rule applies to exact top-k of the declared final score; approximate error envelopes without proof remain hypotheses, not certified instances. Resource effects: Can reduce remaining lookup/decode work per rejected candidate; adds bound checks and survivor staging. Check frequency trades branch overhead against wasted late terms; no measured optimum claimed. Fixed-radius search may bind the threshold to its workload radius instead of a kth score; that branch has no top-k count obligation.

Scope predicate: `p.lower_bound_valid_for_refined_score and (p.threshold_certified_by_k or p.threshold_fixed_by_workload) and p.min_distance_order and p.threshold_equality_policy_defined and p.refinement_preserves_identity and p.survivor_capacity_sufficient`.

- Precondition `pre_0` (semantic): `p.lower_bound_valid_for_refined_score and (p.threshold_certified_by_k or p.threshold_fixed_by_workload)` — Bound is conservative and threshold is justified either by k complete retained scores or by the declared fixed-radius predicate.
- Precondition `pre_1` (semantic): `p.min_distance_order and p.threshold_equality_policy_defined` — Smaller is better; equality is discarded only when the workload/tie contract permits it.
- Precondition `pre_2` (semantic): `p.refinement_preserves_identity and p.survivor_capacity_sufficient` — Retained candidates keep original IDs and survivor storage cannot truncate permitted winners.

## Local example and misuse

For a local instance of `map`, identify the object `certified_distance_refinement_filter` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/certified_distance_refinement_filter@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/certified_distance_refinement_filter@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Can reduce remaining lookup/decode work per rejected candidate; adds bound checks and survivor staging. Check frequency trades branch overhead against wasted late terms; no measured optimum claimed.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `bound_soundness` (proof): No rejected candidate has a final score below L, including quantization and table rounding.
- `threshold` (proof): Each threshold is either a fixed workload cutoff or an upper bound justified by k complete candidates; partial estimates alone cannot tighten it.
- `capacity` (proof): Every survivor is retained until refinement/selection completes.
- `ties` (proof): Strict versus nonstrict bound comparison matches inclusive-radius or top-k tie/identity policy.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
