# enumerated_target_mass_sampling@1: partial categorical enumeration

## When to consider

When speculative candidates already identify useful target tokens, assign them their actual target-probability intervals before invoking a full categorical scan. If none wins, draw from the complete untested target mass; preserve distinct IDs, fresh exclusion state and matched tree histories.

## Computation before and after

Enumerate useful target-mass intervals, then sample the untested target residual.

For full_cdf sample from the complete target categorical row. In candidate_intervals_then_residual, draw u in[0,1) and traverse distinct candidate IDs in fixed order, accumulating their target probabilities. Accept the first positive-probability candidate whose cumulative interval exceeds u; no full vocabulary scan is needed on that path. If all tested intervals fail, sample independently from target probabilities with exactly those tested coordinates removed, normalized by the complete remaining mass. For a candidate tree, after acceptance move to the matching child-history target row and use its fresh test uniform; repeat until no useful child or output budget remains. Emit one residual/bonus token at the last matched history. The residual workspace is an exclusion mask times target mass, not the generating draft Q.

Recover the original computation: A tested candidate i owns an interval of width P_i, so its direct emitted mass is P_i. The failure event has mass1-sum_tested(P); its independent normalized remaining-target sample contributes P_j for every untested j. Thus each history samples exactly the target categorical law under the stated ideal/native contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| selection | compile / software | ["full_cdf", "candidate_intervals_then_residual"] | Direct complete categorical CDF or the same target law via an ordered candidate interval prefix and an independent residual draw. |

## Coupled constraints and scope

Local TreeSpeculativeSamplingTargetOnly in jit/include/sgl_kernel/speculative/sampling.cuh:39-209 with threshold_acc1,threshold_single1, distinct sibling token IDs and fresh zero draft_probs scratch. The source wrapper admits other thresholds, which are outside this exact-law binding. Independent draft probabilities are not used by this algorithm. Source vector loads, CUB reduction/scan, uniform control stores and zero-mass fallback retain native realization obligations.

Scope predicate: `(p.target_finite_nonnegative_normalized and p.tested_token_ids_distinct_and_valid) and (p.fresh_test_and_fallback_uniforms_at_each_matched_history and p.tree_row_matches_selected_prefix) and (p.cumulative_threshold_scale == 1 and p.single_token_forcing_disabled_except_probability_one) and (p.residual_workspace_initially_zero and p.only_tested_rejected_coordinates_removed and p.fallback_support_complete) and (p.native_prefix_sum_comparisons_cdf_endpoints_and_zero_residual_policy_admitted and p.output_generation_complete_before_read)`.

- Precondition `pre_target` (semantic): `p.target_finite_nonnegative_normalized and p.tested_token_ids_distinct_and_valid` — One declared target categorical row; ordered candidate IDs are distinct support coordinates.
- Precondition `pre_uniform` (implementation): `p.fresh_test_and_fallback_uniforms_at_each_matched_history and p.tree_row_matches_selected_prefix` — Initial test uniform and independent fallback uniform have the declared conditional law and token-history identity.
- Precondition `pre_exact` (semantic): `p.cumulative_threshold_scale == 1 and p.single_token_forcing_disabled_except_probability_one` — Threshold scaling or forced acceptance cannot alter target mass; source exact branch uses threshold_acc1,threshold_single1.
- Precondition `pre_residual` (implementation): `p.residual_workspace_initially_zero and p.only_tested_rejected_coordinates_removed and p.fallback_support_complete` — Residual subtracts exactly the masses of tested rejected IDs, once, from the same target row; no unrelated draft distribution is substituted.
- Precondition `pre_native` (implementation): `p.native_prefix_sum_comparisons_cdf_endpoints_and_zero_residual_policy_admitted and p.output_generation_complete_before_read` — Finite comparisons/sums and final CDF have a matching endpoint/fallback policy, with coherent scratch and output visibility.

## Local example and misuse

For a local instance of `reduce`, identify the object `partial_categorical_enumeration` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/enumerated_target_mass_sampling@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/enumerated_target_mass_sampling@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Tree navigation, target rows and fresh per-history exclusion scratch.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `single_row_mass` (proof): Candidate intervals and conditional fallback partition the same target categorical mass without duplication.
- `matched_tree_history` (test): Every accepted child selects the target row for that emitted prefix with fresh conditional random identity.
- `scratch_and_endpoints` (test): Fresh exclusions, unique IDs, numerical endpoints and complete fallback implement the declared native categorical contract.

Threshold scales below1 or forced single-token acceptance alter target mass. Duplicate sibling IDs double-count it. Stale exclusion scratch or finite endpoint fallback can invalidate the exact conditional distribution.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
