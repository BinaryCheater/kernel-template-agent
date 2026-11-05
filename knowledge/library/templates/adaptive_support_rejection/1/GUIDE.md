# adaptive_support_rejection@1: certified adaptive conditional sampler

## When to consider

When a threshold-defined categorical support is expensive to sort/materialize, repeatedly sample from a certified superset and prune rejected weight levels. Use a full-row count or mass certificate, keep all target weights in every candidate set, update its original mass and use fresh conditional randomness. This preserves the target-conditioned law while allowing sampled-only or sampled-plus-midpoint refinement; retain whole boundary ties unless another contract is supplied.

## Computation before and after

Adaptive rejection with support-preserving probability-threshold pruning.

Define C(t)=count_i(w_i>t) for above_count or sum_i(w_i>t) for above_mass, and limit=k or top_p. Target A contains positive-weight indices with C(w_i)<limit. Start low=0, high=1, q=1. Each round draw i categorically from S={j:w_j>low} with probabilities w_j/q, retaining original weights and q=sum_S(w). Scan the full original row to evaluate C(w_i). If it is below limit, return i. Otherwise reject i and choose t=w_i. With paired_midpoint also evaluate m=(w_i+high)/2 under the declared arithmetic: if C(m)<limit set high=m and retain t=w_i; otherwise set t=m. Set low=t and q=sum_{w>t}w, then draw again with fresh conditional randomness. Both cutoffs remove only rejected target classes. Never renormalize the row before computing the next full-row membership certificate, reuse the previous uniform without a proved conditional remapping, or return an uncertified last proposal after a round cap.

Recover the original computation: With exact fresh categorical proposals, erase rejection/pruning history: output i has probability w_i/sum_{j in A}w_j for i in A and zero otherwise. Each rejection removes a positive nontarget level, so at most the number of distinct rejected positive levels plus1 draws occur. With per-proposal total-variation error epsilon and unchanged exact certificates, a coupling/union bound gives at most width*epsilon output deviation.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| criterion | compile / boundary | ["above_count", "above_mass"] | Requested membership: original full-row count above the candidate is less than k, or original mass above it is less than top_p. |
| refinement | compile / software | ["sampled_threshold", "paired_midpoint"] | After rejection, prune at the sampled value alone or additionally evaluate its midpoint with the certified high threshold. Source uses paired_midpoint; sampled_threshold is an inferred simpler schedule. |

## Coupled constraints and scope

FlashInfer sampling.cuh TopKSamplingFromProbKernel and TopPSamplingFromProbKernel use a sampled pivot and midpoint, full-row count/mass certificates, updated surviving mass and fresh curand_uniform calls. Target ties are whole weight cohorts under strict-greater statistics, not exactly k stable IDs or a tie-split nucleus. Source native scan fallback, finite RNG grid, seed/offset state and floating-point mass classification remain realization obligations. Top-k-first/top-p composition, min-p and speculative decoding have separate support/acceptance contracts.

Scope predicate: `p.width >= 1 and (p.finite_nonnegative_weights and p.original_weight_sum_is_one and p.positive_target_support_nonempty) and (p.support_uses_full_row_strictly_greater_statistics and p.boundary_tie_cohorts_admitted) and (p.count_limit_between_one_and_width_or_mass_limit_in_open_zero_closed_one and p.initial_low_zero_high_one) and (p.full_row_certificates_match_requested_support_and_comparison_policy and p.rejected_threshold_removes_no_target_weight) and (p.each_proposal_uses_current_original_mass and p.fresh_conditional_categorical_draws_and_rng_domains_explicit) and (p.each_rejection_removes_at_least_the_sampled_weight_level and p.accepted_sample_is_only_success_exit) and (p.row_weights_limits_and_mapping_immutable_through_attempts and p.all_certificate_and_draw_participants_converged) and (p.cdf_rng_endpoint_fallback_and_rounding_policy_admitted and p.proposal_error_budget_explicit)`.

- Precondition `pre_weights` (implementation): `p.finite_nonnegative_weights and p.original_weight_sum_is_one and p.positive_target_support_nonempty` — One finite row has nonnegative normalized weights; target support contains positive mass. Zero-weight coordinates never enter the proposal set.
- Precondition `pre_target` (implementation): `p.support_uses_full_row_strictly_greater_statistics and p.boundary_tie_cohorts_admitted` — Membership is count{w>value}<k or sum{w>value}<top_p over the original row, including entire boundary-value cohorts; exact-cardinality ID tie breaking is a different contract.
- Precondition `pre_bounds` (implementation): `p.count_limit_between_one_and_width_or_mass_limit_in_open_zero_closed_one and p.initial_low_zero_high_one` — Use1<=k<=width or0<top_p<=1, initially low0/high1; positive target weights exceed low and high is a certified accepted threshold.
- Precondition `pre_certificates` (implementation): `p.full_row_certificates_match_requested_support_and_comparison_policy and p.rejected_threshold_removes_no_target_weight` — Counts/masses and strict comparisons certify the requested target membership. Each new low cutoff excludes only nontarget weights.
- Precondition `pre_draws` (implementation): `p.each_proposal_uses_current_original_mass and p.fresh_conditional_categorical_draws_and_rng_domains_explicit` — Every attempt samples original weights restricted to w>low, using that exact remaining mass and fresh independent conditional random draws; draw/output/invocation identities are explicit.
- Precondition `pre_progress` (implementation): `p.each_rejection_removes_at_least_the_sampled_weight_level and p.accepted_sample_is_only_success_exit` — Rejection raises the cutoff at least to the sampled weight, removing a nonempty level; successful exit returns only a membership-certified sample, without arbitrary truncation.
- Precondition `pre_generation` (implementation): `p.row_weights_limits_and_mapping_immutable_through_attempts and p.all_certificate_and_draw_participants_converged` — All rounds refer to the same row/threshold metadata, including output-to-data-row mapping; collective scans/reductions complete consistently.
- Precondition `pre_numeric` (implementation): `p.cdf_rng_endpoint_fallback_and_rounding_policy_admitted and p.proposal_error_budget_explicit` — Native CDF arithmetic, random endpoint grid, last-valid fallback and any per-draw deviation are declared. Exact categorical proposals give the exact restricted target law; otherwise retain a bound accumulated over the finite attempt count.
- Constraint `c_width` (semantic): `p.width >= 1` — At least one row coordinate.

## Local example and misuse

For a local instance of `contract`, identify the object `certified_adaptive_conditional_sampler` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/adaptive_support_rejection@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/adaptive_support_rejection@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Current low/high, surviving mass, sample and full-row certificate reductions; no explicit sorted support list required.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `support_invariant` (proof): Every proposal set retains the same complete target A with its original relative weights; every removed threshold class is certified outside A.
- `conditional_law` (proof): Fresh categorical proposals and immediate acceptance give the original weights conditioned on A despite adaptive rejection history.
- `finite_progress` (test): Each rejection removes at least one positive nontarget weight level, and no arbitrary cap/fallback returns an out-of-support result.
- `native_random_and_numeric` (test): RNG identity, remaining mass, CDF endpoints, tie comparisons and finite-precision error meet the requested sampling policy.

Finite CDF endpoints, inaccurate remaining mass, shared uniforms, rounded boundary certificates or a capped last-candidate return can alter the categorical law. Whole boundary cohorts can contain more than k entries.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
