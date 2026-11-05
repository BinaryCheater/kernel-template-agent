# centered_moment_merge@1: mergeable centered statistics

## When to consider

When a row statistic spans several warps or CTAs, merge count, mean and centered squared deviation instead of materializing all values or subtracting large raw second moments. Counts may be omitted from transport only when equal partition sizes are known. Preserve the count and normalization convention; floating-point reassociation remains part of the numerical policy.

## Computation before and after

Merge centered moments by count-weighted mean and between-partition correction.

A partial over samples A carries n_A, mean mu_A and M2_A=sum_(x in A)(x-mu_A)^2. Merge disjoint nonempty A and B with n=n_A+n_B, delta=mu_A-mu_B, mu=(n_A*mu_A+n_B*mu_B)/n and M2=M2_A+M2_B+delta^2*n_A*n_B/n. Return the other operand for a zero-count identity and guard the zero/zero case. At the end population variance is M2/n; RMS second moment is M2/n+mu^2. With equal_inferred, transport only mu,M2 and recover each level count from its exact partition shape. Local partial statistics may use separate mean and centered-deviation passes; merging them does not require an online per-sample Welford update.

Recover the original computation: Replace every merged partial by the union of its disjoint samples; the state summarizes the same global count, mean and centered squared deviation.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| count_mode | compile / software | ["explicit", "equal_inferred"] | Carry each partition count or reconstruct equal nonempty counts from the partition shape. |

## Coupled constraints and scope

Scalar finite centered moments with exact nonnegative sample counts and population normalization. TransformerEngine tuned norm paths compute mean/M2 per warp, merge equal-size warp/CTA partials with inferred counts and store FP32 mean/rsigma. Uneven tails need explicit counts or a separately justified padding policy. Exact floating-point associativity, automatic stability bounds and arbitrary weighted sample measures are outside this version.

Scope predicate: `(count_mode != 'equal_inferred' or p.equal_nonempty_partition_counts) and p.disjoint_complete_samples and p.partial_counts_match and p.partial_centered_stats_valid and p.zero_count_identity_guarded and p.nonempty_final_domain and p.finite_sample_domain and p.moment_merge_rounding_admitted and p.statistics_visible_before_merge`.

- Precondition `pre_partition` (semantic): `p.disjoint_complete_samples and p.partial_counts_match and p.partial_centered_stats_valid` — Each partial summarizes its exact sample set as count, mean and sum of squared deviations.
- Precondition `pre_empty` (semantic): `p.zero_count_identity_guarded and p.nonempty_final_domain` — Empty partials are explicit identities and no undefined zero-total merge is evaluated.
- Precondition `pre_numeric` (implementation): `p.finite_sample_domain and p.moment_merge_rounding_admitted and p.statistics_visible_before_merge` — Finite arithmetic, changed merge order and partial publication satisfy the numerical and visibility contract.
- Constraint `c_counts` (semantic): `count_mode != 'equal_inferred' or p.equal_nonempty_partition_counts` — Omitted counts are justified only by the actual partition cardinalities.

## Local example and misuse

For a local instance of `reduce`, identify the object `mergeable_centered_statistics` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/transformerengine/delta/centered_moment_merge@1/template.yaml`. Original outline: `knowledge/studies/transformerengine/delta/centered_moment_merge@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Count/mean/M2 state; equal partitions may communicate only mean/M2.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `moment_identity` (proof): The merge equations give the count, mean and centered second moment of the disjoint union.
- `count_inference` (test): Implicit counts match every partition including tails and inactive lanes.
- `numeric_and_visibility` (test): Native arithmetic and publication match the admitted partial-state merge.

Avoiding raw E[x²]-E[x]² does not remove floating-point error; count representation, merge tree and epsilon/final normalization remain numerical choices.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
