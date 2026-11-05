# positive_difference_log_encoding@1: encoded positive probability difference

## When to consider

When a consumer accepts log weights, encode positive P-Q residuals without first materializing tiny linear values. Keep support and subtraction orientation explicit; omit a normalization only for a consumer invariant to common log shifts.

## Computation before and after

Encode positive difference as logP+log1p(-Q/P), masking nonpositive support.

For r=max(P-Q,0), linear encoding forms the admitted positive difference. Log encoding returns -inf when P=0 or Q>=P; when P>Q, compute log_r=logP+log1p(-exp(logQ-logP)), with logQ=-inf giving log_r=logP. A log-weight consumer may use log_r directly, since exponentiating it recovers r in exact arithmetic. Common residual normalization is a shared additive log shift and may be omitted only when the downstream operation is invariant to it. Preserve a separate explicit zero-total-mass policy.

Recover the original computation: Decode a linear record directly or exp(log_r), with exp(-inf)=0; each logical value is max(P-Q,0) under its admitted numerical recipe.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| encoding | compile / software | ["linear", "log"] | Requested residual representation consumed downstream; log0 is represented by negative infinity. |

## Coupled constraints and scope

DSpark blockwise_rejection.py:292-357 uses normalized target/draft logits and log1p(-exp(logq-logp)), then a log-weight race; one-hot draft excludes the rejected token directly. Classic reject_sampling.py uses linear positive difference plus full mass/CDF. This representation step does not establish draft provenance, correction-law validity or native near-equal support preservation.

Scope predicate: `p.target_and_draft_logweights_exactly_identify_declared_probabilities and p.log_inputs_finite_or_negative_infinity and p.consumer_decodes_selected_encoding_and_preserves_zero_support and p.target_minus_draft_orientation_fixed and p.exp_ratio_log1p_rounding_and_near_equal_support_policy_admitted`.

- Precondition `pre_input` (semantic): `p.target_and_draft_logweights_exactly_identify_declared_probabilities and p.log_inputs_finite_or_negative_infinity` — Log inputs describe the same correctly normalized target/draft probabilities at the same history, or a declared common measure; no NaNs/+inf.
- Precondition `pre_consumer` (semantic): `p.consumer_decodes_selected_encoding_and_preserves_zero_support and p.target_minus_draft_orientation_fixed` — A downstream sampler/contraction understands the selected representation; the subtraction is P-Q, not reversed.
- Precondition `pre_numeric` (implementation): `p.exp_ratio_log1p_rounding_and_near_equal_support_policy_admitted` — The selected native recipe, near-equal rounding and zeros are admitted; exact algebra alone does not certify support after rounding.

## Local example and misuse

For a local instance of `map`, identify the object `encoded_positive_probability_difference` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/positive_difference_log_encoding@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/positive_difference_log_encoding@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One linear or log scalar per residual coordinate; direct fused consumption optional.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `positive_difference` (proof): Decoded values and their zero-support set equal the declared positive P-Q residual.
- `numerical_support` (test): Native exp/log1p and cancellation satisfy the admitted near-equal, underflow and zero-total-mass policy.

FP32 exp(logQ-logP) can round to1 even when P>Q, erasing a tiny positive residual; this source recipe needs an admitted support policy. Log encoding does not automatically cure near-equality cancellation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
