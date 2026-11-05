# weighted_centered_gram_covariance@1: weighted centered gram covariance

## When to consider

When full weighted covariance is needed and centered arithmetic is required, form square-root-weighted centered rows and compute their Gram matrix before normalization. Points0 and2, weights1 each, center1, denominator2 give Gram2/2=1. If denominator is2+epsilon, preserve that value. Reusing stale center0 gives Gram4 and a different covariance.

## Computation before and after

When full weighted covariance is needed and centered arithmetic is required, form square-root-weighted centered rows and compute their Gram matrix before normalization.

For each component k and sample i form Y[i,d]=sqrt(weight[i,k])*(X[i,d]-mean[k,d]). Accumulate C[k]=sum_batches transpose(Y)*Y, using beta0 first then beta1 for subsequent batches. Divide by the explicit denominator, mirror one authoritative triangle and add reg_covar to diagonal. Means and responsibilities are fixed across the centered accumulation pass. Raw second moment minus mean outer product is a separate numerical algorithm, not substituted without its own denominator/rounding argument.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| batch_rows | compile / software | {"lo": 1, "hi": 2147483647} | Number of samples accumulated per matrix batch; changing it changes floating reduction order. |

## Coupled constraints and scope

NVIDIA GMM full covariance M-step. Source denominator is N_k+10*epsilon, so that exact regularized convention must be retained; no unregularized normalization or equivalence to diagonal raw moments implied. Resource effects: Materializes batch_rows*dim centered scratch per component and contracts dim*dim Gram outputs; often reuses matrix multiplication infrastructure, with additional pass over samples after means finish.

Scope predicate: `p.weights_nonnegative and p.center_and_denominator_defined and p.matrix_product_numeric_policy_admitted and p.batches_cover_samples and p.symmetric_output_contract and p.regularization_defined`.

- Precondition `pre_0` (semantic): `p.weights_nonnegative and p.center_and_denominator_defined` — Weights are finite nonnegative; mean center and normalization denominator are explicit, denominator positive.
- Precondition `pre_1` (semantic): `p.matrix_product_numeric_policy_admitted and p.batches_cover_samples` — Gram-product arithmetic and beta accumulation are allowed; batches partition samples exactly.
- Precondition `pre_2` (semantic): `p.symmetric_output_contract and p.regularization_defined` — Output covariance is symmetric with declared diagonal regularization.

## Local example and misuse

For a local instance of `map`, identify the object `weighted_centered_gram_covariance` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/weighted_centered_gram_covariance@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/weighted_centered_gram_covariance@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Materializes batch_rows*dim centered scratch per component and contracts dim*dim Gram outputs; often reuses matrix multiplication infrastructure, with additional pass over samples after means finish.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `weighted_product` (proof): Each Gram term equals the specified weighted centered outer product under admitted sqrt/multiply rounding.
- `batch_accumulation` (proof): Each sample contributes once and first/subsequent beta settings preserve all batches.
- `center_lifetime` (proof): All batches use the same finalized means and weight generation.
- `symmetry` (proof): Triangle mirroring has unique writers and regularization is added once per diagonal.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
