# nonnegative_scale_hoisting@1: factored relu scale

## When to consider

Use for weighted ReLU reductions with one nonnegative scale shared over the group; choose after_activation or after_reduction placement. Apply ReLU to unscaled scores and move the scale to the chosen site, for example scale2 and scores[-3,4] with weights[1,-1] give2*(0-4)=-8. Keep a separate scale for each output and require numerical admission before changing finite-precision placement.

## Computation before and after

Factor a shared nonnegative scale through ReLU and a weighted group reduction.

Refine y=sum_i weight_i*relu(scale*score_i). With after_activation compute sum_i weight_i*(scale*relu(score_i)); with after_reduction compute scale*sum_i weight_i*relu(score_i). A parallel reduction may apply this to each local group before combining its partials, provided all those groups use the same output's scale. Signed weights are allowed; the scale is nonnegative.

Recover the original computation: Move the scale back inside ReLU and each weighted term using nonnegative homogeneity and distributivity, under the admitted numerical contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| placement | compile / software | ["after_activation", "after_reduction"] | Apply the shared scale after each ReLU or once after the weighted group reduction. |

## Coupled constraints and scope

Multiplicative nonnegative scales shared over a weighted group of ReLU inputs. Signed weights are legal. Negative scales invalidate homogeneity; per-term scales cannot move outside a group reduction. The source witness is SM90 MQA with one positive quantization scale per KV token shared across query heads. Arbitrary nonlinearities,affine offsets and changed quantization are outside this rule.

Scope predicate: `p.relu_requested and p.min_scale >= 0 and p.scale_shared_over_group`.

- Precondition `pre_finite` (semantic): `p.finite_values and p.finite_scale and p.relu_requested` — The coarse operation uses ReLU with finite scores,weights and scales.
- Precondition `pre_numerics` (semantic): `p.numeric_contract_allows_scale_hoisting` — The caller admits the selected intermediate rounding and multiplication/addition order.
- Constraint `c_scale` (semantic): `p.min_scale >= 0 and p.scale_shared_over_group` — ReLU is positively homogeneous and the same scale applies to every term in the factored group.

## Local example and misuse

For a local instance of `reduce`, identify the object `factored_relu_scale` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/deepgemm/delta/nonnegative_scale_hoisting@1/template.yaml`. Original outline: `knowledge/studies/deepgemm/delta/nonnegative_scale_hoisting@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Original score/weight intermediates and one scale per factored output group.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `homogeneity` (exhaustive): ReLU(scale*x) equals scale*ReLU(x) for every admitted finite score and nonnegative scale in exact arithmetic.
- `scale_ownership` (exhaustive): Every factored term belongs to the output and group for which this scale is defined.
- `numerical_policy` (test): Finite-precision placement and reduction order satisfy the requested tolerance,including cancellation and large scales.

Changing scale placement and sum order changes rounding; signed-weight cancellation,zero scales and large_dynamic_range require numerical admission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
