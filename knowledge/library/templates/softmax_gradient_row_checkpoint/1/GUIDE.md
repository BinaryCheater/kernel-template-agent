# softmax_gradient_row_checkpoint@1: output contracted softmax gradient center

## When to consider

When softmax backward would revisit all keys just to reduce P*dP, compute the same row center as dot(O,dO) using the saved weighted output. Subtract incoming natural-LSE gradient from that center, then each tile computes P*(dP-center). A zero-value sink uses -P_sink*center. Match the original support and checkpoint precision; a rounded or transformed O changes this identity unless its error is admitted or repaired by a better checkpoint.

## Computation before and after

Contract the softmax Jacobian row center through the saved weighted output, then apply each probability tile locally.

For O_i=sum_j P_ij*V_j and incoming g_i=dO_i, set a_ij=dot(g_i,V_j). Replace D_i=sum_j P_ij*a_ij with D_i=dot(g_i,O_i). Let e_i be the incoming gradient of natural-log LSE when lse_gradient=present and zero otherwise; save c_i=D_i-e_i. Each probability tile forms dZ_ij=P_ij*(a_ij-c_i). A finite zero-value sink has a_sink=0 and P_sink=exp(sink-LSE), so its per-row logit gradient is -P_sink*c_i; sum those terms over rows if the sink parameter is shared. Keep the probability tile's support and row identity unchanged. dV=P^T*g remains separate. Chain dZ through score scale/modifier derivatives before Q/K gradient contractions; c is a row statistic, not a replacement for those chain rules. A base2 LSE output's incoming gradient e2 must first be converted to natural-LSE units as e=e2/ln(2).

Recover the original computation: Substitute O=sum(P*V), expand dot(g,O), and recover the full probability-domain Jacobian-vector product and optional natural-LSE gradient.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| lse_gradient | compile / boundary | ["absent", "present"] | Whether the natural-log row normalizer has an incoming gradient e; e is zero when absent. |
| denominator_sink | compile / boundary | ["absent", "finite_zero_value"] | Whether normalization includes one finite trainable or constant sink logit with zero value vector. |

## Coupled constraints and scope

Deterministic normalized weighted sums with finite values/gradients and a positive finite normalization mass. A denominator-only finite sink contributes a zero value and may own all mass when ordinary support is empty. Zero total mass, arbitrary dropout replay, transformed O and gradients of the discrete support-selection operation are outside this version. Native rounding need not preserve the exact row-sum identities without numerical admission.

Scope predicate: `p.positive_finite_row_normalizer and p.finite_values_and_incoming_gradients and p.probability_support_matches_forward and p.output_is_same_probability_weighted_value_sum and p.no_unreplayed_dropout_or_post_output_transform and p.row_checkpoint_coordinates_generation_and_visibility_match and p.padded_rows_guarded and p.output_checkpoint_error_and_reduction_reassociation_admitted and p.probability_and_derivative_casts_admitted`.

- Precondition `pre_normalization` (semantic): `p.positive_finite_row_normalizer and p.finite_values_and_incoming_gradients and p.probability_support_matches_forward` — The row has defined normalized probabilities over its original support, possibly including a finite zero-value sink.
- Precondition `pre_output` (semantic): `p.output_is_same_probability_weighted_value_sum and p.no_unreplayed_dropout_or_post_output_transform` — O is the matching sum of P_j*V_j before any unrelated transform; changed probability domains and unreplayed random masks are excluded.
- Precondition `pre_state` (semantic): `p.row_checkpoint_coordinates_generation_and_visibility_match and p.padded_rows_guarded` — O, dO, P, optional dLSE and metadata agree on row and generation, and only valid rows are observed.
- Precondition `pre_numeric` (implementation): `p.output_checkpoint_error_and_reduction_reassociation_admitted and p.probability_and_derivative_casts_admitted` — Checkpoint error, row-reduction reassociation, probability generation and gradient casts satisfy the numerical contract.

## Local example and misuse

For a local instance of `reduce`, identify the object `output_contracted_softmax_gradient_center` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flash-attention/delta/softmax_gradient_row_checkpoint@1/template.yaml`. Original outline: `knowledge/studies/flash-attention/delta/softmax_gradient_row_checkpoint@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: One row statistic serves every derivative score tile and optional sink derivative.. Storage: One center per query row replaces a separate probability-domain center reduction; P storage/recomputation is chosen independently.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `row_identity` (proof): The output contraction equals the probability-domain center in real arithmetic and includes the same support and optional sink.
- `gradient_units_and_chain` (test): Optional LSE gradient has natural-log units and subsequent score derivatives, scales, grouped heads and shared sink reductions are complete.
- `numeric_checkpoint` (test): Actual saved O precision, reconstruction, probability generation and derivative reduction/casts satisfy the chosen numerical policy.

A rounded O perturbs every centered dZ term by -P*dot(g,O_error); strongly peaked rows can amplify its relative effect near cancellation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
