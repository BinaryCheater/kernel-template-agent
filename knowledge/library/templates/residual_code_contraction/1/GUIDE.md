# residual_code_contraction@1: scaled high residual codes with separate contractions

## When to consider

When a single narrow probability operand loses significant PV accuracy, encode its rounding residual in a second signed operand row and run a second contraction. Add both FP32 results before applying their shared scale. Keep original row mass and verify the actual MMA/cast order rather than assuming a checkpoint reconstruction rule applies.

## Computation before and after

Trade an extra narrow contraction for a signed low-code correction to scaled weights.

For row/tile weights w_j and positive decoder scale s, form x_j=fl_FP32(w_j*encode_multiplier) under the explicit reciprocal policy, then apply declared finite-range clamp. Encode h_j=RN_E4M3(x_j). With high_and_residual compute e_j=fl_FP32(x_j-decode(h_j)), clamp as declared, and encode signed l_j=RN_E4M3(e_j). Keep h and l as separate narrow operand rows. Compute H_d=sum_j decode(h_j)*V_jd and L_d=sum_j decode(l_j)*V_jd with independent FP32 MMA accumulation; single_code omits L. Add H_d+L_d in FP32, multiply by s once and add to the current online numerator. Original mass sum_j w_j stays independent and is corrected in the same online frame. Normalize once by that original mass; any V decoder scale and final output cast are separate declared stages. Do not reconstruct h+l in E4M3 or move reconstruction before the contractions without a numerical equivalence binding.

Recover the original computation: Distributivity identifies s*(H+L) with a contraction over reconstructed weights s*(decode(h)+decode(l)) in exact arithmetic. Native separate-MMA rounding and encoding error define the selected approximation to the original numerator; the original denominator is unchanged.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| encoding | compile / software | ["single_code", "high_and_residual"] | One E4M3 contraction or two component contractions. Source dense MLA uses paired codes; single-code projection is an inferred accuracy/work alternative requiring its own backend binding. |

## Coupled constraints and scope

Pinned B12x dense MLA SM120/121 FP8 PV uses64-key score chunks, eight logical heads, high/low E4M3 operand rows and FP32 MMA. Source s=max(amax(w),1e-10)/448 with explicit FP32 reciprocal encoding, high clamp±448 and signed residual clamp±448. Other scale policies and single-code projection require their own native/numeric binding. This is not exact FP32 recovery and does not promise monotone final output error improvement.

Scope predicate: `p.key_rows >= 1 and p.value_features >= 1 and (p.nonnegative_finite_weights_and_finite_values and p.complete_logical_weight_value_coordinates) and (p.positive_finite_row_tile_scale_and_encode_reciprocal_policy_explicit and p.scale_code_and_value_generation_match) and (p.high_code_is_rounded_scaled_weight and (encoding != 'high_and_residual' or (p.low_code_is_rounded_difference_from_actual_decoded_high and p.signed_e4m3_residual_bits_preserved))) and (p.each_code_component_contracts_all_keys_against_same_values and p.component_results_added_in_fp32_before_one_decoder_scale) and (p.normalizer_uses_declared_unquantized_weight_mass_and_shared_online_frame) and (p.encode_cast_mma_component_sum_scale_and_final_error_admitted and p.saturation_underflow_and_clamp_policy_explicit) and (p.high_low_storage_published_and_alive_through_all_component_readers and p.selected_encoding_backend_and_resources_supported)`.

- Precondition `pre_domain` (semantic): `p.nonnegative_finite_weights_and_finite_values and p.complete_logical_weight_value_coordinates` — The complete finite weight/value contraction is identified before encoding; low residual components may be signed.
- Precondition `pre_scale` (implementation): `p.positive_finite_row_tile_scale_and_encode_reciprocal_policy_explicit and p.scale_code_and_value_generation_match` — Each row/tile uses one decoder scale and its declared FP32 encode multiplier; every component shares that scale and V generation.
- Precondition `pre_encode` (implementation): `p.high_code_is_rounded_scaled_weight and (encoding != 'high_and_residual' or (p.low_code_is_rounded_difference_from_actual_decoded_high and p.signed_e4m3_residual_bits_preserved))` — Compute residual before releasing the producer value, using the actual rounded high code; never clamp low residuals to nonnegative probabilities.
- Precondition `pre_contraction` (semantic): `p.each_code_component_contracts_all_keys_against_same_values and p.component_results_added_in_fp32_before_one_decoder_scale` — Separate MMAs cover identical key/value coordinates; sum high/low FP32 products and decode scale once.
- Precondition `pre_mass` (semantic): `p.normalizer_uses_declared_unquantized_weight_mass_and_shared_online_frame` — Probability encoding changes the PV numerator; it does not replace denominator mass by code sums or add another copy of mass.
- Precondition `pre_numeric` (implementation): `p.encode_cast_mma_component_sum_scale_and_final_error_admitted and p.saturation_underflow_and_clamp_policy_explicit` — Two-code arithmetic is an admitted approximation; scale reciprocals, signed low underflow, saturation and MMA association are part of the recipe.
- Precondition `pre_lifetime` (implementation): `p.high_low_storage_published_and_alive_through_all_component_readers and p.selected_encoding_backend_and_resources_supported` — Both component tiles retain coordinate/generation identity until every MMA reader finishes; selected single/paired mode must be implemented.
- Constraint `c_shape` (semantic): `p.key_rows >= 1 and p.value_features >= 1` — Nonempty weight/value contraction domain.

## Local example and misuse

For a local instance of `contract`, identify the object `scaled_high_residual_codes_with_separate_contractions` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/residual_code_contraction@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/residual_code_contraction@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One or two E4M3 weight rows and one shared row/tile decoder scale.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `residual_generation` (test): Low code is computed from the same scaled input and decoded high code, retaining its sign and generation.
- `component_coverage` (test): Both components visit every intended key/value once and map their outputs to the same final feature coordinates.
- `arithmetic_order` (test): Separate MMA accumulation, FP32 component addition and one scale implement the declared approximation.
- `mass_frame` (test): No code-derived or duplicate mass replaces the original denominator; online correction treats old/new numerator terms in their intended frame.

Even if h+l reconstructs a weight more accurately, separate FP32 accumulation and cancellation can change final error. Rounding h+l back to E4M3 can erase the entire correction; unsigned low codes turn negative corrections into positive bias.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
