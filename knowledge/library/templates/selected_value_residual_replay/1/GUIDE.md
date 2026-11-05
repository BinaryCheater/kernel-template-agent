# selected_value_residual_replay@1: selected value error rows replayed without mass

## When to consider

When a small fraction of V rows carry large quantization residuals, replay only those key identities with narrow residual values. Add their PV corrections but keep them out of the denominator. Preserve the original partial tail and base-code generation, and check replay rounding instead of assuming a copied key gives identical native probabilities.

## Computation before and after

Replay selected key scores to add value quantization residuals without duplicating normalization mass.

Encode original values with per-channel s[d] and base codes b[j,d]. Choose repair_tokens unique rows by the selected policy. For selected j form e[j,d]=V[j,d]/s[d]-decode(b[j,d]) and store narrow repair code r[j,d]=encode(e[j,d]) with copied key/score identity j. Original rows contribute w[j]*decode(b[j,d]) to N and w[j] to kept mass L. Replay rows contribute w_replay[j]*decode(r[j,d]) to N and zero to L. Keep all terms in the current online frame, correcting N and L at reference updates. A diagnostic all-row mass may be maintained but must not normalize output. Finalize s[d]*N[d]/L and the declared output cast. Physical repair tiles may precede the moved original partial tail; roles, not a naive physical cutoff, determine denominator membership.

Recover the original computation: With identical common-frame original/replay weights, combine each selected base and repair term into w[j]*(decode(b[j])+decode(r[j])). Unselected rows retain their base value, and normalization counts each original key once. Native replay/cast differences are admitted separately.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| repair_tokens | compile / software | {"lo": 1, "hi": 2147483647} | Number of unique original rows to repair; source ranks squared per-channel decoded residual energy, independent of query probability. |

## Coupled constraints and scope

Source VC-Attention V repair, dense fixed-length D128 MHA, demean disabled, E4M3 base/repair values, BF16 or separately quantized E4M3 Q/K. Repair records occupy full128-row tiles inserted before the shifted original tail. The source chooses high residual energy without an accuracy-optimality guarantee. No exact equivalence of later recomputed ExpCast weights or independently scaled repair keys is assumed.

Scope predicate: `(repair_tokens >= 1 and repair_tokens <= p.original_key_rows and p.value_features >= 1) and (p.base_value_quantizer_and_residual_decoder_explicit and p.same_per_channel_scale_for_base_and_repair) and (p.selected_original_rows_unique_in_range and p.repair_padding_codes_zero) and (p.repair_keys_query_mask_bias_and_positions_replay_original_scores_with_admitted_rounding) and (p.only_original_rows_contribute_to_final_denominator and p.original_tail_rows_retained_once) and (p.base_repair_numerator_and_kept_mass_share_online_frame and p.frame_changes_rescale_all_old_contributions_once) and (p.base_repair_probability_and_output_cast_error_admitted and p.final_kept_mass_positive_finite) and (p.repair_maps_codes_scales_and_keys_same_generation_and_alive and p.native_replay_path_and_resources_supported)`.

- Precondition `pre_residual` (semantic): `p.base_value_quantizer_and_residual_decoder_explicit and p.same_per_channel_scale_for_base_and_repair` — Repair codes represent V/s minus the actual decoded base code, not V itself or a residual from another quantizer generation.
- Precondition `pre_selection` (semantic): `p.selected_original_rows_unique_in_range and p.repair_padding_codes_zero` — Every selected source row is corrected once; padding adds no numerator contribution.
- Precondition `pre_score` (implementation): `p.repair_keys_query_mask_bias_and_positions_replay_original_scores_with_admitted_rounding` — Each replay row refers to the original key and support identity. Requantized copied keys and changed reduction order need an explicit error contract.
- Precondition `pre_mass` (semantic): `p.only_original_rows_contribute_to_final_denominator and p.original_tail_rows_retained_once` — Repair/padding rows add no denominator mass; original rows, including a moved partial tail, remain present exactly once.
- Precondition `pre_frame` (semantic): `p.base_repair_numerator_and_kept_mass_share_online_frame and p.frame_changes_rescale_all_old_contributions_once` — Both correction and base terms use a common reference even when replay changes the physical traversal/max history.
- Precondition `pre_numeric` (implementation): `p.base_repair_probability_and_output_cast_error_admitted and p.final_kept_mass_positive_finite` — The output is a selected approximation with a positive original-support denominator; duplicated keys do not make it exact.
- Precondition `pre_lifetime` (implementation): `p.repair_maps_codes_scales_and_keys_same_generation_and_alive and p.native_replay_path_and_resources_supported` — Prepared residual/code/key maps and scratch remain paired through the complete consumer.
- Constraint `c_budget` (semantic): `repair_tokens >= 1 and repair_tokens <= p.original_key_rows and p.value_features >= 1` — Unique repaired original rows fit the selected budget.

## Local example and misuse

For a local instance of `contract`, identify the object `selected_value_error_rows_replayed_without_mass` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/selected_value_residual_replay@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/selected_value_residual_replay@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Selected IDs, copied keys, narrow value residual rows and a separate kept original mass.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `residual_identity` (test): Repair residuals are computed from the actual stored base code and use its decoder/scale generation.
- `weight_replay` (test): Copied keys retain original score/mask identity, and replay probability differences meet the numeric policy.
- `mass_roles` (test): All original rows contribute once to kept mass; no replay or padding row does, including around a relocated tail.
- `selection_lifetime` (test): Selection is unique and complete within budget; keys, codes and metadata remain paired and live.

Counting copied-key rows in the denominator changes attention. Repair does not necessarily improve every query: selection ignores query probabilities, and new score/probability casts can introduce their own error.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
