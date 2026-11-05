# numerator_weight_gain@1: gained narrow weights with ungained normalizer

## When to consider

When a narrow probability operand loses small exponential weights, multiply only that operand by a fixed gain before conversion. Accumulate the gained PV numerator, retain the original higher-precision denominator, and divide output by the gain once. This can change useful representable weights but requires an explicit error/range policy. The denominator and LSE never receive the gain; this differs from scaling a whole normalized reduction state.

## Computation before and after

Quantize gained exponential weights for PV while keeping the original high-precision denominator.

For each row/tile form ordinary nonnegative weights w in the current online reference. Update the high-precision denominator d from those ungained weights. Let G be 1, 2**gain_log2 or the selected format maximum according to gain_policy. Convert G*w into the declared narrow probability format Q and accumulate Q(G*w)*Vcode in a wide numerator A. On reference changes multiply only the previous A by the same online correction c used for previous d before adding new terms; G is unchanged throughout the live state. At finalization for d>0 return A*Vscale/(G*d), where Vscale may be factored only when constant across contracted keys. Preserve per-term scales otherwise. For empty support take the declared branch without dividing by zero. Form row LSE from the original reference and ungained d, with no log(G) correction. Quantized weights are not a replacement denominator. Coordinate permutations and storage depend on the actual MMA realization and remain explicit obligations.

Recover the original computation: Replace each quantized weight by G*w plus its declared quantization residual; dividing the final numerator by G recovers the original weighted contraction plus the admitted residual contribution, normalized by the original row mass.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| gain_policy | compile / software | ["unit", "power_of_two", "format_max"] | Choose no gain, a power-of-two gain or the selected FP8 format maximum. |
| gain_log2 | compile / software | {"lo": 0, "hi": 8} | Power-of-two exponent when that policy is selected; source classic FP8 value-split path binds eight. |

## Coupled constraints and scope

Finite E4M3/E5M2 contraction weights with a fixed numerator-only gain, wider row mass/output accumulation and an explicit approximation contract. Source classic value-split FP8 binds G=256; Hopper binds G=448 or 57344 and folds a per-KV-head V scale into finalization. No common denominator gain, arbitrary changing tile gain, inferred uniform error bound or bitwise equivalence to ungained quantization is supplied.

Scope predicate: `p.weights_nonnegative_finite_in_actual_online_reference and p.masked_weights_zero_and_empty_support_branch_defined and p.mass_accumulated_from_ungained_weights_before_quantization and p.lse_uses_ungained_mass_and_actual_reference and p.gain_finite_positive_and_constant_over_each_live_numerator and p.gain_matches_quantizer_format_and_range and p.old_numerator_corrected_to_new_reference_before_new_terms and p.value_scale_constant_over_contracted_keys_or_applied_per_term and p.weight_gain_quantizer_cast_and_mma_error_policy_admitted and p.final_reciprocal_and_accumulator_range_admitted and p.quantized_weight_fragment_map_matches_value_contraction and p.weight_generations_publish_before_reads_and_retire_before_reuse`.

- Precondition `pre_weights` (semantic): `p.weights_nonnegative_finite_in_actual_online_reference and p.masked_weights_zero_and_empty_support_branch_defined` — Weights use the declared online reference; excluded terms are zero and empty rows have an explicit nondividing finalization.
- Precondition `pre_mass` (semantic): `p.mass_accumulated_from_ungained_weights_before_quantization and p.lse_uses_ungained_mass_and_actual_reference` — The row denominator is formed from original weights, not gained or quantized probabilities; LSE stays in that original frame.
- Precondition `pre_gain` (implementation): `p.gain_finite_positive_and_constant_over_each_live_numerator and p.gain_matches_quantizer_format_and_range` — One known positive gain applies to all probability contributions held in a numerator; changes require an explicit state conversion.
- Precondition `pre_reference` (semantic): `p.old_numerator_corrected_to_new_reference_before_new_terms and p.value_scale_constant_over_contracted_keys_or_applied_per_term` — Online correction acts on the old numerator; a factored value scale is constant over its contraction, otherwise scales remain on individual terms.
- Precondition `pre_numeric` (implementation): `p.weight_gain_quantizer_cast_and_mma_error_policy_admitted and p.final_reciprocal_and_accumulator_range_admitted` — The numerical contract includes gained-weight rounding/underflow, accumulator range and final reciprocal arithmetic.
- Precondition `pre_layout` (implementation): `p.quantized_weight_fragment_map_matches_value_contraction and p.weight_generations_publish_before_reads_and_retire_before_reuse` — The selected backend consumes the correct quantized P coordinates with legal publication/lifetime.
- Constraint `c_gain` (implementation): `(gain_policy == 'unit' and p.gain_value == 1) or (gain_policy == 'power_of_two' and p.gain_value == 2 ** gain_log2) or (gain_policy == 'format_max' and p.gain_value == p.quantizer_max)` — The actual multiplier is derived from the selected policy, not an independent hidden scale.
- Constraint `c_format` (implementation): `p.quantizer_max == 448 or p.quantizer_max == 57344` — This version binds E4M3/E5M2 probability formats; other encodings need another numeric binding.
- Constraint `c_nonempty` (semantic): `p.row_count >= 1 and p.value_features >= 1` — Nonempty row/feature domain; individual rows may have empty key support.
- Constraint `c_power` (implementation): `gain_policy != 'power_of_two' or gain_log2 % 1 == 0` — Power-of-two gain uses an integral exponent.

## Local example and misuse

For a local instance of `contract`, identify the object `gained_narrow_weights_with_ungained_normalizer` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/numerator_weight_gain@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/numerator_weight_gain@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Larger weight codes use more of a narrow format range without changing the logical normalizer.. Storage: Narrow P fragments plus wide numerator and ungained row state.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: quantized_weights
  level: reg
  scope: warp
  bytes: p.live_weight_elements * p.quantized_element_bytes
overlap: []
```

## Obligations and numerical checks

- `gain_and_frame` (proof): One fixed gain and matching reference correction govern every term in the live numerator, while the denominator and LSE remain ungained.
- `quantized_error` (test): Quantizer, exponent, gain arithmetic, accumulation and reciprocal errors satisfy the requested policy over the admitted finite range.
- `scale_and_layout` (test): Factored value scales are constant over contracted terms; actual quantized fragment layouts and generations match the PV operation.
- `empty_and_final` (test): Positive mass receives exactly one inverse-gain compensation; empty rows avoid undefined division and preserve their declared outputs.

Q(G*w)/G and Q(w) differ. G can preserve small weights or change rounding and range pressure; non-power-of-two gain adds its own arithmetic error. Ungained row mass must not be replaced by a sum of quantized P.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
