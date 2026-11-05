# row_relative_fixed_point_mass@1: row exponent relative nonnegative mass codec

## When to consider

When reproducible probability prefixes span mixed FP32 exponents, choose a common unit from the row maximum exponent and align significands into bounded uint64 masses. Source padding29 keeps each maximum below2**53 and a1024-value total below2**63. Preserve ID/order, account for every dropped residual, and bind threshold, RNG and output-numerator policies independently of integer-scan determinism.

## Computation before and after

Quantize each row value downward in a unit set by the maximum exponent and fixed significand padding, then perform exact integer mass scans.

Let E0 be the maximum value biased exponent and e0=max(E0,1). For each FP32 value w_i, let e_i=max(E_i,1) and m_i=fraction_i+(2**23 if E_i>0 else0). Set shared real unit Delta=2**(e0-150-fraction_padding_bits), used as an exact analysis/decoder scale rather than requiring a representable FP32 scale. Emit q_i=floor((m_i*2**fraction_padding_bits)/2**(e0-e_i)), using a guarded right shift (oversized shift gives0). Equivalently q_i=floor(w_i/Delta). Preserve original ID/order and scan q_i in uint64 with exact nonoverflowing additions. The decoded mass is q_i*Delta<=w_i; each discarded residual is below Delta. The maximum retains positive mass, so Q=sum(q_i)>0. For a normalized encoded distribution use q_i/Q, not automatically the original w_i divided by Q*Delta. Downstream sorted top-p, CDF sampling, finite integer-to-FP64 comparisons, special rows and optional original-numerator output are separate contracts.

Recover the original computation: Replace integer q_i by exact decoded mass q_i*Delta at the original coordinate. Total discarded mass D=sum_i(w_i-q_i*Delta) satisfies0<=D<N*Delta. On the unchanged complete row, total variation between normalized original and decoded masses is at most D/sum(w), by expressing original mass as decoded mass plus a nonnegative residual. This does not bound a changed top-p membership without a separate crossing premise.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| fraction_padding_bits | compile / software | {"lo": 0, "hi": 29} | Append this many zero bits to the24-bit FP32 significand before aligning to the row maximum exponent. Source fixes29; smaller inferred widths trade finer mass/error range for lower integer magnitude and require their own supported converter/policy. |

## Coupled constraints and scope

Native Cake sparse top-p uses padding29 and at most1024 slab entries, with row maximum effective exponent, guarded shifts and uint64 scan. Generated t32_i2 witness handles up to64 entries; wider kernels/slab menus have separate shape bindings. Smaller padding widths are inferred numerical variants, not registered source modes. This codec permits mixed exponents but only finite positive-sign inputs; infinity and empty-row branches are outside it. Native top-p/finite-uniform comparisons and original-value renorm output are not covered by the ideal integer mass-law identity.

Scope predicate: `p.input_bits == 32 and p.fraction_bits == 23 and p.mass_word_bits == 64 and (p.positive_finite_row_max and p.all_values_nonnegative_finite_fp32_with_zero_sign_bit) and (p.row_max_exponent_and_value_same_generation_as_all_values) and (p.hidden_bits_subnormals_and_oversized_shifts_handled and p.selected_padding_has_implemented_converter) and (p.row_elements_between_one_and_1024 and p.uint64_prefix_census_exact_and_nonoverflowing and p.selected_scan_shape_covers_all_live_row_elements) and (p.value_id_order_and_padding_identity_preserved and p.prefix_partitions_cover_same_declared_order) and (p.downward_mass_quantization_and_dropped_positive_values_admitted and p.normalized_mass_error_budget_explicit) and (p.row_values_and_scale_published_before_conversion and p.integer_partials_publish_before_prefix_readers and p.mass_buffers_retire_before_reuse) and (p.consumer_integer_to_float_comparisons_rng_and_output_numerators_explicit)`.

- Precondition `pre_values` (implementation): `p.positive_finite_row_max and p.all_values_nonnegative_finite_fp32_with_zero_sign_bit` — The row is already sanitized to nonnegative finite positive-sign FP32 and has a positive maximum. NaN/negative/negative-zero/infinite/zero-row policies belong to a separate producer contract.
- Precondition `pre_peak` (implementation): `p.row_max_exponent_and_value_same_generation_as_all_values` — The shared row maximum is exact for the same immutable row; each value exponent is no larger, so unsigned exponent differences are valid.
- Precondition `pre_conversion` (implementation): `p.hidden_bits_subnormals_and_oversized_shifts_handled and p.selected_padding_has_implemented_converter` — Extract the FP32 significand with a hidden one only for normal inputs, use effective exponent max(E,1), and produce zero when the right shift exceeds the native word width. Source converter padding29 is implemented; other widths are inferred bindings.
- Precondition `pre_count` (implementation): `p.row_elements_between_one_and_1024 and p.uint64_prefix_census_exact_and_nonoverflowing and p.selected_scan_shape_covers_all_live_row_elements` — Every row/slab occurrence contributes once to a uint64 prefix sum. At most1024 values and padding<=29 keep every total below2**63. The selected native scan shape covers every live entry, including smaller64-entry versus full1024-entry variants.
- Precondition `pre_coordinates` (implementation): `p.value_id_order_and_padding_identity_preserved and p.prefix_partitions_cover_same_declared_order` — Codec and scan preserve requested value/ID order and original occurrence multiplicity. Sorting and exact-k/tie selection are separate refinements, and inactive padding has zero mass.
- Precondition `pre_numeric` (implementation): `p.downward_mass_quantization_and_dropped_positive_values_admitted and p.normalized_mass_error_budget_explicit` — The contract admits downward fixed-point masses, possibly dropping positive values below one unit, and the resulting normalized-law error. A changed top-p support boundary needs its own certificate.
- Precondition `pre_lifetime` (implementation): `p.row_values_and_scale_published_before_conversion and p.integer_partials_publish_before_prefix_readers and p.mass_buffers_retire_before_reuse` — All values/exponents/partial sums belong to one complete row generation, publish before readers and survive through the prefix consumers.
- Precondition `pre_consumers` (implementation): `p.consumer_integer_to_float_comparisons_rng_and_output_numerators_explicit` — Consumers declare any integer-to-FP64 conversion, finite RNG endpoint convention, threshold comparisons and whether output numerators are original values or decoded integer masses. Exact integer scans do not imply exact downstream real decisions.
- Constraint `c_representation` (implementation): `p.input_bits == 32 and p.fraction_bits == 23 and p.mass_word_bits == 64` — FP32 values and uint64 masses with source-compatible bounded prefix arithmetic.

## Local example and misuse

For a local instance of `contract`, identify the object `row_exponent_relative_nonnegative_mass_codec` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/row_relative_fixed_point_mass@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/row_relative_fixed_point_mass@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One uint64 mass per live/padded coordinate and complete prefix totals; original value/ID may be retained separately.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `codec_identity` (test): Significand/exponent alignment equals floor(w_i/Delta), including subnormals, zeros and shifts beyond word width.
- `integer_prefix` (proof): Every occurrence contributes once; all scan regroupings yield identical prefixes and total without uint64 overflow.
- `mass_error` (proof): Decoded masses are downward with per-value residual below one unit and the declared whole-row normalized error bound.
- `consumer_policy` (test): Each threshold/sample/output consumer uses its declared mass, numerator, floating-conversion and RNG endpoint policy without assuming exact integer decisions imply exact floating decisions.

Small positive values can become zero mass. FP64 conversions can round large exact prefixes, and original-value renormalization is not the same vector as q_i/Q. Top-p support changes may amplify a tiny mass difference at the threshold.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
