# shared_exponent_integer_sum@1: count and fraction sum for one exponent

## When to consider

When many positive FP32 values in a histogram bin share one exponent, accumulate their integer fraction bits and counts instead of racing floating additions. Reconstruct the exact significand once with a declared toward-zero rounding. Keep powers-of-two counts, separate empty/subnormal bins and publish both totals before use. This gives order-independent bucket bits, not automatic nearest-even accuracy or unchanged downstream top-p boundaries.

## Computation before and after

Exact fraction/count integer reduction for a shared exponent, followed by one explicit toward-zero FP32 reconstruction.

For each nonnegative FP32 value in a bin, extract its23-bit fraction m and shared exponent E. Accumulate count N and fraction sum M as exact uint64/signed-safe32 integer channels, directly or through CTA partials. For N0 emit zero. Otherwise form exact integer significand S=M+N*2**23 when E>0, or S=M for E0. Exact real sum is S*2**(max(E,1)-150). If S0 emit zero. Reconstruct FP32 by normalizing S, retaining its most significant24 bits when normal, and discarding lower bits without an increment; a still-subnormal S fits its exact23-bit fraction. Include normal hidden-bit counts and carry into the output exponent. This implements one nonnegative round-toward-zero reconstruction. A native implementation may calculate carries from(M>>23)+N rather than materialize S; all equivalent shifts/ranges must match. Bin mass then feeds separately refined scans/selection; integer determinism does not authorize changing their numerical contract.

Recover the original computation: Collect all original occurrences of one shared-exponent bin, sum them exactly as real values, then apply the requested one final nonnegative FP32 toward-zero rounding. Exact integer addition is independent of producer order and grouping.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| aggregation | compile / software | ["direct_global", "cta_then_global"] | Accumulate exact integer channels directly in global bins or first in CTA-private bins then merge globally. Source uses cta_then_global; direct_global is inferred with the same census/completion premises. |

## Coupled constraints and scope

AIR deterministic FP32 histogram uses uint64 mantissa sums and Int32 counts per bin; initial11-bit descending-key buckets fix the exponent, and later prefix passes retain it. Source empty bins bypass calcFloatValue. This is one-exponent nonnegative summation, not a general superaccumulator, mixed-sign/mixed-exponent sum or nearest-even reduction. Wider input formats/counters and output-overflow policy require their own binding.

Scope predicate: `p.bins >= 1 and p.fraction_bits == 23 and p.input_bits == 32 and (p.nonnegative_finite_fp32_inputs and p.each_bin_has_one_exact_fp32_exponent_field and p.input_sign_bits_zero) and (p.each_value_contributes_one_count_and_its_23_bit_fraction_once and p.count_fits_signed_int32_and_fraction_sum_fits_uint64) and (p.reconstructed_exponent_matches_all_inputs_and_generation) and (p.partial_integer_additions_exact_without_overflow and p.both_count_and_fraction_totals_publish_before_reconstruction) and (p.zero_count_bins_bypass_normal_reconstruction_and_emit_zero) and (p.exact_nonnegative_sum_within_finite_fp32_range and p.exponent_carry_and_shift_widths_valid) and (p.one_final_toward_zero_reconstruction_requested and p.downstream_admits_truncated_bucket_sum) and (p.private_bin_counters_initialized_and_reader_retirement_precedes_reuse and p.selected_integer_atomic_and_completion_backend_supported)`.

- Precondition `pre_domain` (implementation): `p.nonnegative_finite_fp32_inputs and p.each_bin_has_one_exact_fp32_exponent_field and p.input_sign_bits_zero` — All inputs are nonnegative finite FP32, and every bin shares one exact biased exponent field E, including a separate E0 domain for zeros/subnormals. Mixing adjacent exponents is not admitted. Input sign bits are zero; negative zero needs a separate canonicalization policy.
- Precondition `pre_census` (implementation): `p.each_value_contributes_one_count_and_its_23_bit_fraction_once and p.count_fits_signed_int32_and_fraction_sum_fits_uint64` — For every occurrence retain N+=1 and M+=its23-bit fraction; N<=INT32_max and M fits uint64. Zero mantissa still contributes count, even if its fraction atomic can be omitted.
- Precondition `pre_exponent` (implementation): `p.reconstructed_exponent_matches_all_inputs_and_generation` — The shared exponent is stored or recovered from a complete key prefix that identifies the same bin/data generation. Sign bits or an unrelated bucket representative cannot replace it.
- Precondition `pre_merge` (implementation): `p.partial_integer_additions_exact_without_overflow and p.both_count_and_fraction_totals_publish_before_reconstruction` — CTA partials and global totals add exact nonoverflowing integers; both channels are complete before reconstruction, so no atomic pairing is needed during accumulation.
- Precondition `pre_empty` (implementation): `p.zero_count_bins_bypass_normal_reconstruction_and_emit_zero` — Empty bins emit zero explicitly. Normal-exponent reconstruction with N0/M0 would not represent an empty sum.
- Precondition `pre_range` (implementation): `p.exact_nonnegative_sum_within_finite_fp32_range and p.exponent_carry_and_shift_widths_valid` — The exact sum stays within FP32 finite range and count limits keep native shifts/exponent carries defined.
- Precondition `pre_rounding` (implementation): `p.one_final_toward_zero_reconstruction_requested and p.downstream_admits_truncated_bucket_sum` — Reconstruction discards excess low significand bits once, equivalent to rounding the nonnegative exact sum toward zero, not necessarily nearest-even. Downstream cumulative boundaries must admit that policy separately.
- Precondition `pre_lifetime` (implementation): `p.private_bin_counters_initialized_and_reader_retirement_precedes_reuse and p.selected_integer_atomic_and_completion_backend_supported` — Each bin epoch has zeroed exclusive counters, implemented native integer atomics/completion, and no reset or reuse before readers retire.
- Constraint `c_bins` (implementation): `p.bins >= 1 and p.fraction_bits == 23 and p.input_bits == 32` — Nonempty bin table with FP32 fraction width and fixed integer counter representation.

## Local example and misuse

For a local instance of `reduce`, identify the object `count_and_fraction_sum_for_one_exponent` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/shared_exponent_integer_sum@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/shared_exponent_integer_sum@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: 8-byte fraction sum plus4-byte count per global/optional CTA bin; exponent may come from its key prefix.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: global_integer_bins
  level: dram
  scope: kernel
  bytes: 12 * p.bins
- id: cta_integer_bins
  level: smem
  scope: cta
  bytes: 12 * p.bins
  when: aggregation == 'cta_then_global'
overlap: []
```

## Obligations and numerical checks

- `significand_identity` (proof): N and M recover the exact shared-exponent real sum, including hidden leading bits, powers of two and zero/subnormal bins.
- `order_and_grouping` (test): Every nonoverflowing producer permutation and CTA partition yields the same N/M totals and reconstructed bits.
- `reconstruction_policy` (test): Native exponent carries and discarded low bits implement the declared toward-zero result, with empty bins bypassed and no overflow.
- `publication` (test): Both integer channels and the shared exponent describe one complete generation before reconstruction and survive every reader.

Reproducible reconstruction can differ from nearest-even and from sequential FP32 sums. Dropping zero-mantissa counts loses hidden ones; mixing exponent fields corrupts the sum even when mantissa atomics remain exact.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
