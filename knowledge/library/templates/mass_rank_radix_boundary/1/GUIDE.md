# mass_rank_radix_boundary@1: weighted radix mass crossing

## When to consider

When a top-p boundary is defined by cumulative mass rather than element count, use descending radix mass histograms to refine only its current value prefix. Carry residual target mass and separate counts; compact a prefix only when its full count fits, otherwise rescan original values. Keep the complete boundary-value cohort and normalize from the original coordinates with an explicit native sum policy.

## Computation before and after

Prefix-restricted weighted radix crossing with complete-cohort normalization.

Encode each value by the declared descending key. Start with empty prefix and residual target r=p*sum(values). For each most-significant digit (11-bit chunks, final5 or10), histogram only occurrences matching the previous prefix, storing both total mass H[b] and occurrence count N[b]. Traverse bins in descending-value order and choose the first occupied bin b whose cumulative mass through b is>=r. Subtract mass of all earlier bins from r, append b to the prefix and retain its count for the next pass. Count is presence/capacity metadata, never the mass rank. In rescan_original mode filter the original row anew. In compact_if_fits, a pass may copy its complete incoming-prefix occurrences into a private bank only when the known count fits; else read original input and omit that compact output. Following passes still filter by their current prefix within any broader cached incoming prefix. First pass reads original input. After all key bits, decode threshold tau, rescan original coordinates and retain every value>=tau, including the entire equality cohort; normalize by their complete retained mass and cast output. Compaction order is not output order and no boundary-ID subset is selected.

Recover the original computation: Sort the original nonnegative values descending, find the first cumulative mass>=p*W, take its value tau, and retain every original occurrence with value>=tau. Erasing radix digits and optional compact transport recovers this same threshold-cohort normalization.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| key_bits | compile / boundary | [16, 32] | Width of the admitted sortable value representation, giving radix digit widths11/5 or11/11/10. Native registration is32-bit;16-bit is an inferred logical binding requiring its own backend. |
| candidate_access | compile / software | ["rescan_original", "compact_if_fits"] | Re-read the immutable original row at each prefix pass or compact a complete incoming prefix into a bounded bank when it fits. Source uses capacity-dependent rescan/compaction. |

## Coupled constraints and scope

AIR top-p source uses11-bit descending CUB sortable digits, full-row mass fraction, per-bin mass and count, two bounded candidate banks, and inclusive final threshold. FP32 key schedule11/11/10 and16-bit11/5 are representation-derived. Source float atomics versus deterministic mantissa/count reconstruction are separate native arithmetic bindings; neither alone establishes exact-real cumulative sums. Source normalization floor1e-8, nonfinite values, ID-limited tie splitting and arbitrary weighted keys independent of their values are outside this version. csrc/renorm.cu registers float only. The16-bit schedule follows the generic header and has logical checks, not registered native support.

Scope predicate: `p.length >= 1 and p.histogram_bins >= 2048 and p.counts_and_indices_fit_int32 and p.candidate_capacity >= 0 and (p.finite_nonnegative_values and p.positive_finite_total_mass and p.mass_fraction_in_open_zero_closed_one) and (p.descending_key_encoding_monotone_and_equal_values_share_key and p.zero_and_special_key_policy_explicit and p.selected_key_width_has_implemented_backend) and (p.each_active_occurrence_histogrammed_once and p.histogram_count_and_mass_domains_identical) and (p.mass_accumulation_prefix_scan_and_residual_subtraction_policy_explicit and p.native_mass_comparisons_preserve_requested_crossing) and (candidate_access != 'compact_if_fits' or (p.complete_active_count_known_before_compaction and p.compact_only_when_count_fits_capacity and p.overflow_rescans_original_without_dropping_values)) and (candidate_access != 'compact_if_fits' or (p.compact_values_preserve_occurrence_multiplicity_and_prefix and p.compact_bank_publication_and_lifetime_safe)) and (p.histograms_and_counters_initialized_per_row_pass and p.all_producer_partials_published_before_boundary_fold and p.pass_dependencies_and_original_row_generation_stable) and (p.final_threshold_cohort_inclusive_and_original_coordinates_preserved and p.retained_mass_exceeds_native_normalization_floor and p.output_cast_and_normalization_error_admitted)`.

- Precondition `pre_values` (implementation): `p.finite_nonnegative_values and p.positive_finite_total_mass and p.mass_fraction_in_open_zero_closed_one` — Values are finite nonnegative masses; total W>0 and requested fraction0<p<=1 define target mass p*W, which stays positive finite.
- Precondition `pre_keys` (implementation): `p.descending_key_encoding_monotone_and_equal_values_share_key and p.zero_and_special_key_policy_explicit and p.selected_key_width_has_implemented_backend` — A fixed16/32-bit sortable encoding orders positive values descending and groups equal values; zeros cannot replace a positive crossing and nonfinite values are excluded. The selected key width has an implemented native binding; source registration uses FP32 only.
- Precondition `pre_partition` (implementation): `p.each_active_occurrence_histogrammed_once and p.histogram_count_and_mass_domains_identical` — Every occurrence matching the accumulated key prefix contributes once to its digit mass and companion count, preserving repeated values.
- Precondition `pre_mass` (implementation): `p.mass_accumulation_prefix_scan_and_residual_subtraction_policy_explicit and p.native_mass_comparisons_preserve_requested_crossing` — The declared exact or admitted finite-precision sums and residual target subtraction choose the same requested first cumulative>= crossing; deterministic storage alone does not prove this.
- Precondition `pre_capacity` (implementation): `candidate_access != 'compact_if_fits' or (p.complete_active_count_known_before_compaction and p.compact_only_when_count_fits_capacity and p.overflow_rescans_original_without_dropping_values)` — An optional compact table is used only after its complete incoming prefix count fits the buffer; otherwise rescan original input. Never truncate an overflowing active prefix.
- Precondition `pre_cache` (implementation): `candidate_access != 'compact_if_fits' or (p.compact_values_preserve_occurrence_multiplicity_and_prefix and p.compact_bank_publication_and_lifetime_safe)` — Compact buffers contain every incoming-prefix occurrence, including duplicates, and publish before later passes. Physical order may vary because the final apply pass rereads original coordinates.
- Precondition `pre_lifetime` (implementation): `p.histograms_and_counters_initialized_per_row_pass and p.all_producer_partials_published_before_boundary_fold and p.pass_dependencies_and_original_row_generation_stable` — Each pass reads a complete current histogram/count table, with reset, producer completion and pass/kernel ordering; original data/metadata remain immutable through final apply.
- Precondition `pre_apply` (implementation): `p.final_threshold_cohort_inclusive_and_original_coordinates_preserved and p.retained_mass_exceeds_native_normalization_floor and p.output_cast_and_normalization_error_admitted` — Final output keeps every original value>=threshold and divides by the complete retained mass under the admitted cast policy. Source fallback at retained mass<=1e-8 is outside this normalized binding.
- Constraint `c_shape` (implementation): `p.length >= 1 and p.histogram_bins >= 2048 and p.counts_and_indices_fit_int32 and p.candidate_capacity >= 0` — Nonempty row, bins for11-bit digits, exact bounded counters and optional nonnegative compact capacity.

## Local example and misuse

For a local instance of `reduce`, identify the object `weighted_radix_mass_crossing` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/mass_rank_radix_boundary@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/mass_rank_radix_boundary@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Mass/count histograms, residual target/prefix, optional two capacity-bounded value banks and final original-row output.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: candidate_banks
  level: dram
  scope: kernel
  bytes: 2 * p.candidate_capacity * p.element_bytes
  when: candidate_access == 'compact_if_fits'
overlap: []
```

## Obligations and numerical checks

- `weighted_prefix` (proof): At each digit, earlier-bin mass plus residual target identifies the same global descending weighted crossing; companion counts never substitute for masses.
- `occurrence_and_capacity` (test): Every active occurrence contributes once even after optional compaction; known-count guards and original rescans prevent capacity loss.
- `cohort_apply` (test): The final decoded value identifies the complete equality cohort and the retained mass/output coordinates match the original row.
- `native_sum_and_completion` (test): Native sum reconstruction, comparison rounding, histogram reset and all-producer publication implement the admitted weighted crossing and normalization.

Floating-point atomic order or deterministic sum reconstruction may move a mass boundary. Losing a compact overflow, confusing candidate counts with mass, or splitting equality cohorts changes support and normalization.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
