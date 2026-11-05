# radix_boundary_topk@1: score boundary and tied identity cut

## When to consider

When exact top-k needs more than a small sorted-run merge, find the boundary score with most-significant-first radix histograms. Carry the rank remaining inside the chosen prefix, then choose the required largest original IDs only among equal-score entries. This avoids a whole-row sort while retaining exact deterministic membership. Output may be presented in ascending ID order even though ranking favors larger IDs on ties.

## Computation before and after

Radix selection of exact score boundary followed by equal-score ID boundary.

Map each finite FP32 score s to a monotone unsigned 32-bit order key: fold either signed zero to positive zero, complement bits for negative values, otherwise set the sign bit. For want=min(k,valid_count)>0, start with empty prefix and residual rank want. For each digit from most to least significant, count only keys matching the prior prefix, scan bins in descending key order, choose the bin containing the residual rank, subtract the counts of strictly higher bins and append that digit to the prefix. After all 32 bits the prefix is the exact boundary score key and residual rank is how many equal-score entries belong among the winners. Apply the same descending radix procedure to the original IDs of only those boundary-score entries, obtaining the residual-rank-th largest ID. Emit every key above the score boundary plus boundary keys whose ID is at least that ID cut. If all tied entries are needed, skip the ID passes. A surrounding bounded candidate stage may instead retain all boundary ties when they fit, but must eventually apply this exact ID decision before output. Arrange the selected set in the caller requested order, such as ascending ID or unspecified set order; never confuse that presentation with the descending ranking comparator. Empty domains emit padding without requesting a nonexistent rank.

Recover the original computation: The score boundary and tied-ID cut represent the first min(k,valid_count) identities under descending score then descending original ID, with signed zeros equal.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| digit_schedule | compile / software | ["bytes_8_8_8_8", "bits_11_11_10"] | Most-significant-first partition of the 32-bit score/ID key; derive shifts and masks from this schedule. |
| k | runtime / boundary | {"lo": 1, "hi": 4096} | Requested exact winner slots; actual winner count is min(k,valid_count). |

## Coupled constraints and scope

Finite FP32 scores with distinct nonnegative int32 IDs and larger-ID tie breaking. Four byte passes and 11/11/10-bit passes are source-bound schedules; bins, scans and publication have concrete backend obligations. Source ragged SM120 DSA presents final IDs ascending. Decode selection has a different presentation and coarse-histogram/refinement realization. NaNs, duplicate IDs, arbitrary comparison transforms and smallest-ID ties are outside this version.

Scope predicate: `p.finite_fp32_scores_and_unique_nonnegative_int32_ids and p.score_key_encoding_monotone_with_signed_zero_fold and p.valid_domain_count_exact_and_invalids_excluded and p.histograms_cover_each_active_identity_once and p.histogram_and_prefix_counts_fit_counter_type and p.histogram_generation_zeroed_and_complete_before_rank_selection and p.every_participant_uses_same_prefix_rank_and_generation and p.phase_state_and_histograms_live_until_readers_retire and p.emit_exact_score_greater_or_boundary_id_ge_cut and p.output_slots_unique_and_presentation_padding_defined and p.selected_radix_schedule_implemented_and_scratch_capacity_fits`.

- Precondition `pre_encoding` (semantic): `p.finite_fp32_scores_and_unique_nonnegative_int32_ids and p.score_key_encoding_monotone_with_signed_zero_fold` — FP32 scores map to monotone unsigned 32-bit keys, folding signed zero; valid IDs are unique nonnegative signed-32-bit values.
- Precondition `pre_domain` (semantic): `p.valid_domain_count_exact_and_invalids_excluded and p.histograms_cover_each_active_identity_once` — Every histogram counts exactly the current valid prefix set, with no invalid or duplicate contributions.
- Precondition `pre_counters` (implementation): `p.histogram_and_prefix_counts_fit_counter_type and p.histogram_generation_zeroed_and_complete_before_rank_selection` — Bin counts fit; each new histogram generation is cleared and complete before locating the rank.
- Precondition `pre_state` (implementation): `p.every_participant_uses_same_prefix_rank_and_generation and p.phase_state_and_histograms_live_until_readers_retire` — All participants share one prefix/residual-rank state and preserve each phase data until its readers finish.
- Precondition `pre_emit` (semantic): `p.emit_exact_score_greater_or_boundary_id_ge_cut and p.output_slots_unique_and_presentation_padding_defined` — Emit only scores above the boundary and the largest required tied IDs; output allocation/order/padding is defined.
- Precondition `pre_backend` (implementation): `p.selected_radix_schedule_implemented_and_scratch_capacity_fits` — The selected digit schedule and any distributed combine/publication path are implemented and fit their storage.
- Constraint `c_histogram` (resource): `(digit_schedule == 'bytes_8_8_8_8' and p.histogram_bins >= 256) or (digit_schedule == 'bits_11_11_10' and p.histogram_bins >= 2048)` — Allocate enough bins for the largest digit.
- Constraint `c_storage` (resource): `p.valid_count >= 0 and p.output_capacity >= k and p.histogram_bytes >= 4 * p.histogram_bins` — Counts and output/histogram capacity are valid.

## Local example and misuse

For a local instance of `reduce`, identify the object `score_boundary_and_tied_identity_cut` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/radix_boundary_topk@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/radix_boundary_topk@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Avoid sorting the whole score domain; only the unresolved prefix is counted each pass.. Storage: Digit histogram plus prefix/rank state and optional distributed row histograms; emission buffering is a separate binding.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: radix_histograms
  level: smem
  scope: cta
  bytes: 4 * p.histogram_bins
- id: distributed_histograms
  level: dram
  scope: kernel
  bytes: p.distributed_histogram_bytes
overlap: []
```

## Obligations and numerical checks

- `rank_prefix` (proof): Every prefix step retains exactly the bin containing the unresolved rank and subtracts exactly the better-key count.
- `tie_identity` (proof): Only equal-boundary-score IDs enter the tie selector and the final cut retains exactly the required largest distinct IDs.
- `histogram_generation` (test): Local/distributed counts, phase completion and reset preserve one consistent prefix/rank generation.
- `encoding_and_output` (test): Encoding matches finite FP32 order including signed zeros; emitted IDs/scores remain aligned with unique slots and defined presentation/padding.

Selection introduces no new score rounding; score producer changes and signed-zero/NaN policy can change the order before selection.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
