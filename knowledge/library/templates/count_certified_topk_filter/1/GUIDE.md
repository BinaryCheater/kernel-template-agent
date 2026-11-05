# count_certified_topk_filter@1: bounded threshold candidates with exact fallback

## When to consider

When exact top-k scans have a cheap threshold estimate, retain scores at or above it in a bounded candidate buffer. Accept only after the full qualifier count is at least the requested winner count and fits the buffer. This certificate makes threshold quality a cost question: too few or too many candidates trigger exact full-domain selection. Continue counting after storage fills; selecting a truncated buffer is not exact.

## Computation before and after

Threshold scan with full-count acceptance certificate and exact full-domain fallback.

Choose any non-NaN row score threshold tau, for example from a sample. Scan the entire original valid domain exactly once. For each score >= tau atomically reserve a unique append position and increment the total qualifier count; write the score/identity only when that position is below capacity. After all scan writes and counts complete, let want=min(k,valid_count) and c be the full count, not the number of stored records. If want<=c<=capacity, select the exact top want records from the complete candidate list using the original total order. Otherwise discard this list, flag the row and run an exact selector over the entire original valid domain. Do not select from truncated overflow records or pad an underfull candidate set without fallback. For zero valid inputs emit the defined empty result; initialize all unused output slots according to the caller contract. Reuse candidate storage as fallback histograms only after every old candidate reader retires.

Recover the original computation: Every accepted candidate set contains all original top-want winners; rejected sets are replaced by exact full-domain selection, so final membership and tie order match global exact top-k.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| capacity | compile / software | {"lo": 1, "hi": 1048576} | Maximum candidate records retained per row; actual count continues past this limit. |
| k | runtime / boundary | {"lo": 1, "hi": 4096} | Requested exact top-k slot count, fixed by the workload. |

## Coupled constraints and scope

Exact finite-score top-k with unique identities and a complete inclusive scalar-threshold filter. Sampling changes work only, not accepted membership. Source SM120 long ragged DSA uses capacity 8192, requested k<=4096 and a sample-derived threshold. Threshold estimation, exact fallback implementation and native publication are separate bindings. Approximate top-k, hidden count saturation, changed fallback arithmetic and predicates unrelated to the ranking score are excluded.

Scope predicate: `p.finite_scores_unique_ids_and_defined_tie_order and p.invalid_slots_excluded_before_scan and p.full_valid_domain_scanned_once_and_filter_is_score_ge_threshold and p.threshold_not_nan and p.count_includes_all_qualifiers_without_overflow and p.atomic_append_slots_unique and p.writes_bounded_by_capacity and p.score_bits_and_identity_match_between_scan_and_exact_fallback and p.candidate_and_fallback_selectors_exact_for_same_requested_order and p.output_presentation_and_padding_preserve_identity and p.scan_and_count_complete_visible_before_acceptance and p.fallback_flags_complete_before_rescan and p.scratch_not_reused_until_all_readers_retire`.

- Precondition `pre_order` (semantic): `p.finite_scores_unique_ids_and_defined_tie_order and p.invalid_slots_excluded_before_scan` — Original valid identities have finite scores and one deterministic total score/identity order.
- Precondition `pre_filter` (semantic): `p.full_valid_domain_scanned_once_and_filter_is_score_ge_threshold and p.threshold_not_nan` — Retain every valid score greater than or equal to one row threshold; no winning equality or valid region is silently omitted.
- Precondition `pre_append` (implementation): `p.count_includes_all_qualifiers_without_overflow and p.atomic_append_slots_unique and p.writes_bounded_by_capacity` — Count all qualifying identities even beyond the storage bound, with unique append slots and bounded writes.
- Precondition `pre_score` (semantic): `p.score_bits_and_identity_match_between_scan_and_exact_fallback` — The complete fallback evaluates or reads exactly the same logical score bits and identity order as the first scan.
- Precondition `pre_select` (semantic): `p.candidate_and_fallback_selectors_exact_for_same_requested_order and p.output_presentation_and_padding_preserve_identity` — Both selectors implement exact requested membership/ties; result ordering and padding follow the caller contract.
- Precondition `pre_lifetime` (implementation): `p.scan_and_count_complete_visible_before_acceptance and p.fallback_flags_complete_before_rescan and p.scratch_not_reused_until_all_readers_retire` — Acceptance sees final counts and payloads; fallback flags publish before their consumer, and scratch survives all phases.
- Constraint `c_capacity` (resource): `capacity >= k and p.allocated_candidate_capacity >= capacity` — Candidate storage can retain at least the requested winners.
- Constraint `c_count` (implementation): `p.valid_count >= 0 and p.valid_count <= p.max_safe_count` — All qualifier counts fit the counter type, including a fully qualifying row.

## Local example and misuse

For a local instance of `reduce`, identify the object `bounded_threshold_candidates_with_exact_fallback` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/count_certified_topk_filter@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/count_certified_topk_filter@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Most rows may finalize a bounded candidate list; unfavorable thresholds affect rescans rather than correctness.. Storage: Bounded score/ID candidates, untruncated counts and fallback row flags.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: candidate_records
  level: dram
  scope: kernel
  bytes: p.row_count * capacity * (p.score_bytes + p.id_bytes)
- id: counts_and_flags
  level: dram
  scope: kernel
  bytes: p.row_count * (p.counter_bytes + p.flag_record_bytes)
overlap: []
```

## Obligations and numerical checks

- `filter_certificate` (proof): A complete threshold-qualified set with at least want entries contains every top-want winner.
- `count_and_storage` (test): Qualifier count is complete and untruncated; every accepted candidate record is stored exactly once and overflow writes are suppressed.
- `fallback_identity` (test): Underfull or overflowing rows execute exact full-domain fallback with unchanged scores, IDs, masks and ties.
- `phase_and_output` (test): Acceptance/fallback decisions observe completed generations and produce the requested ordering/padding without scratch reuse races.

Any changed upstream rounding between scan and fallback can alter membership/ties even if both local selectors are exact.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
