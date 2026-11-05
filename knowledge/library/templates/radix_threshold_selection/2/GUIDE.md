# radix_threshold_selection@2: rank threshold

## When to consider

When unsorted top-k is needed from ordered scalar keys, refine the kth threshold through four- or eight-bit histograms rather than sorting all pairs. Digit width trades bin storage against pass count. Emit every greater key and exactly the remaining ties using the requested identity policy; a deterministic smallest-index request needs an ordered tie-emission binding.

## Computation before and after

MSB-first radix rank refinement plus policy-bound threshold-tie quota.

For an active key prefix and remaining rank r, count the next digit_bits-bit digit. Let C_ge(b) count digits at least b and C_gt(b) count digits strictly above b. Choose the boundary digit satisfying C_gt(b)<r<=C_ge(b). Append that digit to the prefix, subtract C_gt(b) from r, and refine only matching keys at the next digit. After the full key the threshold q satisfies count(key>q)<k<=count(key>=q). An early lower-edge threshold is permitted only when its complete bucket accounts for exactly the remaining rank; then count(key>q)<=k<=count(key>=q) still holds. Select every key>q and exactly k-count(key>q) keys equal to q. A preliminary coarse monotone histogram may discard whole worse buckets before complete-key refinement, but cannot decide ordering inside the boundary bucket. With tie_policy=smallest_index, choose the lowest logical candidate indices among equal-threshold keys; output need not be value-sorted. Packed histogram counters must not carry between fields before extraction and warp reduction.

Recover the original computation: Forget histograms and prefixes; output is a size-k subset with no excluded key strictly greater than any selected key.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| key_bits | compile / software | [16, 32] | Width of the complete monotone key: 16 or 32 bits. |
| digit_bits | compile / software | [4, 8] | Bits refined per pass; four bits use sixteen histogram bins and eight bits use 256 bins. |
| tie_policy | compile / boundary | ["any", "smallest_index"] | Requested tied-identity selection: arbitrary identities or lowest logical candidate indices. |

## Coupled constraints and scope

Exact unsorted top-k over complete monotone 16/32-bit keys with four- or eight-bit radix digits and explicit arbitrary/smallest-index threshold ties. No NaNs. TransformerEngine fused_router/utils.h uses four-bit digits for FP32, packed eight-bit per-thread bucket counters and at most 8160 experts to keep each local count <=255 before extracting and reducing bins. TileLang bindings remain eight-bit/any-tie instances. Counter widths, packing, subgroup ownership and complete-key conversion remain realization obligations; stable output ordering across unequal values is not supplied. If distinct signed-zero encodings share a numerical tie, a smallest-index value policy needs key canonicalization or a domain excluding that ambiguity; raw total-key order alone is insufficient.

Scope predicate: `p.k > 0 and p.k <= p.length and p.key_order_matches_values and p.nan_policy_excludes_inputs and p.valid_candidates_exact_once and p.histogram_counts_do_not_overflow and p.histogram_visible_before_rank_search and p.candidate_storage_sufficient and p.equal_key_quota_enforced and p.tie_emission_matches_requested_policy and p.output_positions_disjoint and key_bits % digit_bits == 0 and (tie_policy == 'any' or (p.threshold_ties_by_ascending_index and p.equal_value_ties_match_keys))`.

- Precondition `pre_keys` (semantic): `p.key_order_matches_values and p.nan_policy_excludes_inputs and p.valid_candidates_exact_once` — The complete key preserves the value ordering; this version excludes NaNs and ignores padding.
- Precondition `pre_hist` (implementation): `p.histogram_counts_do_not_overflow and p.histogram_visible_before_rank_search and p.candidate_storage_sufficient` — Histograms and boundary-bucket compaction preserve every candidate and are visible before consumption.
- Precondition `pre_ties` (implementation): `p.equal_key_quota_enforced and p.tie_emission_matches_requested_policy and p.output_positions_disjoint` — Emit the exact threshold quota using the requested tie policy into disjoint output positions.
- Constraint `c_rank` (semantic): `p.k > 0 and p.k <= p.length` — The requested rank exists in the valid candidate set.
- Constraint `c_digits` (implementation): `key_bits % digit_bits == 0` — Refinement covers the complete key with whole radix digits.
- Constraint `c_tie` (implementation): `tie_policy == 'any' or (p.threshold_ties_by_ascending_index and p.equal_value_ties_match_keys)` — A smallest-index request requires ordered threshold ties and matching key/value tie classes.

## Local example and misuse

For a local instance of `reduce`, identify the object `rank_threshold` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/transformerengine/delta/radix_threshold_selection@2/template.yaml`. Original outline: `knowledge/studies/transformerengine/delta/radix_threshold_selection@2/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Digit histograms and the active prefix or boundary candidates.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `rank_invariant` (proof): Discarded better buckets reduce the rank exactly and discarded worse buckets cannot contain a required key.
- `quota_and_placement` (test): All greater keys and exactly the necessary threshold ties are emitted once without collisions.
- `key_binding` (test): Value/key conversion and padding satisfy the declared total ordering and exceptional-value policy.

A nonmonotone compressed key or unhandled NaN can discard a true winner; coarse keys require complete-key refinement of the boundary bucket.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
