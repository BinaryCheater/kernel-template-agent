# radix_threshold_selection@1: rank threshold

## When to consider

When unsorted top-k is needed from ordered scalar keys, select the kth threshold by most-significant-byte histograms instead of sorting every pair. Refine only the boundary bucket and preserve the remaining rank. Emit all greater keys plus a quota of equal keys; output order and tie stability are separate contracts.

## Computation before and after

MSB-first rank refinement plus greater/equal quota selection.

For an active key prefix and remaining rank r, count the next eight-bit digit. Let C_ge(b) count digits at least b and C_gt(b) count digits strictly above b. Choose the boundary digit satisfying C_gt(b)<r<=C_ge(b). Append that digit to the prefix, subtract C_gt(b) from r, and refine only matching keys at the next digit. After the full key the threshold q satisfies count(key>q)<k<=count(key>=q). An early lower-edge threshold is permitted only when its complete bucket accounts for exactly the remaining rank; then count(key>q)<=k<=count(key>=q) still holds. Select every key>q and exactly k-count(key>q) keys equal to q. A preliminary coarse monotone histogram may discard whole worse buckets before complete-key refinement, but cannot decide ordering inside the boundary bucket.

Recover the original computation: Forget histograms and prefixes; output is a size-k subset with no excluded key strictly greater than any selected key.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| key_bits | compile / software | [16, 32] | Width of the complete monotone key; source BF16 uses 16 and FP32 uses 32. The radix digit width is fixed at eight bits. |

## Coupled constraints and scope

Exact unsorted top-k over complete 16/32-bit monotone keys, with eight-bit radix digits and arbitrary tie identity. TileLang deep_select supports BF16/FP32 and no NaNs; source deepseek_v32 uses a coarse FP16 high byte before refining a full FP32 key. Packed greater/equal prefix counters require each field below 65536; that limit belongs to the specific output-offset realization, not to every radix selector. Sorted or stable output needs another refinement.

Scope predicate: `p.k > 0 and p.k <= p.length and p.key_order_matches_values and p.nan_policy_excludes_inputs and p.valid_candidates_exact_once and p.histogram_counts_do_not_overflow and p.histogram_visible_before_rank_search and p.candidate_storage_sufficient and p.equal_key_quota_enforced and p.output_tie_policy_admits_any and p.output_positions_disjoint`.

- Precondition `pre_keys` (semantic): `p.key_order_matches_values and p.nan_policy_excludes_inputs and p.valid_candidates_exact_once` — The complete key preserves the value ordering; this version excludes NaNs and ignores padding.
- Precondition `pre_hist` (implementation): `p.histogram_counts_do_not_overflow and p.histogram_visible_before_rank_search and p.candidate_storage_sufficient` — Histograms and boundary-bucket compaction preserve every candidate and are visible before consumption.
- Precondition `pre_ties` (implementation): `p.equal_key_quota_enforced and p.output_tie_policy_admits_any and p.output_positions_disjoint` — Emit exactly the remaining number of threshold ties into disjoint positions; any tied identities are acceptable.
- Constraint `c_rank` (semantic): `p.k > 0 and p.k <= p.length` — The requested rank exists in the valid candidate set.

## Local example and misuse

For a local instance of `reduce`, identify the object `rank_threshold` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/tilelang/delta/radix_threshold_selection@1/template.yaml`. Original outline: `knowledge/studies/tilelang/delta/radix_threshold_selection@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

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
