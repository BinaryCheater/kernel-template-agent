# witness_bootstrapped_sharded_topk@1: witness bootstrapped sharded topk

## When to consider

When one row is split among workers that initially hold fewer than k local winners, obtain k global rank witnesses from a shared bootstrap sample before filtering each shard. Raise a local threshold only after that worker has its own k retained witnesses.

## Computation before and after

When one row is split among workers that initially hold fewer than k local winners, obtain k global rank witnesses from a shared bootstrap sample before filtering each shard. Raise a local threshold only after that worker has its own k retained witnesses.

Select exact global top-k from the union of initial per-worker windows and distribute those winner records to their original workers. Let b be its kth value. For the rest of each disjoint shard retain records strictly greater than b; discarded ties are permitted because the retained bootstrap winners already supply k witnesses. A worker can compact its complete local candidates to k once at least k exist; thereafter it may raise its filter threshold to the least local winner. Discarding below either threshold is safe: the bootstrap threshold has k global witnesses, and any raised local threshold has k local witnesses. Finally select exact global top-k over every retained worker candidate, including workers that never reached k locally.

Recover the original computation: Reinsert each discarded record: its discard had k distinct actually retained-or-dominating witnesses in the global union. Local exact top-k substitution preserves the global top-k value multiset, so final selection returns an allowed full-row top-k.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| workers | compile / software | [2, 3, 4, 8, 16] | Disjoint workers cooperate on one row. |
| bootstrap_per_worker | compile / software | [32768] | Maximum initial sample records per worker; a shorter slice contributes its complete contents. |

## Coupled constraints and scope

Pinned DeepSelect BF16 cluster2/3/4/8/16, initial window32768 per CTA, k<=MAX_TOPK512/1024, arbitrary ties and complete disjoint slices. Source NaN handling is separate. Bootstrap windows visit the only partial tail first; short slices use the direct complete-window branch. Local candidate buffer4096 and input stages8192 use reservation rollback before exact compaction; that independent storage choice is reservation_rollback_retry@1.

Scope predicate: `(p.k > 0 and p.k <= p.length and p.shards_partition_valid_row and p.bootstrap_records_distinct_and_part_of_row) and (p.bootstrap_contains_at_least_k_valid_records and p.bootstrap_global_topk_exact) and (p.bootstrap_winners_retained_with_exact_tie_quota and p.all_later_inputs_scanned) and (p.local_threshold_raises_only_after_exact_local_topk and p.local_topk_inputs_complete) and (p.final_selection_covers_all_retained_workers and p.arbitrary_tie_identity_admitted and p.nan_inputs_excluded)`.

- Precondition `pre_0` (implementation): `p.k > 0 and p.k <= p.length and p.shards_partition_valid_row and p.bootstrap_records_distinct_and_part_of_row` — The requested rank exists; bootstrap records are not duplicated or invented.
- Precondition `pre_1` (implementation): `p.bootstrap_contains_at_least_k_valid_records and p.bootstrap_global_topk_exact` — Initial global selection supplies actual k witnesses and exact kth threshold.
- Precondition `pre_2` (implementation): `p.bootstrap_winners_retained_with_exact_tie_quota and p.all_later_inputs_scanned` — At least one allowed bootstrap top-k remains represented; each later valid record is considered once.
- Precondition `pre_3` (implementation): `p.local_threshold_raises_only_after_exact_local_topk and p.local_topk_inputs_complete` — A worker may replace its retained records by exact local top-k only after at least k actual records are available.
- Precondition `pre_4` (implementation): `p.final_selection_covers_all_retained_workers and p.arbitrary_tie_identity_admitted and p.nan_inputs_excluded` — Final global top-k is exact over the complete retained union; arbitrary tied identities are legal.

## Local example and misuse

For a local instance of `reduce`, identify the object `witness_bootstrapped_sharded_topk` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/witness_bootstrapped_sharded_topk@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/witness_bootstrapped_sharded_topk@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Candidate records, input stages and per-worker summaries with explicit live intervals.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `logical_equivalence` (proof): Reinsert each discarded record: its discard had k distinct actually retained-or-dominating witnesses in the global union. Local exact top-k substitution preserves the global top-k value multiset, so final selection returns an allowed full-row top-k.
- `protocol` (test): Native participants, counters, visibility and scratch lifetimes implement the declared complete dataflow.

Ordered comparisons must agree across thresholds, local compaction and final selection; finite ties use the declared arbitrary-identity policy.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
