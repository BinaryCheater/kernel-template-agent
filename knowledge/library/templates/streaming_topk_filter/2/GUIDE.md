# streaming_topk_filter@2: bounded topk candidates

## When to consider

When a long-row top-k scan has a certified running threshold, keep exact survivors plus bounded incomers and periodically reconstruct the exact top-k. Choose scan round4096/8192 and a compaction multiple2..32; capacity must fit the full next all-hit round. This extends the original2*k schedule while retaining arbitrary tie identity and complete scan obligations.

## Computation before and after

Threshold-filtered tile scan with bounded exact top-k compaction.

Initially retain every valid candidate until at least k exist. After each exact compaction retain k winners and a value threshold no greater than the least winner. Later input values at or below that threshold may be ignored because k retained values already dominate or tie them. Append all surviving value/index pairs; if count reaches compact_multiple*k, or final count exceeds k, compact the complete candidates back to k using an exact selector. The count before an append is below compact_multiple*k, so capacity >= compact_multiple*k+scan_tile admits an all-hit input tile. At end expose k winners. For sharded scans, merging each shard top-k into a final top-k is separate and must include every nonempty shard.

Recover the original computation: Restore discarded inputs; each has at least k already retained values no smaller than it, so some allowed global top-k is unchanged.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| scan_tile | compile / software | [4096, 8192] | Input values per scan round; concrete retained NVIDIA menus4096/8192. |
| compact_multiple | compile / software | {"lo": 2, "hi": 32} | Compact when total logical retained+incoming candidates reach this multiple ofk, or at the final round. |

## Coupled constraints and scope

One complete scalar row, scan tiles4096/8192 and configurable compaction multiple2..32. Retains exact unsorted top-k with arbitrary tied identities; resource and append capacity premises remain unchanged. SourceDeepSelect bindingsk512 usemultiple9 for incoming threshold4096 or3 for long-row early threshold1024;MAX_TOPK512 survivors are complete logical records in this binding. Output sorting and cluster final merge are separate refinements.

Scope predicate: `p.k > 0 and p.k <= p.length and p.capacity >= compact_multiple * p.k + scan_tile and p.scan_visits_each_input_once and p.valid_tail_masked and p.compactor_exact_topk and p.threshold_not_above_retained_min and p.value_key_filter_order_compatible and p.arbitrary_tie_identity_admitted and p.unsorted_output_admitted and p.nan_inputs_excluded and p.append_reservations_disjoint and p.append_and_compaction_synchronized and p.alias_lifetimes_valid`.

- Precondition `pre_traversal` (semantic): `p.scan_visits_each_input_once and p.valid_tail_masked` — Scan order is a bijection over the valid input tiles and excludes padded lanes.
- Precondition `pre_threshold` (semantic): `p.compactor_exact_topk and p.threshold_not_above_retained_min and p.value_key_filter_order_compatible` — Compaction retains exactly k best candidates and the subsequent filter threshold never exceeds the least retained value.
- Precondition `pre_ties` (semantic): `p.arbitrary_tie_identity_admitted and p.unsorted_output_admitted and p.nan_inputs_excluded` — The result is unsorted top-k with arbitrary tie identities and no NaNs.
- Precondition `pre_storage` (implementation): `p.append_reservations_disjoint and p.append_and_compaction_synchronized and p.alias_lifetimes_valid` — Candidates and identities stay paired across atomic reservation, compaction and scratch aliasing.
- Constraint `c_rank` (semantic): `p.k > 0 and p.k <= p.length` — At least k valid inputs exist.
- Constraint `c_capacity` (resource): `p.capacity >= compact_multiple * p.k + scan_tile` — The candidate buffer holds the pre-compaction count plus one all-hit tile.

## Local example and misuse

For a local instance of `reduce`, identify the object `bounded_topk_candidates` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/streaming_topk_filter@2/template.yaml`. Original outline: `knowledge/studies/sglang/delta/streaming_topk_filter@2/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: O(k+scan_tile) candidate value/index pairs per worker.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `winner_preservation` (proof): Every discarded value is dominated by k retained candidates under the tie policy.
- `capacity` (proof): Appending an all-hit tile never exceeds capacity and compaction keeps at least k winners.
- `storage` (test): Reservations and compaction preserve disjoint paired value/index records.

The comparison threshold must be conservative after dtype conversion; a rounded-up threshold can drop a winner.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
