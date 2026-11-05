# grouped_sparse_union_masks@1: shared key union with row membership masks

## When to consider

When neighboring sparse rows share many keys and the same cache mapping, build one logical-key union and a membership bitmask for each row. Load union payloads once but mask each row before its own normalization. Bitmap union removes duplicate keys, so use this only for set support. Keep mask columns tied to logical-key rank even if page translation makes physical slots unsorted, and clear row staging before reuse.

## Computation before and after

Bitmap key union, logical-key rank and per-row membership masks for shared sparse payloads.

Mark every valid logical key selected by any grouped row in a zeroed bitmap; repeated keys set the same bit. Prefix-scan bitmap popcounts to rank all union keys in ascending logical-key order. For union column c with key k, translate k through the common page table and write its physical payload slot once; publish union size U. For each original row, set bit c iff it selected key k. Initialize reusable row-mask staging, complete all writers, write whole 128-column mask tiles through ceil(U/128), and clear staging before another row reuses it. A masked dense consumer loads shared union payloads but applies each query row mask before exponentiation/normalization, retaining one original row normalizer. Only union[0:U] and mask tiles up to that bound are valid; capacity tails remain outside the contract. Suppressed output row IDs skip stores while all roles still follow bitmap/scan/staging synchronization.

Recover the original computation: For each row select the union keys whose membership bits are set and undo the common page mapping; the original valid key set is recovered without changing that row normalization support.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| group_rows | compile / software | {"lo": 1, "hi": 1024} | Number of consecutive rows sharing one union; actual bitmap/staging capacity limits the selected implementation. |

## Coupled constraints and scope

Set-valued sparse rows sharing one logical request/page table and payload generation. Source token_sparse.py uses PAGE64, 128-column mask tiles, 512-thread CTAs with four reusable row teams, and padded support widths multiple512. The group may span more rows than teams. Arbitrary occurrence multiplicity, incompatible page tables, score-dependent dynamic selection and reading unwritten capacity tails are outside this version.

Scope predicate: `p.support_has_set_semantics_and_duplicate_collapse_admitted and p.group_rows_share_logical_key_domain_page_table_and_payload_generation and p.invalid_key_positions_excluded_before_bit_or_page_access and p.page_table_entries_and_offsets_in_bounds and p.union_columns_rank_logical_keys_and_all_row_masks_use_same_rank and p.consumer_applies_row_mask_before_its_own_normalization and p.output_union_and_mask_row_ids_unique_or_exclusive and p.negative_output_ids_skip_only_stores_and_preserve_control and p.bitmap_and_staging_zeroed_before_reuse and p.union_masks_counts_published_before_consumer_and_live_through_reads and p.consumer_reduction_order_and_padding_contract_admitted and p.selected_group_bitmap_capacity_and_scan_legal`.

- Precondition `pre_set` (semantic): `p.support_has_set_semantics_and_duplicate_collapse_admitted` — Each row denotes set membership, not repeated occurrence weights.
- Precondition `pre_payload` (semantic): `p.group_rows_share_logical_key_domain_page_table_and_payload_generation` — Equal logical keys denote the same K/V payload for every grouped row.
- Precondition `pre_valid` (implementation): `p.invalid_key_positions_excluded_before_bit_or_page_access and p.page_table_entries_and_offsets_in_bounds` — Invalid keys never address bitmap/pages; valid translations fit allocated payloads.
- Precondition `pre_masks` (semantic): `p.union_columns_rank_logical_keys_and_all_row_masks_use_same_rank and p.consumer_applies_row_mask_before_its_own_normalization` — Membership masks use the exact union order; sharing payloads never merges row support or row normalizers.
- Precondition `pre_maps` (implementation): `p.output_union_and_mask_row_ids_unique_or_exclusive and p.negative_output_ids_skip_only_stores_and_preserve_control` — Visible row IDs have unique/exclusive writers and suppressed stores do not remove required barriers.
- Precondition `pre_lifetime` (implementation): `p.bitmap_and_staging_zeroed_before_reuse and p.union_masks_counts_published_before_consumer_and_live_through_reads` — Group bitmap and reusable per-team mask staging have clean generations; consumers wait for complete metadata.
- Precondition `pre_numeric` (implementation): `p.consumer_reduction_order_and_padding_contract_admitted and p.selected_group_bitmap_capacity_and_scan_legal` — Changed logical key order and tile padding satisfy numerical policy and actual shared/scan capacity.
- Constraint `c_rows` (semantic): `group_rows >= 1 and p.logical_key_count >= 1 and p.row_capacity >= 1` — Nonempty logical key universe and support storage; each row may select zero keys.
- Constraint `c_output` (resource): `p.union_capacity >= min(group_rows * p.row_capacity, p.logical_key_count) and p.mask_capacity_bits >= 128 * ((p.union_capacity + 127) // 128)` — Output capacities cover the maximum union and full 128-column mask tiles.
- Constraint `c_scratch` (resource): `p.bitmap_and_scan_staging_bytes <= p.available_shared_bytes` — The selected key-universe bitmap, prefix state and row-mask stages fit.

## Local example and misuse

For a local instance of `work_assign`, identify the object `shared_key_union_with_row_membership_masks` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/grouped_sparse_union_masks@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/grouped_sparse_union_masks@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: One union payload tile serves neighboring rows with overlapping support while each row retains its own mask.. Storage: Logical key-universe bitmap plus prefix state, reusable row-mask staging, union slots/count and output masks.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: shared_union
  level: dram
  scope: kernel
  bytes: 4 * p.union_capacity + p.union_count_bytes
- id: row_masks
  level: dram
  scope: kernel
  bytes: group_rows * p.mask_capacity_bits // 8
- id: bitmap_staging
  level: smem
  scope: cta
  bytes: p.bitmap_and_scan_staging_bytes
overlap: []
```

## Obligations and numerical checks

- `union_rank` (proof): Every selected logical key has exactly one ascending union column and no unselected key appears.
- `membership` (test): Each row mask represents exactly its original key set under the same column rank; duplicates are collapsed only under the set contract.
- `payload_identity` (test): Page translation and row-mask ranks preserve logical key identity even when physical slots are nonmonotone.
- `staging_and_consumer` (test): Prefix and row-mask stages publish/reset safely; final consumers apply the proper row mask, bounds and lifetime before normalization.

Ascending logical-key enumeration can reorder an unsorted input reduction. Collapsing duplicate occurrences changes attention mass unless set semantics were requested.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
