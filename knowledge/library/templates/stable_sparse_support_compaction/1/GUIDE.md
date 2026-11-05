# stable_sparse_support_compaction@1: stable valid occurrence prefix

## When to consider

When sparse rows contain invalid holes, consider packing valid occurrences into a stable prefix so consumers process the valid count. Apply caller slot limits and causal/index validity first at original positions, then prefix-rank the surviving occurrences. Keep duplicates and per-occurrence metadata, and pass the same prepared support to backward. Cutting the packed prefix again or deduplicating changes the operator.

## Computation before and after

Stable prefix rank of valid original occurrences, with a matching count or hole-preserving span.

For each original row evaluate its validity predicate at original slot s, including source index bounds, segment/causal identity and any caller position limit. With keep_positions, write the already translated valid key/metadata at slot s and an invalid sentinel otherwise; the processing span is last valid slot plus one, or zero. With stable_compact, let rank(s) be the number of valid original slots before s. Write each valid occurrence at rank(s) with its translated key and associated metadata, retain duplicates as separate entries, pad the remaining row invalid, and set processing length to total valid count. Apply no original position cut again to the packed slots. Publish row, length and metadata together. Every consumer, including backward, uses the same prepared representation or an explicitly equivalent occurrence map; keep original-slot labels if a consumer needs them. Do not sort by key ID or deduplicate as an incidental compaction step.

Recover the original computation: Expand valid packed positions through their original-slot occurrence order, restoring the original valid key/metadata sequence with holes ignored; identical key multiplicities and semantic support are recovered.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| support_layout | compile / software | ["keep_positions", "stable_compact"] | Keep invalid holes with last-valid span or pack valid occurrences with a count. |

## Coupled constraints and scope

Fixed-capacity sparse occurrence rows with explicit invalidity and stable relative order. Source DSA training prepares global key rows from document-local inputs, applying original-slot topk_length and optional causal support before compaction. Forward/backward retain the prepared rows and lengths. No sorting/deduplication, gradients through discrete selection, or automatic equality of native tiled rounding is supplied.

Scope predicate: `p.validity_decided_in_original_slot_coordinates and p.bounds_segment_causal_and_position_cuts_applied_before_compaction and p.valid_occurrences_keep_original_relative_order_and_multiplicity and p.key_translation_and_per_occurrence_metadata_follow_same_map and p.all_consumers_use_transformed_rows_and_matching_length_definition and p.invalid_padding_never_contributes and p.prepared_rows_lengths_and_metadata_published_before_consumers_and_immutable_until_last_reader and p.changed_tile_grouping_and_reduction_order_admitted and p.selected_compaction_kernel_and_capacity_legal`.

- Precondition `pre_validity` (semantic): `p.validity_decided_in_original_slot_coordinates and p.bounds_segment_causal_and_position_cuts_applied_before_compaction` — All invalidity and position-prefix decisions refer to original slots and original query/key identities.
- Precondition `pre_order` (semantic): `p.valid_occurrences_keep_original_relative_order_and_multiplicity` — Retain every valid occurrence including repeated key IDs, in stable original order.
- Precondition `pre_metadata` (semantic): `p.key_translation_and_per_occurrence_metadata_follow_same_map` — Translated key IDs and any slot-dependent metadata use the same stable map.
- Precondition `pre_consumers` (implementation): `p.all_consumers_use_transformed_rows_and_matching_length_definition and p.invalid_padding_never_contributes` — Forward/backward and other consumers agree on row contents, count versus span, and invalid padding.
- Precondition `pre_publish` (implementation): `p.prepared_rows_lengths_and_metadata_published_before_consumers_and_immutable_until_last_reader` — The complete transformed support generation survives all consumers, including deferred backward.
- Precondition `pre_numeric` (implementation): `p.changed_tile_grouping_and_reduction_order_admitted and p.selected_compaction_kernel_and_capacity_legal` — Stable logical order does not itself establish bitwise equality after changed tile grouping; native capacity/scan legality is explicit.
- Constraint `c_capacity` (resource): `p.row_count >= 0 and p.row_capacity >= 1 and p.prepared_row_capacity >= p.row_capacity` — Prepared rows can hold every valid occurrence and padding.

## Local example and misuse

For a local instance of `work_assign`, identify the object `stable_valid_occurrence_prefix` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/stable_sparse_support_compaction@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/stable_sparse_support_compaction@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Sparse consumers skip trailing invalid work and avoid carrying holes into later tiles.. Storage: Prepared indexed support, optional occurrence metadata and one matching row length.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: prepared_support
  level: dram
  scope: kernel
  bytes: p.row_count * (p.row_capacity * p.prepared_occurrence_bytes + p.length_bytes)
overlap: []
```

## Obligations and numerical checks

- `stable_occurrence_bijection` (proof): Valid original occurrences map one-to-one in ascending slot order to packed positions, retaining repeated IDs.
- `validity_and_metadata` (test): All original-position decisions precede packing and translated keys plus slot-dependent metadata follow the same map.
- `span_count_and_consumers` (test): Hole-preserving span and packed count are not interchanged; every forward/backward consumer reads the same prepared support generation.
- `publication_and_numeric` (test): Scan/write bounds, publication lifetime and changed tile grouping satisfy native and numerical policy.

Stable occurrence order preserves mathematical support, but packing changes tile boundaries and can change online max/reduction association and low-precision rounding.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
