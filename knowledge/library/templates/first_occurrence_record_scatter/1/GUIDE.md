# first_occurrence_record_scatter@1: unique first input owner of scattered records

## When to consider

When duplicate destination indices append multi-field records, elect the earliest input before writing any field. Scan prior indices for small batches or reset/atomic-min owners before the writer pass. This keeps one input responsible for codes, scales and metadata together. It supplies deterministic first-occurrence semantics, not additive scatter or atomic publication to concurrent readers; consumers still wait for the complete writer grid.

## Computation before and after

Elect a lowest-index input per destination before writing a multi-field record.

Input row i proposes destination slot[i]. Predicate invalid destinations before address calculation. For earlier_scan, cooperatively compare this slot with every earlier input row and uniformly skip the record if an equal destination exists. Otherwise encode/write every payload and metadata field of row i. For atomic_min_passes, first reset touched valid slots owner words to UINT32_MAX; complete reset before any claim. Atomically minimize each slot owner with its input index; complete all claims before any writer checks ownership. Only i==owner[slot[i]] may encode/write that entire record. Owner storage may use ABI padding only if previous readers retired, restoration cannot create a false later winner, and required padding bits are restored before any consumer. Unmapped records stay unchanged. Wait for all record writers before consumers, since one elected writer does not make multiple field stores transactionally atomic.

Recover the original computation: For each valid destination choose its lowest-index proposing input once and apply the original complete record encoder; all other proposals and invalid mappings have no output effect.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| election | compile / software | ["earlier_scan", "atomic_min_passes"] | Scan earlier input destinations in one writer launch or elect per-slot minimum input IDs across ordered passes. |

## Coupled constraints and scope

Deterministic first-input-wins scatter of independent complete records, with nonnegative valid destination slots and invalid-slot predication. Source SM120 quantized MLA cache append has warp-parallel duplicate scans; DSV4 NVFP4 additionally uses uint32 owner election in four reserved zero-padding bytes. The owner storage expression is logical capacity, possibly aliased with padding. Atomic visibility, encoder precision, overlapping invocations and concurrent record readers require explicit bindings.

Scope predicate: `(p.lowest_input_index_wins_requested and p.slot_map_immutable_through_all_passes) and (p.same_slot_means_same_complete_record and p.winner_owns_all_payload_metadata_fields) and (p.invalid_slots_predicated_before_record_addressing and p.record_address_map_injective_for_distinct_valid_slots) and (election != 'earlier_scan' or p.all_writer_lanes_agree_on_earlier_duplicate_result) and (election != 'atomic_min_passes' or (p.owner_words_aligned_and_exclusive and p.reset_complete_before_claims and p.atomic_min_claims_complete_before_winner_reads and p.owner_restore_cannot_create_another_winner)) and (p.previous_readers_retired_before_owner_scratch_or_payload_writes and p.complete_writer_grid_before_record_consumers and p.all_scratch_readers_retired_before_scratch_reuse) and (p.selected_record_encoder_and_capacity_supported and p.record_numeric_policy_admitted)`.

- Precondition `pre_contract` (semantic): `p.lowest_input_index_wins_requested and p.slot_map_immutable_through_all_passes` — Repeated destinations request first input occurrence, not addition or last-writer timing.
- Precondition `pre_records` (semantic): `p.same_slot_means_same_complete_record and p.winner_owns_all_payload_metadata_fields` — One elected input owns every code, scale and side field of a destination record.
- Precondition `pre_bounds` (implementation): `p.invalid_slots_predicated_before_record_addressing and p.record_address_map_injective_for_distinct_valid_slots` — Invalid slots do not form payload addresses; distinct destinations have nonoverlapping complete records.
- Precondition `pre_scan` (implementation): `election != 'earlier_scan' or p.all_writer_lanes_agree_on_earlier_duplicate_result` — The earlier-occurrence scan produces one uniform write/no-write decision for all lanes of a record.
- Precondition `pre_atomic` (implementation): `election != 'atomic_min_passes' or (p.owner_words_aligned_and_exclusive and p.reset_complete_before_claims and p.atomic_min_claims_complete_before_winner_reads and p.owner_restore_cannot_create_another_winner)` — Ordered reset/claim/write passes share exclusive aligned owner words and restoration cannot let another candidate qualify.
- Precondition `pre_lifetime` (implementation): `p.previous_readers_retired_before_owner_scratch_or_payload_writes and p.complete_writer_grid_before_record_consumers and p.all_scratch_readers_retired_before_scratch_reuse` — Readers see only completed records; temporary owner bits are not published as valid cache metadata.
- Precondition `pre_encode` (implementation): `p.selected_record_encoder_and_capacity_supported and p.record_numeric_policy_admitted` — The winning input uses the declared complete record encoder and numerical contract.
- Constraint `c_index` (implementation): `p.input_count >= 0 and p.input_count < 4294967295` — Unsigned 32-bit owner IDs reserve all-ones as the reset sentinel.
- Constraint `c_owner` (implementation): `election != 'atomic_min_passes' or p.owner_word_bytes == 4` — Atomic-min election uses an aligned 32-bit owner word.

## Local example and misuse

For a local instance of `store`, identify the object `unique_first_input_owner_of_scattered_records` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/first_occurrence_record_scatter@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/first_occurrence_record_scatter@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Use a one-pass earlier scan for small appends or ordered atomic-min election to avoid quadratic comparisons.. Storage: Earlier scan needs only local duplicate flags; atomic path needs one exclusive 32-bit owner per addressable slot, possibly existing padding.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: owner_words
  level: dram
  scope: kernel
  bytes: 4 * p.owner_slot_capacity
  when: election == 'atomic_min_passes'
overlap: []
```

## Obligations and numerical checks

- `unique_first_owner` (proof): Both election methods select exactly the minimum proposing input ID for every touched valid destination.
- `complete_record` (test): All payload and metadata fields come from that elected input; distinct destinations do not overlap.
- `pass_and_scratch_order` (test): Reset, all claims, winner reads, optional owner restoration and later consumers have the declared order and exclusivity.
- `invalid_and_unmapped` (test): Invalid inputs never address records and untouched destinations retain their previous contents.

The encoder is unchanged; selecting a different occurrence would change the data and scales. Interleaving multiple writers can tear a record even if each individual field store is atomic.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
