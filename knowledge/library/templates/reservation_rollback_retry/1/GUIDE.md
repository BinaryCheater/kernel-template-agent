# reservation_rollback_retry@1: reservation rollback retry

## When to consider

When parallel workers append a variable number of records to bounded storage, reserve a whole batch speculatively, and on overflow roll back all reservations before retrying smaller units or compacting the complete still-live input.

## Computation before and after

When parallel workers append a variable number of records to bounded storage, reserve a whole batch speculatively, and on overflow roll back all reservations before retrying smaller units or compacting the complete still-live input.

Snapshot the conceptual committed prefix length c. Each participant reserves a disjoint range for its batch hits and only writes when that range fits the physical candidate buffer. Collectively wait until all claims and writes are complete. If every range fits, publish the appended count and release input stages. Otherwise sum every batch reservation, restore count=c, discard speculative suffix writes and retry each still-live stage. A stage that fits commits; a stage that does not fit has its reservations rolled back and is processed directly with the complete committed prefix by a semantics-preserving compactor. Only after the whole retry finishes may the input ring be refilled. Speculative writes beyond c are dead after rollback; they cannot overwrite the old prefix.

Recover the original computation: Every successful fast batch appends its records once; every failed batch has no visible committed effect before its original input is retried exactly once. Compaction consumes the full declared logical input, so physical candidate capacity never causes silent loss.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| batch_stages | compile / software | [4] | Input stages attempted together before a collective overflow decision. |

## Coupled constraints and scope

Concrete source has512 threads, one atomic per warp plus lane prefixes, four8192-value stages per batch,4096 packed candidate slots and exact top-k fallback. Failed batch retries stages; failed stage compaction loads old candidates and the full8192 stage into registers before writing back k records, with a census barrier covering all old reads. This is bounded storage protocol, not a substitute for a valid compaction algorithm.

Scope predicate: `(p.reservation_counter_exact and p.reserved_ranges_disjoint and p.counter_cannot_wrap) and (p.only_in_bounds_reservations_write and p.no_external_reader_before_batch_commit) and (p.collective_wait_covers_all_reservations_and_writes and p.rollback_subtracts_every_batch_reservation) and (p.original_batch_inputs_retained_until_retry_finished and p.retry_replays_each_stage_exact_once) and (p.compactor_consumes_complete_committed_prefix_and_current_stage and p.all_input_reads_finish_before_overwrite) and (p.commit_or_compaction_publishes_count_after_records and p.retry_protocol_terminates)`.

- Precondition `pre_0` (implementation): `p.reservation_counter_exact and p.reserved_ranges_disjoint and p.counter_cannot_wrap` — Atomic reservations allocate paired nonoverlapping ranges; the counter can temporarily exceed capacity but cannot overflow its integer range.
- Precondition `pre_1` (implementation): `p.only_in_bounds_reservations_write and p.no_external_reader_before_batch_commit` — Successful writers touch only in-bounds speculative slots; no consumer observes an unresolved batch.
- Precondition `pre_2` (implementation): `p.collective_wait_covers_all_reservations_and_writes and p.rollback_subtracts_every_batch_reservation` — After collective synchronization, subtract the entire batch reservation count, including successful writes, restoring the previous committed length.
- Precondition `pre_3` (implementation): `p.original_batch_inputs_retained_until_retry_finished and p.retry_replays_each_stage_exact_once` — Original stage data remains available and immutable until retry/compaction is complete.
- Precondition `pre_4` (implementation): `p.compactor_consumes_complete_committed_prefix_and_current_stage and p.all_input_reads_finish_before_overwrite` — If even one stage does not fit, rollback it and compact from the prior committed records plus complete current stage; no input is silently truncated.
- Precondition `pre_5` (implementation): `p.commit_or_compaction_publishes_count_after_records and p.retry_protocol_terminates` — Only complete record sets publish a new count; compaction/commit progresses and all collectives have the same participants.

## Local example and misuse

For a local instance of `map`, identify the object `reservation_rollback_retry` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/reservation_rollback_retry@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/reservation_rollback_retry@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Candidate records, input stages and per-worker summaries with explicit live intervals.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `logical_equivalence` (proof): Every successful fast batch appends its records once; every failed batch has no visible committed effect before its original input is retried exactly once. Compaction consumes the full declared logical input, so physical candidate capacity never causes silent loss.
- `protocol` (test): Native participants, counters, visibility and scratch lifetimes implement the declared complete dataflow.

Ordered comparisons must agree across thresholds, local compaction and final selection; finite ties use the declared arbitrary-identity policy.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
