# byte_arrival_completion_join@1: byte and arrival stage completion

## When to consider

When one operand stage mixes engine-counted transfers with writes completed by independent copy producers, consider joining both readiness streams in one barrier. Preserve both counters and reserve one arrival for atomic transaction registration, so early copy arrivals cannot publish unfinished data. Bind exact transaction_bytes and copy_arrivals to the signal incidence; the pinned SM100 path uses128/256 copy arrivals plus one owner, and its two-group delay requires a progress-safe stage capacity and explicit short-tail drain.

## Computation before and after

A full-stage barrier completes only after byte and copy-arrival counters both drain.

Initialize each full barrier generation with copy_arrivals+1 pending arrivals and no registered transaction bytes. The extra arrival belongs to the single transaction owner. After acquiring the empty slot, that owner atomically registers transaction_bytes and consumes its reserved arrival before issuing the corresponding transfers. Copy producers independently issue their writes, wait for their own completion and each deliver their counted arrival. Transfer completion decrements the byte counter. The consumer may read only after the barrier generation completes with both counters zero. Release the slot only after all consumers finish; reset both counts for the next generation.

Recover the original computation: Replace the joint full-barrier wait by independent waits for every required transaction and copy-producer completion, then read the same stage generation.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| transaction_bytes | compile / boundary | {"lo": 1, "hi": 4194304} | Exact bytes registered by the single transaction owner and delivered to this barrier generation. |
| copy_arrivals | compile / boundary | {"lo": 1, "hi": 4096} | Copy-producer completion arrivals excluding the one reserved transaction-registration arrival. |

## Coupled constraints and scope

One transaction-registration owner plus independent copy-completion producers per stage generation, on a backend joining byte and arrival counters. Pinned convolution joins A/B/SFB TMA bytes with128 SFA thread arrivals for one CTA or256 for two, initializing129/257 arrivals including the reserved owner. Both CTAs signal the leader for two-CTA SFA writes. Source delays arrivals by two copy groups and drains pending arrivals at each tile tail; adequate stage capacity remains a separate progress premise. Different signal incidence, registration ownership, release domain or completion semantics needs a verified binding/variant.

Scope predicate: `p.joint_counter_backend_supported and p.atomic_expect_and_reserved_arrive and p.copy_arrives_after_own_writes and p.transactions_signal_after_writes and p.signals_target_correct_generation and p.slot_empty_before_production and p.all_consumers_retire_before_reuse and p.delay_schedule_can_progress`.

- Precondition `pre_backend` (implementation): `p.joint_counter_backend_supported and p.atomic_expect_and_reserved_arrive` — The backend waits for both counters and atomically registers bytes while consuming the reserved arrival.
- Precondition `pre_copy` (semantic): `p.copy_arrives_after_own_writes and p.transactions_signal_after_writes and p.signals_target_correct_generation` — Every completion signal certifies its own data and reaches the correct barrier generation.
- Precondition `pre_lifetime` (semantic): `p.slot_empty_before_production and p.all_consumers_retire_before_reuse and p.delay_schedule_can_progress` — Slot generations do not overlap and any delayed producer-arrival schedule can issue enough work to drain.
- Constraint `c_incidence` (semantic): `transaction_bytes == p.delivered_transaction_bytes and copy_arrivals == p.delivered_copy_arrivals` — Both counters match the actual generation's signal incidence including cross-CTA targets.

## Local example and misuse

For a local instance of `sync`, identify the object `byte_and_arrival_stage_completion` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/byte_arrival_completion_join@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/byte_arrival_completion_join@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One full barrier per stage plus independently managed empty/lifetime state.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: joint_full_barriers
  level: smem
  scope: cta
  bytes: p.stages * p.full_barrier_bytes
  note: Full barriers only; empty barriers and extra source allocations are accounted
    separately.
overlap: []
```

## Obligations and numerical checks

- `no_early_read` (exhaustive): Completion implies both all registered transaction writes and all independent copy writes are ready for the selected generation.
- `registration_race` (exhaustive): Copy producers finishing before transaction registration cannot complete the barrier prematurely because the registration arrival is reserved.
- `incidence_and_target` (test): Exactly the registered bytes and arrivals reach each local or remote generation once without omission or double count.
- `progress_and_reuse` (test): Delayed arrivals and tail drain are satisfiable before slot reuse and all readers retain their generation until release.

Synchronization refinement only; early readiness or wrong generations corrupt operand values.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
