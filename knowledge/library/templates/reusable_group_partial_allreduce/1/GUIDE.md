# reusable_group_partial_allreduce@1: repeated inter cta reduction

## When to consider

When the same resident CTA group repeatedly publishes partials and every CTA needs the result, alternate two data banks and two release/acquire counters. Four phases raise each counter to the participant count, then lower it to zero. The intervening group rendezvous retires old readers before a bank or counter is reused; a one-shot barrier cannot be reused without this generation invariant.

## Computation before and after

Repeated partial-table allreduce with double banks and alternating up/down counter generations.

For generation t, write the completed per-CTA partial into bank t%2 at a sender-injective slot. Choose counter t%2. In phases t%4 in {0,1}, each CTA leader release-adds +1 and acquire-waits for peers; in phases {2,3}, release-add -1 and wait for zero. Broadcast completion to all local threads, read every peer partial from the selected data bank and combine locally. Complete those reads before this CTA arrives at generation t+1. The next generation uses the other bank/counter. Before a peer can touch bank/counter t again at t+2, all peers have arrived at t+1, proving their old t reads/waits have retired. Counter states cycle (0,0)->(peers,0)->(peers,peers)->(0,peers)->(0,0).

Recover the original computation: Project each completed generation to one allreduce over its peer partials; forget the double banks and four-phase implementation.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| peers | compile / boundary | {"lo": 2, "hi": 32} | Fixed CTA participants in one reduction group; source static norm path combines at most 32 peers per row. |

## Coupled constraints and scope

Fixed inter-CTA groups of 2..32 with two initialized GPU-global signed counters and two private partial banks. Source common/utils.cuh uses release reduction-add, acquire loads and CTA completion, then repeats across rows/statistics. This is not a cluster transaction barrier, cancellation-capable group, dynamic membership or arbitrary oversubscribed grid barrier. Publication by other threads must be ordered before the leader release; occupancy/launch progress is a native binding, not supplied by counter arithmetic.

Scope predicate: `peers == p.participants and p.counter_initial_value == 0 and p.partial_bank_count == 2 and p.counter_bank_count == 2 and p.all_peers_same_generation_order and p.exactly_one_arrival_per_peer and p.every_peer_participates and p.partial_slots_disjoint and p.leader_release_publishes_partials and p.acquire_before_partial_reads and p.local_thread_convergence_valid and p.reads_complete_before_next_arrival and p.group_progress_guaranteed and p.counters_and_banks_exclusively_owned and p.combine_identity_and_rounding_admitted and p.every_partial_combined_once`.

- Precondition `pre_generation` (semantic): `p.all_peers_same_generation_order and p.exactly_one_arrival_per_peer and p.every_peer_participates` — Each generation has the same fixed participants with exactly one leader arrival each.
- Precondition `pre_publication` (implementation): `p.partial_slots_disjoint and p.leader_release_publishes_partials and p.acquire_before_partial_reads and p.local_thread_convergence_valid` — CTA-local publication and global release/acquire make all completed partials visible before reads.
- Precondition `pre_lifetime` (implementation): `p.reads_complete_before_next_arrival and p.group_progress_guaranteed and p.counters_and_banks_exclusively_owned` — Every old read retires before its next-generation arrival; spinning peers can all make progress and storage is group-private.
- Precondition `pre_combine` (semantic): `p.combine_identity_and_rounding_admitted and p.every_partial_combined_once` — The complete partial table is folded once using the requested combine/numerical policy.
- Constraint `c_group` (semantic): `peers == p.participants and p.counter_initial_value == 0` — Participant count is fixed and both counters start at zero.
- Constraint `c_storage` (implementation): `p.partial_bank_count == 2 and p.counter_bank_count == 2` — Two data banks and two counters implement the extracted four-phase protocol.

## Local example and misuse

For a local instance of `reduce`, identify the object `repeated_inter_cta_reduction` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/transformerengine/delta/reusable_group_partial_allreduce@1/template.yaml`. Original outline: `knowledge/studies/transformerengine/delta/reusable_group_partial_allreduce@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Two partial tables and two GPU-global counters per fixed CTA group.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `generation_safety` (proof): No counter changes for a new use until all peers have retired its previous wait; no partial bank is overwritten before old reads finish.
- `native_visibility` (test): Leader release/acquire and CTA convergence publish every relevant write under the native memory model.
- `progress` (test): All spinning CTA peers are scheduled and reach every group rendezvous.

The synchronization rule does not make a floating-point combine associative or change its requested fold order.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
