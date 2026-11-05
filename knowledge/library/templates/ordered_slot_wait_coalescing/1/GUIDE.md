# ordered_slot_wait_coalescing@1: ordered empty credit batch

## When to consider

When a producer writes consecutive ring slots and later returned credits certify all earlier readers finished, consider waiting only on the last slot of a short batch. Preserve each item's publication and exact wrap generation, choose batch_width within capacity, and wait on the last valid tail item. The pinned conversion loop uses pairs plus an odd single tail; out-of-order completion invalidates the optimization even if release calls appear in source order.

## Computation before and after

One latest-credit wait protects an ordered short batch of distinct ring slots.

Process one finite item sequence in consecutive batches of up to batch_width. For a batch ending at sequence index q, wait for the exact empty-slot generation q mod stages. Any previous user of that slot has index q-stages. Prefix-ordered retirement implies every earlier batch slot's previous user has also retired; negative previous indices refer to initially empty slots. Produce and publish every item separately in sequence order without re-waiting each earlier slot. A short tail waits on its last valid item, and an empty sequence waits on none. Preserve per-slot phase advance, full-readiness and consumer release protocol.

Recover the original computation: Reinsert each omitted earlier per-item empty wait; it is already satisfied by the batch's last-credit wait under the retirement-prefix premise.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| batch_width | compile / software | {"lo": 1, "hi": 2} | Consecutive wide-output slots protected by one latest-slot wait; pinned transform loop unrolls pairs with an odd single tail. |

## Coupled constraints and scope

One cooperative producer sequence over an ordered ring with one-/two-item batches and prefix-retired old readers. Pinned accumulator-scale mixed GEMM waits/acquires the second wide slot for a pair, still waits on both narrow inputs, publishes both outputs and handles an odd output separately. Source allocation requires at least two wide stages. Native asynchronous retirement ordering is a backend premise; source release-call order alone does not prove it. Other batch widths, out-of-order retirement or shared producer ownership need another verified variant.

Scope predicate: `p.single_producer_sequence and p.retirement_is_prefix_ordered and p.generation_wait_exact and p.initial_slots_empty and p.each_item_published_once and p.consumer_readiness_unchanged and p.tail_waits_last_valid_item`.

- Precondition `pre_prefix` (semantic): `p.retirement_is_prefix_ordered` — Readiness of a later returned credit certifies all earlier sequence positions are retired, not merely that release calls were issued in order.
- Precondition `pre_generation` (implementation): `p.generation_wait_exact and p.initial_slots_empty` — The waited credit belongs to the exact requested wrap generation and initial slots are empty.
- Precondition `pre_protocol` (semantic): `p.each_item_published_once and p.consumer_readiness_unchanged and p.tail_waits_last_valid_item` — Only redundant empty waits change; every item still publishes readiness once and the tail waits on actual work.
- Constraint `c_capacity` (semantic): `p.items >= 0 and batch_width <= p.stages and p.single_producer_sequence` — A batch cannot alias a ring slot and one cooperative producer owns the sequence.

## Local example and misuse

For a local instance of `sync`, identify the object `ordered_empty_credit_batch` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/ordered_slot_wait_coalescing@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/ordered_slot_wait_coalescing@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Existing slot ring and generation state; no new data buffer.. Structured effects describe work and resource changes, not timings.

```yaml
transactions:
- id: empty_waits
  pattern: sync
  kind: mbarrier_sync
  count: (p.items + batch_width - 1) // batch_width
  note: Empty-wait invocations for one finite sequence only; replaces that component
    rather than adding to a complete handshake cost.
storage: []
overlap: []
```

## Obligations and numerical checks

- `no_live_overwrite` (exhaustive): Every omitted earlier wait is satisfied before that item's slot is written.
- `phase_and_tail` (exhaustive): Ring phase/generation remains exact across wraps and partial batches never wait on nonexistent items.
- `publication_identity` (test): Every item still produces its own unchanged readiness event and is consumed once in order.

Synchronization-only; missing the retirement-prefix premise permits live-slot overwrite.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- refines: per_slot_handshake (outside menu); Only the empty-wait component is replaced; full/empty lifetime and publication obligations remain.

Structured contract: [template.yaml](template.yaml).
