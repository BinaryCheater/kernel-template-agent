# designated_segment_completion@1: segmented completion phase

## When to consider

When independent CTA partials must be read by one cleanup CTA, separate publication from numerical combination. Choose one-shot disjoint entrances or reusable phases with a second retirement join. For three producers, two increments of one plus H-2 flip the phase only after all arrive. Reject a persistent launch whose waiting CTAs can prevent a producer from being scheduled.

## Computation before and after

Join separate CTA publications with a phase-flipping counter and designated consumer.

Partition CTAs into independent segments and publish each CTA partial in its unique scratch slot. Fence writes and synchronize the CTA before one representative arrives. For phase bit H=2^63, ordinary producers add 1 and the designated last coordinate adds H-(N-1), modulo 2^64. The designated coordinate waits for the phase bit to differ from its pre-arrival state; in reusable mode every producer waits. After the consumer finishes cleanup, reusable mode executes a second all-producer join before scratch reuse. One-shot entrances use distinct scratch and counters.

Recover the original computation: Erase counters and phases, leaving a join after each producer publication and retirement before destructive reuse.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| mode | compile / software | ["one_shot", "reusable"] | One designated consumer waits, or every participant joins with a second retirement barrier. |

## Coupled constraints and scope

Device-local segmented CTA completion, one live phase per semaphore. This source realization requires explicit scheduling/residency premises; it is not a general oversubscribed-grid barrier. Effects unpriced.

Scope predicate: `p.counter_initialized and p.publication_visible and p.awaited_producers_can_progress and p.no_overlapping_generations`.

- Precondition `pre_protocol` (implementation): `p.counter_initialized and p.publication_visible and p.designated_consumer_unique and p.awaited_producers_can_progress and p.no_overlapping_generations` — Counter initialization, device publication ordering, unique consumer and scheduling progress are supplied by the realization.
- Constraint `c_participants` (semantic): `p.arrivals == p.segment_size and p.segment_size >= 1 and p.segment_size < 2 ** 63` — Exactly one arrival per producer and no count ambiguity.

## Local example and misuse

For a local instance of `sync`, identify the object `segmented_completion_phase` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/fuser/delta/designated_segment_completion@1/template.yaml`. Original outline: `knowledge/studies/fuser/delta/designated_segment_completion@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Per-entrance scratch plus per-segment 64-bit semaphore.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `phase_join` (proof): The designated waiter cannot observe a phase transition until all N arrivals occurred.
- `publication` (test): The waiter reads every producer payload only after device-visible publication.
- `reuse` (test): One-shot entrances are disjoint and reusable mode retires all scratch readers.
- `progress` (test): All awaited CTAs execute; persistent waiting cannot strand unscheduled producers.

The completion protocol does not establish reduction associativity or floating-point error bounds.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
