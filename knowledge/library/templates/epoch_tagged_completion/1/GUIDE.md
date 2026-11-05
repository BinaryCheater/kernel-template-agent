# epoch_tagged_completion@1: epoch completion counter

## When to consider

Use when repeated launches reuse completion state and payload buffers; bind participants to the actual producer count and keep only one live epoch. Tag the counter with128*(epoch+1), accept initialization in the current count interval and wait for the exact terminal count before consumption. For16 split producers a stale previous-epoch terminal cannot satisfy the next join; advance the epoch only after every user retires.

## Computation before and after

Initialize a fresh epoch tag then join exact producer arrivals and retire before reuse.

For epoch e use base=(e+1)*128 in one64-bit counter. An initializer publishes base. Before arriving each producer waits until unsigned64(state-base)<participants; it then publishes its completed payload and atomically increments the counter with release ordering exactly once. A consumer acquires until state==base+participants before reading all payloads. All users of the counter and payload retire before a separate grid exit count permits advancing the shared epoch and resetting that exit count. Advancing the epoch is not itself permission to overwrite still-live payloads.

Recover the original computation: Replace the tagged counter with one completed-producer event per payload and a join after every event; erase the epoch once all users have retired.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| participants | compile / boundary | {"lo": 1, "hi": 64} | Actual producers of this completion domain; derived from the upstream partition. |

## Coupled constraints and scope

Device-local completion joins with1..64 arrivals per counter and one nonoverlapping launch epoch per shared buffer. This version fixes64-bit states and128-wide tags. It joins separately stored partials rather than atomically combining their numerical values. Repeated graph execution must acquire a new epoch even when a grid identifier is reused. Source progress and release-sequence visibility require device evidence.

Scope predicate: `p.one_live_epoch_per_buffer and p.release_acquire_supported`.

- Precondition `pre_lifetime` (semantic): `p.one_live_epoch_per_buffer and p.previous_epoch_retired` — No new epoch overlaps a previous user of the same counter or payload.
- Precondition `pre_memory` (implementation): `p.release_acquire_supported and p.producers_can_progress` — The environment implements the required publication ordering and allows awaited producers to execute.
- Constraint `c_count` (semantic): `participants == p.producer_count` — The terminal count matches the contributing producers.
- Constraint `c_epoch_range` (implementation): `p.epoch >= 0 and p.epoch <= 2 ** 57 - 2` — This version's64-bit state and128-wide epoch tags do not wrap.

## Local example and misuse

For a local instance of `sync`, identify the object `epoch_completion_counter` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/deepgemm/delta/epoch_tagged_completion@1/template.yaml`. Original outline: `knowledge/studies/deepgemm/delta/epoch_tagged_completion@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One tagged counter per completion domain plus shared epoch and grid exit count.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `current_generation` (exhaustive): A stale counter value never satisfies current-epoch initialization or completion waits.
- `exact_arrivals` (exhaustive): Every required producer arrives exactly once after publishing its payload and no unrelated producer increments this counter.
- `visibility` (test): An acquiring terminal wait observes all completed producers' payload writes through the required release ordering.
- `retirement` (test): Epoch advancement and payload reuse follow every producer and consumer's last use of the current generation.
- `progress` (test): Initializers and awaited producers can run without circular waits or residency deadlock.

No numerical ordering is implied; the consumer's partial-combine order is a separate refinement.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
