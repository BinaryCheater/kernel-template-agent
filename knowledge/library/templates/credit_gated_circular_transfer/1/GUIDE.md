# credit_gated_circular_transfer@1: credit gated circular transfer

## When to consider

When peers repeatedly reuse a bounded communication FIFO, use monotone ready/consumed sequence numbers to gate both payload reads and slot overwrites. For fifo_steps=8,d=1,s=8, sender requires consumed>=1 before overwriting slot0; receiver requires ready>=9 to accept new slot0. Seeing ready=1 from the prior cycle must not satisfy the wait.

## Computation before and after

When peers repeatedly reuse a bounded communication FIFO, use monotone ready/consumed sequence numbers to gate both payload reads and slot overwrites.

For sequence s and slice increment d, sender waits until consumed+fifo_steps >= s+d before writing slot s%fifo_steps; receiver waits until ready>=s+d before reading it. After payload work, synchronize participant threads and publish the corresponding advanced counter: sender publishes ready after visibility fence, receiver publishes consumed after final payload read. This separates permission to read a produced slice from permission to overwrite a retired slice.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| fifo_steps | compile / software | [8] | Source NCCL_STEPS binding must be verified against include/device definitions; fixed for this instance. |
| steps_per_slice | compile / software | {"lo": 1, "hi": 8} | Protocol slice advances this many sequence units and fits FIFO capacity. |

## Coupled constraints and scope

NVIDIA CUDA NCCL SIMPLE buffered point-to-point slice protocol. Direct registered buffers, network FIFO offsets and multimem polling need their own pointer/ordering bindings. Source volatile polling is not promoted to a generic C++ acquire proof. Resource effects: Bounded circular payload storage fifo_steps*step_elements and ready/consumed counters per connection; polling and barriers per slice. Smaller slices increase synchronization frequency.

Scope predicate: `p.sequence_no_wrap and p.peer_protocol_agrees and p.payload_publication_and_consumption_ordered and p.peers_can_progress and p.abort_handled`.

- Precondition `pre_0` (semantic): `p.sequence_no_wrap and p.peer_protocol_agrees` — Both endpoints use matching sequence units, slot mapping and initial counters without wrap.
- Precondition `pre_1` (semantic): `p.payload_publication_and_consumption_ordered` — Ready is published after writes are visible and consumed is published only after last read.
- Precondition `pre_2` (semantic): `p.peers_can_progress and p.abort_handled` — Peers can execute awaited work and failure/abort is propagated.

## Local example and misuse

For a local instance of `map`, identify the object `credit_gated_circular_transfer` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/nccl/delta/credit_gated_circular_transfer@1/template.yaml`. Original outline: `knowledge/studies/nccl/delta/credit_gated_circular_transfer@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Bounded circular payload storage fifo_steps*step_elements and ready/consumed counters per connection; polling and barriers per slice. Smaller slices increase synchronization frequency.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `no_overwrite` (proof): A sender cannot reuse a circular slot while its previous receiver still reads it.
- `no_stale_read` (proof): A receiver does not accept a previous generation of the same slot as the current slice.
- `visibility` (proof): Transport-specific polling and payload loads observe publication across the actual peer memory domain.
- `progress` (proof): Credit dependencies and participant scheduling admit progress or terminate on abort.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
