# acknowledged_direct_buffer_exchange@1: acknowledged direct buffer exchange

## When to consider

When direct peer reads or writes replace staging copies, exchange the registered endpoint once per operation and acknowledge descriptor consumption before reusing the exchange slot. prims_simple.h:814-821,836-925. Missing exchange slots use work dnInputs/dnOutputs mapping; cross-clique peer index differs from local-rank index.

## Computation before and after

When direct peer reads or writes replace staging copies, exchange the registered endpoint once per operation and acknowledge descriptor consumption before reusing the exchange slot.

Provider waits until exchange slot is null, computes endpoint in peer address space, and publishes nonnull pointer. Acceptor waits nonnull, captures pointer, clears slot to null acknowledging descriptor consumption. Use direct write to receiver output or direct read from sender input/completed output according to matched mode. Registered predetermined mappings may bypass pointer exchange. Preserve payload ready/consumed counters and provider buffer lifetime until actual peer completion, independently of descriptor-slot acknowledgment.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Registered accessible peer memory, transport-specific pointer visibility and matching direct mode. Not arbitrary host pointers. Resource effects: Eliminates staged payload copy but retains pointer handshake and flow-control state.

Scope predicate: `p.registered_peer_pointer_mapping_valid and p.direct_mode_matched_by_peers and p.exchange_slot_visibility_and_atomic_pointer_access and p.null_is_unpublished_sentinel and p.payload_ready_and_retirement_protocol_retained and p.buffer_lifetime_covers_peer_use`.

- Precondition `pre_0` (semantic): `p.registered_peer_pointer_mapping_valid and p.direct_mode_matched_by_peers` — Direct-write receiver provides output; direct-read sender provides input or completed output.
- Precondition `pre_1` (semantic): `p.exchange_slot_visibility_and_atomic_pointer_access and p.null_is_unpublished_sentinel` — Pointer publication/consumption obey selected transport coherence; valid endpoint is nonnull.
- Precondition `pre_2` (semantic): `p.payload_ready_and_retirement_protocol_retained and p.buffer_lifetime_covers_peer_use` — Consuming the pointer descriptor does not retire the payload buffer.

## Local example and misuse

For a local instance of `map`, identify the object `acknowledged_direct_buffer_exchange` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/nccl/delta/acknowledged_direct_buffer_exchange@1/template.yaml`. Original outline: `knowledge/studies/nccl/delta/acknowledged_direct_buffer_exchange@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Eliminates staged payload copy but retains pointer handshake and flow-control state.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `descriptor` (proof): One provider value is consumed once before overwrite; matched roles select the same semantic buffer and offset.
- `lifetime` (proof): Clearing exchange slot only frees descriptor slot, not payload; direct read waits peer retirement before source reuse.
- `participation` (proof): Constructors of both peers participate even when individual operations select a buffered path.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
