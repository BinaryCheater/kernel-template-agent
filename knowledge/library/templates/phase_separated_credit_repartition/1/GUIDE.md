# phase_separated_credit_repartition@1: phase separated credit repartition

## When to consider

When a shared receive ring changes its slot-to-peer allocation, separate credit generations into phases and reset only a phase that can no longer have in-flight signals. symmetric/gin_scratch__funcs.h:160-193,218-293; reduce_scatter_gin.cuh:65-105 distinguishes soft pipeline chunk cap from hard credit minimum.

## Computation before and after

When a shared receive ring changes its slot-to-peer allocation, separate credit generations into phases and reset only a phase that can no longer have in-flight signals.

Allocate B=2^b slots across N peer steps with quotient B//N plus one credit for first B%N peers. When allocation changes, issue initial credits in next phase, reset only previous phase guaranteed quiescent, then advance phase modulo the declared phase bank count and reset monotonic round/step indices. Sender waits clear-to-send then puts data into slot monoStep%B and signals ready-to-read. Receiver waits readiness, consumes/reduces, then returns a credit to the next step that aliases this slot: peerStep=(step+B%N)%N. End round advances monoStep by N. Keep send-source completion request ring independent from receive credits.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| buffer_count_log2 | compile / software | {"lo": 1, "hi": 20} | Power-of-two receive slot count constrained by per-peer and total credit limits. |

## Coupled constraints and scope

Source GIN inbox four-phase bank and completed peer rounds; phase separation is a protocol premise needing backend signal-order proof. Resource effects: Adapts ring slot count to chunk size/peer count; minimum chunk size enforces hard per-peer and total credit capacities.

Scope predicate: `p.signal_release_order_prevents_old_phase_inflight and p.phase_reuse_separation_proved and p.complete_peer_rounds_and_slot_alias_mapping and p.unique_producer_per_live_slot and p.network_put_signals_payload_ready and p.receive_retirement_after_readers`.

- Precondition `pre_0` (semantic): `p.signal_release_order_prevents_old_phase_inflight and p.phase_reuse_separation_proved` — No reset can erase a valid in-flight credit or readiness signal.
- Precondition `pre_1` (semantic): `p.complete_peer_rounds_and_slot_alias_mapping and p.unique_producer_per_live_slot` — Next credited peer is exactly the peer whose future step aliases the retired slot.
- Precondition `pre_2` (semantic): `p.network_put_signals_payload_ready and p.receive_retirement_after_readers` — Clear-to-send credits prevent overwrite; ready-to-read signals publish completed payload.

## Local example and misuse

For a local instance of `map`, identify the object `phase_separated_credit_repartition` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/nccl/delta/phase_separated_credit_repartition@1/template.yaml`. Original outline: `knowledge/studies/nccl/delta/phase_separated_credit_repartition@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Adapts ring slot count to chunk size/peer count; minimum chunk size enforces hard per-peer and total credit capacities.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `allocation` (proof): Initial credits sum to B and no simultaneously live producers share a slot.
- `retirement` (proof): A new producer receives slot credit only after prior payload readers retire.
- `phase` (proof): No current or in-flight next phase signal is reset; phase wrap cannot alias live old state.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
