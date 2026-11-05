# chunk_ready_collective_pipeline@1: kernel collective chunk handoff

## When to consider

When collective completion of the whole matrix delays otherwise independent GEMM chunks, or GEMM completion delays independent output reduce-scatter chunks, publish readiness at the shared chunk boundary. Choose event-separated launches or a supported device-counter protocol, preserve original chunk identities, and join every stream before reuse. Chunk size/resource allocation may trade more launches and communication overhead for overlap; no benefit is assumed without measurement.

## Computation before and after

Dependency-preserving chunk handoff between matrix contraction and collective communication.

Partition the communicated matrix axis into logical chunks. For gather_input, each resident/local or received chunk may feed its GEMM only after its payload and metadata are complete; keep output addresses keyed to original chunk identity even when arrival order is cyclic. For scatter_output, a chunk reduce-scatter may begin only after the matching GEMM contribution is complete, and it produces exactly that rank-owned output range. stream_event uses a completion event recorded after each producing launch and a wait on its consumer stream. device_counter uses a supported chunk-granularity producer/consumer counter protocol with its explicit initialization, ready/consumed transition and memory ordering. Independent later producers/earlier consumers may overlap. Keep per-stream workspaces disjoint or ordered, and release payload buffers only after their final consumers. Join all compute, communication and optional copy streams before exposing the complete result or reusing launch state.

Recover the original computation: Serialize the same chunk producers and consumers in dependency order; concatenating gathered/GEMM chunks or rank-owned reduce-scatter chunks recovers the requested collective-plus-contraction result.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| direction | compile / boundary | ["gather_input", "scatter_output"] | All-gather chunk supplies a GEMM input, or completed GEMM output chunk feeds reduce-scatter. |
| handoff_mode | compile / software | ["stream_event", "device_counter"] | Separate chunk launches publish completion through CUDA events, or a supported vendor/kernel protocol publishes each chunk through device counters. |

## Coupled constraints and scope

CUDA stream/event or explicitly supported device-counter handoff between GEMM and all-gather/reduce-scatter chunks. Source TransformerEngine provides event-split and cuBLAS atomic-counter paths, plus an external cuBLASMp alternative. Counter mode in this pinned source is gated to CUDA 12.2..<13 and cuBLAS 12.2.5..<13. This rule does not supply the unavailable vendor tile-publication implementation, arbitrary cross-device memory ordering, network protocol or a measured overlap benefit. Bulk unrelated work overlap is a separate no-chunk-dependency case.

Scope predicate: `p.num_chunks > 0 and p.chunk_partition_complete_disjoint and (handoff_mode != 'device_counter' or p.producer_consumer_counter_contract_supported) and p.chunk_stage_dependencies_match and p.each_chunk_ready_before_consumer and p.collective_ownership_matches and p.publication_visible_to_consumer and p.transfer_completion_before_publication and p.buffer_and_workspace_reuse_ordered and p.counter_or_event_epoch_owned and p.all_participating_streams_joined and p.communication_progress_guaranteed and p.collective_numeric_policy_admitted and p.scale_metadata_matches_payload`.

- Precondition `pre_dependencies` (semantic): `p.chunk_stage_dependencies_match and p.each_chunk_ready_before_consumer and p.collective_ownership_matches` — Input/output chunk maps and collective ownership match every consumer dependency.
- Precondition `pre_memory` (implementation): `p.publication_visible_to_consumer and p.transfer_completion_before_publication and p.buffer_and_workspace_reuse_ordered` — Signal visibility and buffer lifetime cover the actual transfer/compute completion, not just launch.
- Precondition `pre_epoch` (implementation): `p.counter_or_event_epoch_owned and p.all_participating_streams_joined and p.communication_progress_guaranteed` — Launch epochs, final stream join and scheduling guarantee progress without stale completion reuse.
- Precondition `pre_numeric` (semantic): `p.collective_numeric_policy_admitted and p.scale_metadata_matches_payload` — Rank reduction and any quantized communication preserve the requested numerical/scale contract.
- Constraint `c_partition` (semantic): `p.num_chunks > 0 and p.chunk_partition_complete_disjoint` — Chunk domains cover the requested input/output exactly once.
- Constraint `c_counter` (implementation): `handoff_mode != 'device_counter' or p.producer_consumer_counter_contract_supported` — Counter granularity and producer/consumer roles are supported by the selected native provider.

## Local example and misuse

For a local instance of `sync`, identify the object `kernel_collective_chunk_handoff` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/transformerengine/delta/chunk_ready_collective_pipeline@1/template.yaml`. Original outline: `knowledge/studies/transformerengine/delta/chunk_ready_collective_pipeline@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Registered payload chunks, per-stream scratch and event/counter launch state retained through final join.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `chunk_cover` (proof): Every logical chunk is produced, consumed and placed once with the required collective rank ownership.
- `ready_means_complete` (test): Native event/counter publication implies the complete payload and metadata are visible to the consumer.
- `lifetime_and_join` (test): No buffer/counter/workspace is reused early and final completion joins every participating stream.
- `progress` (test): Concurrent communication/GEMM kernels have a supported forward-progress and resource allocation contract.

Quantized reduce-scatter must bind decoder scales and accumulation precision; moving a scale outside a rank reduction requires a common admitted factor.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
