# interleaved_consumer_streams@1: striped work stage cursor

## When to consider

When independent consumer groups alternate tiles from one input producer, consider striped item ownership with a cursor tied to the producer's global stage sequence. Skip unowned items by advancing their stage extent without releasing them, preserve wrap phases and count only the active group's arrivals. The pinned two-group source uses uniform K stages per tile; K0 or variable lengths are excluded. Native order arbitration and shared output lifetime are separate premises, and the source's two-slot declaration versus four-slot use is an explicit storage obligation rather than proof of native failure or a working implementation.

## Computation before and after

Alternate fixed-size work items and jump a global ring cursor over the other group's stripe.

Number each physical producer stream step q=item*steps_per_item+k. Group g initially advances its consumer cursor by g*steps_per_item without waiting or releasing skipped generations. Consume/release each step of its owned item once. Before its next item advance over (groups-1)*steps_per_item unowned steps without releasing them; advance its work scheduler groups times. Preserve ring index q mod stages and phase floor(q/stages) mod2 across these jumps. Resetting a tile-local count never resets index or phase. Handle an empty/short final stripe by checking item validity before any stage wait. Uniform positive item length is required; variable-length items need a different map.

Recover the original computation: Merge the disjoint consumer stripes by original item ordinal and reinsert their stage events into the producer's one global stream.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| groups | compile / software | [2] | Number of alternating consumer groups in the pinned implementation. |
| steps_per_item | runtime / boundary | {"lo": 1, "hi": 1048576} | Uniform positive producer-stage count per item; source binds the fixed K tile count. |

## Coupled constraints and scope

Two alternating groups consuming a common fixed-positive-length stage stream. SM120 pingpong uses one four-warp group per tile, advances the other tile's K-stage count without release and skips two scheduler positions between owned items. Its PipelineOrder depth2/length2 requires four barrier slots but the retained source declares only two: that declaration alone does not establish order_protocol_valid. Physical alignment padding may back extra slots and requires a separate layout argument. This candidate does not supply the external MMA/epilogue arbitration or its cost and does not admit K0.

Scope predicate: `p.producer_item_order_immutable and p.item_steps_uniform and p.group_partition_exact and p.active_consumer_count_matches and p.skipped_steps_not_released and p.owned_steps_consumed_once and p.order_protocol_valid and p.shared_output_lifetime_valid and p.progress_admitted`.

- Precondition `pre_stream` (semantic): `p.producer_item_order_immutable and p.item_steps_uniform and p.group_partition_exact` — Group g owns item ordinals g+2*j in the shared producer's item order.
- Precondition `pre_protocol` (semantic): `p.active_consumer_count_matches and p.skipped_steps_not_released and p.owned_steps_consumed_once` — Only the owning group consumes/releases an item's stages; barrier arrival count describes that group rather than both groups.
- Precondition `pre_order` (implementation): `p.order_protocol_valid and p.shared_output_lifetime_valid and p.progress_admitted` — External execution order and shared output storage protocol are correctly allocated and preserve progress/lifetimes.
- Constraint `c_ring` (implementation): `p.stages >= 1 and p.cursor_arithmetic_fits and steps_per_item == p.producer_steps_per_item` — Consumers use the same positive uniform item extent and representable ring cursor as the producer.

## Local example and misuse

For a local instance of `work_assign`, identify the object `striped_work_stage_cursor` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/interleaved_consumer_streams@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/interleaved_consumer_streams@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Separate consumer cursors over an existing shared producer ring.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `ownership` (exhaustive): Every valid item and all its producer steps belong to exactly one consumer group.
- `generation` (exhaustive): Owned waits/releases use the exact global slot generation across initial skips and repeated phase wraps.
- `protocol` (test): Skipped generations receive no release and active-group arrivals match the barrier; external ordering preserves progress and shared output lifetime.

Work/stage ownership only; a wrong phase or release count can consume stale values or overwrite live inputs.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- requires: per_slot_handshake (outside menu); The owning consumer group alone must satisfy each stage's readiness/release protocol.

Structured contract: [template.yaml](template.yaml).
