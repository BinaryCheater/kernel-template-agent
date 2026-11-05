# event_ordered_host_device_staging@1: event ordered host device staging

## When to consider

When host batches feed GPU work across streams, give host staging and device slots separate release events so an input copy cannot overwrite live compute or block its own future submission. At batch2 reusing device slot0, pinned host slot0 becomes reusable when batch0 H2D finishes; device slot0 only when batch0 compute finishes. Treating those as one event either overwrites live GPU data or needlessly delays host staging.

## Computation before and after

When host batches feed GPU work across streams, give host staging and device slots separate release events so an input copy cannot overwrite live compute or block its own future submission.

Before copying into a recycled device slot, copy stream waits on its previous compute-release event. Copy the selected batch, record a ready event on copy stream, and enqueue a compute-stream wait on that ready generation before consumers. After all consumers are submitted, record release on compute stream. For pageable host input, wait for the previous H2D completion before refilling the pinned host staging slot, then submit H2D from that pinned slot and record host-reusable completion. Host slot may be refilled while device computation still reads the device slot; device reuse waits longer. Already resident identical batches can reuse without a transfer.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| device_slots | compile / software | [1, 2] | Observed single/double device buffering menu; pinned host slots mirror reusable transfer sources. |

## Coupled constraints and scope

NVIDIA CUDA out-of-core KMeans batch loader. This host/device stream protocol is not the CTA barrier semantics of seed multi_buffer_rotation@2 or global-to-shared scope of async_transfer@1. Exception teardown drains both streams before storage/events are destroyed. Resource effects: One/two device batch buffers plus optional pinned host staging and three event types per slot; allows overlap but adds host copy for pageable inputs. No overlap benefit claimed without timing.

Scope predicate: `p.stream_events_supported and p.source_lifetime_valid and p.all_consumers_submitted_before_release and p.slot_generation_matched and p.pinned_staging_for_pageable_source`.

- Precondition `pre_0` (semantic): `p.stream_events_supported and p.source_lifetime_valid` — Copy/compute streams support events and source memory remains live until H2D completion.
- Precondition `pre_1` (semantic): `p.all_consumers_submitted_before_release and p.slot_generation_matched` — Release event follows the last consumer in compute stream; waits bind the intended event record generation.
- Precondition `pre_2` (semantic): `p.pinned_staging_for_pageable_source` — Pageable input uses retained pinned staging where asynchronous submission is required for progress.

## Local example and misuse

For a local instance of `map`, identify the object `event_ordered_host_device_staging` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/event_ordered_host_device_staging@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/event_ordered_host_device_staging@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One/two device batch buffers plus optional pinned host staging and three event types per slot; allows overlap but adds host copy for pageable inputs. No overlap benefit claimed without timing.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `generation` (proof): Each compute consumer sees the batch matching its position, not another generation of the slot.
- `separate_retirement` (proof): Host staging waits for H2D reads; device destination waits for compute reads.
- `event_capture` (proof): Every wait captures the intended record before an event is re-recorded for another generation.
- `submission_progress` (proof): Host submission cannot block on a device reuse event that needs not-yet-submitted compute to finish.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
