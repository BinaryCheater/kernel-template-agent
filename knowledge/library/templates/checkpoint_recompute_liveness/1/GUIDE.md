# checkpoint_recompute_liveness@1: checkpoint recompute output and tape lifetime

## When to consider

When checkpoint recompute produces backward tapes, separate two choices: elide output only with a no-reader frame proof, and retain tapes only for their matching backward lifetime. Last-saver-only retention avoids carrying those tapes through later-saver backwards; missing/evicted sets use ordinary local replay.

## Computation before and after

Prove checkpoint replay output deadness and reuse local tapes only within an admitted backward lifetime.

During a checkpoint recompute, preserve the original node saved-tensor sequence. A versioned frame adapter may prove that the executable node will fill exactly the remaining saved holders and early-stop the replay immediately afterward. For elide_with_frame_proof skip the output program only when that proof and no-output-consumer premise hold; keep the allocated tensor shape but never expose its unwritten contents. Otherwise execute output normally. For replay_in_backward discard local recompute tapes and use the admitted local replay later. For retain_for_backward detach and own those tapes for the same backward; for last_saver_only do so only when the verified frame proof succeeds, otherwise choose local replay. Bound the handoff registry to one unconsumed set, evict before new materialization and key by task/node/operand/boundary identity. Keep handoff outside save_for_backward so replay holder matching stays unchanged. Backward takes its matching set once or replays on a miss; validate geometry/artifacts, bind it through last readers then release or safely readopt compatible arena tapes. Task-end cleanup and bounded later eviction handle unconsumed or failed-task leftovers.

Recover the original computation: Replacing a handed-over tape by the identical local replay artifact supplies the same backward operands. Removing the output stage removes only stores whose value has no consumer before replay early-stop or afterward; the original saved-tensor sequence and allocated interface remain intact.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| recompute_output | compile / software | ["compute", "elide_with_frame_proof"] | Always execute the output stage, or omit its stores/computation only when the verified replay frame proves no output consumer. Output allocation and shape still exist. |
| tape_handoff | compile / software | ["replay_in_backward", "retain_for_backward", "last_saver_only"] | Discard recompute tapes and regenerate in backward, retain them for the same backward task, or retain only when frame liveness proves this node closes the checkpoint region. Retain-for-backward can increase the region peak through later-saver backwards. |

## Coupled constraints and scope

FlashInfer deterministic KCP GDN eager non-reentrant torch checkpoint path. Source frame adapter recognizes torch release2.14.1 after removing a local build suffix; this is a retained-source compatibility menu, not a verified installation in this study. Handoff is light saved-boundary mode with local tokens, optional normalized Q/K rows and optional stage_m plane. Output deadness is frame-specific; direct forward, graph capture/replay, unknown hooks/releases and non-executable nodes do not receive that proof. Native runtime callback, nested-checkpoint and task-failure behavior remain integration obligations.

Scope predicate: `(p.checkpoint_recompute_and_original_saved_tensor_sequence_match) and (recompute_output != 'elide_with_frame_proof' or p.frame_proves_executable_last_saver_early_stop_and_no_output_reader) and (tape_handoff != 'last_saver_only' or p.handoff_requires_same_verified_last_saver_frame) and (p.task_operand_boundary_geometry_and_numeric_generation_identify_same_backward) and (p.retained_tapes_unbound_from_reusable_arena_and_owned_until_last_reader) and (p.registry_capacity_one_eviction_before_new_materialization_and_miss_replay) and (p.source_eager_light_save_and_versioned_checkpoint_adapter_supported)`.

- Precondition `pre_recompute` (semantic): `p.checkpoint_recompute_and_original_saved_tensor_sequence_match` — The replay reproduces the original executable autograd node and exactly its saved-tensor sequence. Passing additional tapes through save_for_backward would change holder matching and is not a valid handoff.
- Precondition `pre_dead` (implementation): `recompute_output != 'elide_with_frame_proof' or p.frame_proves_executable_last_saver_early_stop_and_no_output_reader` — Output can be left unwritten only in a verified non-reentrant replay frame: executable node, recognized hook/frame protocol, early stop enabled, remaining holders exactly the tensors this node next saves, and neither backward nor saved tensors reads this output. A graph-task ID or holder count alone is insufficient.
- Precondition `pre_auto` (implementation): `tape_handoff != 'last_saver_only' or p.handoff_requires_same_verified_last_saver_frame` — Automatic retention hands tapes over only when the same executable last-saver frame proof succeeds. Otherwise regenerate them locally in backward; this avoids retaining tapes through later-saver backwards.
- Precondition `pre_identity` (implementation): `p.task_operand_boundary_geometry_and_numeric_generation_identify_same_backward` — Every handed-over artifact belongs to the same graph task, autograd node, operands, entry boundary, arena geometry and numerical generation as its backward consumer. Pointer identity is meaningful only while owning tensors retain that storage generation.
- Precondition `pre_storage` (implementation): `p.retained_tapes_unbound_from_reusable_arena_and_owned_until_last_reader` — Handoff detaches tapes, optional local transfer plane and normalized rows from writable arena reuse. Fresh storage is retained through all backward readers; geometry permits optional arena readoption only after those readers.
- Precondition `pre_bound` (implementation): `p.registry_capacity_one_eviction_before_new_materialization_and_miss_replay` — At most one unconsumed entry lives in the source registry. Evict it before materializing a new retained set; an evicted/missing entry replays instead. Consume a matched entry once, release after its last reader, and clean unconsumed entries at task end or a later bounded eviction. A failed task need not run a final callback.
- Precondition `pre_backend` (implementation): `p.source_eager_light_save_and_versioned_checkpoint_adapter_supported` — Concrete handoff requires light saved-boundary backward, eager execution and live local tokens; graph capture/replay excludes it. Frame-dependent choices recognize only their verified runtime release/protocol and fall back to output compute/local tape replay when proof is unavailable.

## Local example and misuse

For a local instance of `store`, identify the object `checkpoint_recompute_output_and_tape_lifetime` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/checkpoint_recompute_liveness@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/checkpoint_recompute_liveness@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Fresh detached tapes with task-bound ownership, one unconsumed registry entry and release after last backward reader.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `output_deadness` (proof): Skipped output has no reader, with an executable final saver, exact holder sequence, verified protocol and early stop.
- `handoff_identity` (test): Every retained tape matches the sole consuming backward task and native stage; storage is immutable through last use.
- `retention_bound` (test): The unconsumed registry has at most one entry, eviction precedes new retained allocation, misses replay, and cleanup does not strand old generations.
- `peak_scope` (proof): Last-saver-only retention prevents kept tapes spanning later-saver backward work; broader retain-for-backward has no such peak-memory guarantee.

A same-shaped or same-pointer tape can be from the wrong generation. Skipping output based only on being inside backward can expose unwritten data; changing the saved tensor sequence breaks checkpoint holder matching.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
