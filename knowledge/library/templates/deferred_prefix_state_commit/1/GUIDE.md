# deferred_prefix_state_commit@1: frozen checkpoint and selective prefix materialization

## When to consider

When speculative recurrent work must be rolled back, keep its authoritative checkpoint frozen. Choose complete post-token snapshots for direct prefix selection, or retain smaller update-input records and replay only the accepted consecutive prefix. Preserve the exact checkpoint/window epoch and update recipe, make zero acceptance a no-op, and retire all commit readers before the next verify overwrites tape.

## Computation before and after

Freeze the recurrent base during verification, retain full states or transition dependencies, then commit one accepted consecutive prefix.

Let S0 be the authoritative checkpoint and S_(t+1)=F_t(S_t,record_t) for t=0..T-1. Verify/read outputs without mutating S0. In state_snapshots retain complete post-token states S1..ST; in transition_tape retain all update records at their required stage/dtype and keep only virtual verification state. Once the accepted consecutive count a is published, a0 leaves checkpoint and optional tracking destinations unchanged. For a>0, snapshot mode selects S_a; tape mode reloads the original S0 and applies F0..F_(a-1) in order under the declared commit recipe. Publish the completed checkpoint S_a once after all relevant layer/state shards finish. If a requested track step b satisfies0<=b<a, additionally save post-token S_(b+1) at its disjoint destination; otherwise do not touch it. Discard suffix records logically without replaying them. A later verify may overwrite the linear retained window only after every commit/track reader retires. Repeating commit on an already-updated in-place checkpoint is not permitted.

Recover the original computation: Erase speculative readouts and retention mechanics. The authoritative transition sequence is exactly the accepted F0..F_(a-1) from the original checkpoint, with zero-length identity and the requested after-token tracking states. Snapshot selection and replay preserve the same declared recurrence/error contract; no bitwise equivalence across different arithmetic recipes is assumed.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| retention | compile / software | ["state_snapshots", "transition_tape"] | Trade one full post-token state per window position for smaller replayable state-update records. Snapshot copy avoids replay arithmetic; tape saves storage but recomputes accepted transitions. |

## Coupled constraints and scope

GDN MTP ReplaySSM on SM100/SM103 uses frozen FP32 checkpoints with K=V128, T>=3, BF16 raw K/V and FP32 log-decay/beta windows, then one accepted-prefix commit across layers. Full intermediate-state snapshots are the alternative interface and cannot be combined with replay caching in one verify call. This first version covers linear slot-owned windows, in-place authoritative commit, zero-length no-op and optional one after-token track checkpoint. Mamba ring replay/source-to-destination exact copy, quantized state codecs and stochastic-rounding identities need separate bindings/extensions.

Scope predicate: `p.window_length >= 1 and p.retained_state_units >= 1 and p.state_bytes >= 1 and p.transition_record_bytes >= 1 and (p.state_transition_and_readout_dependencies_explicit and p.linear_window_records_match_one_checkpoint_generation) and (p.verify_does_not_mutate_authoritative_checkpoint and p.verify_readout_error_policy_admitted) and ((retention == 'state_snapshots' and p.snapshots_are_complete_after_token_states) or (retention == 'transition_tape' and p.tape_contains_all_state_update_inputs_at_required_precision)) and (p.non_taped_transition_parameters_immutable_or_versioned and p.replay_normalization_gate_and_rounding_recipe_explicit) and (p.accepted_length_between_zero_and_window and p.accepted_length_is_consecutive_verified_prefix and p.commit_uses_original_checkpoint_for_that_window) and (p.live_checkpoint_slots_unique_valid_and_sentinel_policy_explicit and p.tape_snapshot_and_track_storage_ownership_safe) and (p.optional_tracking_saves_only_requested_after_token_state_inside_accepted_prefix) and (p.commit_applied_once_per_checkpoint_window_epoch and p.all_layers_and_state_shards_complete_before_publication) and (p.retained_records_publish_before_commit and p.commit_and_track_readers_retire_before_next_verify_overwrites_window) and (p.selected_retention_and_commit_backend_implemented_and_supported and p.snapshot_or_replay_state_error_admitted)`.

- Precondition `pre_recurrence` (implementation): `p.state_transition_and_readout_dependencies_explicit and p.linear_window_records_match_one_checkpoint_generation` — State evolves by declared ordered transitions F_t and optional readouts. Linear window records and the initial checkpoint refer to one immutable verification generation.
- Precondition `pre_freeze` (implementation): `p.verify_does_not_mutate_authoritative_checkpoint and p.verify_readout_error_policy_admitted` — Speculative verification observes virtual states/readouts but leaves authoritative state frozen. Parallel/MMA verification arithmetic may have a declared error policy distinct from commit replay.
- Precondition `pre_retention` (implementation): `(retention == 'state_snapshots' and p.snapshots_are_complete_after_token_states) or (retention == 'transition_tape' and p.tape_contains_all_state_update_inputs_at_required_precision)` — Snapshots retain post-token S_(t+1); tape retains every update dependency at its declared stage/dtype, not merely readout inputs. Output-only query data need not be saved when state updates do not depend on it.
- Precondition `pre_parameters` (implementation): `p.non_taped_transition_parameters_immutable_or_versioned and p.replay_normalization_gate_and_rounding_recipe_explicit` — Replay uses the required normalization, cached gate stage and arithmetic recipe; parameters not already resolved in tape remain immutable/versioned until commit.
- Precondition `pre_prefix` (implementation): `p.accepted_length_between_zero_and_window and p.accepted_length_is_consecutive_verified_prefix and p.commit_uses_original_checkpoint_for_that_window` — Accepted count a denotes exactly the consecutive prefix from this checkpoint. Commit never starts from a speculative all-token state or from another checkpoint epoch.
- Precondition `pre_slots` (implementation): `p.live_checkpoint_slots_unique_valid_and_sentinel_policy_explicit and p.tape_snapshot_and_track_storage_ownership_safe` — Live mutable checkpoint slots are unique/in range, invalid requests do no state/cache writes, and retained windows/snapshots are private or coherently owned. Optional track destinations are distinct and disjoint from live mutated slots.
- Precondition `pre_tracking` (implementation): `p.optional_tracking_saves_only_requested_after_token_state_inside_accepted_prefix` — An optional zero-based track step b saves S_(b+1) only for0<=b<a. Zero accepted count and out-of-prefix steps leave tracking storage unchanged.
- Precondition `pre_once` (implementation): `p.commit_applied_once_per_checkpoint_window_epoch and p.all_layers_and_state_shards_complete_before_publication` — One window generation commits at most once. All state shards/layers finish before a consumer treats the committed checkpoint generation as ready; no atomic multi-layer publication is implied.
- Precondition `pre_lifetime` (implementation): `p.retained_records_publish_before_commit and p.commit_and_track_readers_retire_before_next_verify_overwrites_window` — Commit sees complete records and accepted counts; the next verification cannot overwrite retained data until commit/tracking readers retire.
- Precondition `pre_backend` (implementation): `p.selected_retention_and_commit_backend_implemented_and_supported and p.snapshot_or_replay_state_error_admitted` — Chosen retention/backend/shape has an implemented path and its state numerical policy is admitted. Snapshot and replay paths need not be bitwise interchangeable without that separate premise.
- Constraint `c_window` (implementation): `p.window_length >= 1 and p.retained_state_units >= 1 and p.state_bytes >= 1 and p.transition_record_bytes >= 1` — Finite nonempty linear retained window and declared per-unit payload sizes.

## Local example and misuse

For a local instance of `contract`, identify the object `frozen_checkpoint_and_selective_prefix_materialization` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/deferred_prefix_state_commit@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/deferred_prefix_state_commit@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: T full post-token states or T update records, plus unchanged original checkpoint and optional track output.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: retained_snapshots
  level: dram
  scope: kernel
  bytes: p.retained_state_units * p.window_length * p.state_bytes
  when: retention == 'state_snapshots'
  note: Logical retained post-token payload; initial checkpoint and layout padding
    are separate.
- id: retained_transition_tape
  level: dram
  scope: kernel
  bytes: p.retained_state_units * p.window_length * p.transition_record_bytes
  when: retention == 'transition_tape'
  note: Logical replay dependencies; metadata and physical alignment are separate.
overlap: []
```

## Obligations and numerical checks

- `dependency_capture` (proof): Every transition input not recoverable from the checkpoint is retained with matching generation/precision; readout-only inputs may be omitted only when absent from F_t.
- `prefix_state` (test): Selected/replayed checkpoint equals the admitted accepted-prefix recurrence, with no suffix or already-mutated base contribution.
- `track_boundary` (test): Tracking uses post-token b+1, only within the accepted prefix, and its destinations cannot overwrite live checkpoint sources.
- `epoch_and_publication` (test): Commit is once per retained epoch and all state/record readers complete before checkpoint publication or next-window overwrite.

Replay can re-round state differently from saved verification snapshots or a tensor-core fold. Wrong accepted-count semantics, stale records, missing update inputs, double commit or a shifted tracking step changes future recurrent state.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
