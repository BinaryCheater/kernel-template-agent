# lagged_state_log_materialization@1: checkpoint with committed update log

## When to consider

When full recurrent state writes dominate but accepted transitions have small scalar-decay low-rank records, let the checkpoint lag and retain a committed update log. Read projected outputs from checkpoint plus log, materializing only for capacity or tracking. Choose log capacity with room for every next verify position, preserve accepted-prefix cursors and publish all fold shards before resetting storage.

## Computation before and after

Defer full state writes by retaining accepted low-rank update records and projecting from checkpoint plus log.

Keep checkpointS0 and accepted log records(g_j,d_j,k_j),j<n. Logical stateS_n=exp(G_n)S0+sum_j exp(G_n-G_j)*d_j*k_j^T, G_j=sum_i<=j g_i. To read S_n*q, project the checkpoint and key rows first, then combine update-weighted dot products without materializingS_n. Speculative verify may append tentative records beyondn, but only accepted prefix length advancesn. Before capacity/next-window overlap, materialize the accepted log into a new checkpoint, then advance circular base and reset retained count only after every state shard/layer finishes. Optional tracking materializes an earlier accepted prefix to a distinct slot without advancing the active checkpoint. Rejected suffix records are logically discarded and may be overwritten only when no reader retains them.

Recover the original computation: Decode checkpoint plus accepted log using the scalar-decay expansion. Ignore tentative suffix records and cursor mechanics; each accepted transition contributes once to the current logical state.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| log_capacity | compile / software | {"lo": 2, "hi": 65536} | Retained update slots; source uses power-of-two circular allocation and reserves space for the full next speculative window. |

## Coupled constraints and scope

Linear chains of scalar-decay matrix updates with immutable accepted-log records and explicit state generation. Source GDN supports exact-raw fold and compact high/low update/key records; their numerical contracts differ. Feature-dependent KDA decay and draft trees need separate factorization. The checkpoint may lag logical current state, unlike per-window immediate prefix commit.

Scope predicate: `p.state_updates_are_scalar_decay_rank_one and p.corrected_update_records_complete and p.record_rounding_and_projection_fold_error_admitted`.

- Precondition `pre_update` (semantic): `p.state_updates_are_scalar_decay_rank_one and p.corrected_update_records_complete` — Recorded transitionS_j=exp(g_j)*S_prev+d_j*k_j^T; d is the actual feedback-corrected update, not raw value.
- Precondition `pre_epoch` (semantic): `p.checkpoint_generation_matches_log and p.acceptance_is_consecutive_prefix and p.unique_live_mutable_slots` — Checkpoint, committed count, circular base and appended speculative records share one epoch and ownership.
- Precondition `pre_numeric` (implementation): `p.record_rounding_and_projection_fold_error_admitted and p.readers_retire_before_reuse` — Narrow records, reassociation and fold precision have an explicit contract; records stay live until readout/fold completes.
- Constraint `c_capacity` (resource): `log_capacity >= 2 * p.max_window and p.retained_count + p.next_window <= log_capacity` — Every verify position fits and produces output before acceptance; a full unconsumed log is never overwritten.

## Local example and misuse

For a local instance of `store`, identify the object `checkpoint_with_committed_update_log` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/lagged_state_log_materialization@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/lagged_state_log_materialization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One checkpoint plus bounded decay/key/update log, cursor and circular base.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `state_identity` (proof): The weighted-log expansion equals the ordered recurrence in exact arithmetic, using corrected updates from matching state generation.
- `prefix_cursor` (test): Accepted count includes exactly the consecutive committed prefix; rejected/tentative suffix never contributes to readout or fold.
- `capacity_epoch` (test): The full next verify window fits; folds finish before cursor reset and no live record is overwritten.
- `precision` (test): Projected readout and folded checkpoint satisfy the declared record/accumulation error policy.

Scalar-decay expansion changes association; high/low compact records need signed residual reconstruction. Exact-raw replay and compact MMA fold are different numerical realizations.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
