# empty_work_pipeline_projection@1: no work epoch protocol

## When to consider

When an empty sparse tile removes all arithmetic, decide which pipeline channels still represent a logical work epoch. A forward empty tile may publish synthetic no-work statistics and require acknowledgment, while a backward empty tile may skip both producer and consumer data states entirely. Keep each retained channel balanced and every omitted channel unchanged. Skipping a signal while its consumer still waits, or signaling nonexistent P/O payload, is invalid even though the mathematical tile is empty.

## Computation before and after

Project a pipeline event graph onto its required no-work control protocol.

Specialize each logical work epoch on its shared emptiness predicate. Keep the original event graph when work is nonempty. For control_only, remove payload production/consumption but execute the declared no-work control publication and acknowledgment once, including required downstream empty output/statistic handling. Do not signal data-ready barriers for nonexistent operands or MMA results. For skip_data_epoch, skip both sides of each omitted data channel and do not advance any of their indices/phases; retain scheduler advancement and required empty-output stores as separate owned work. Drain prior acknowledgments before reusing retained control channels and preserve all declared tail/progress conditions. The two protocols are not interchangeable unless every waiter and state transition is rewritten consistently.

Recover the original computation: Reinsert semantically empty payload work at the omitted epoch. The original outputs and retained control dependencies are recovered, while future nonempty epochs see the same declared channel generations.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| empty_protocol | compile / software | ["control_only", "skip_data_epoch"] | Retain the externally required per-work control handshake or leave all data-pipeline states unchanged on an empty epoch; choose with the matching consumer protocol. |

## Coupled constraints and scope

Conditional no-work epochs in role-specialized reusable pipelines. Empty arithmetic does not by itself permit skipping synchronization. This does not cover divergent role predicates, changing a control-only path to skip-all without adapting consumers, or one-shot unmatched tails reused persistently.

Scope predicate: `p.all_roles_agree_empty_at_same_logical_scope and p.empty_outputs_and_side_effects_defined and p.retained_event_graph_matches_all_waiters and p.no_phantom_payload_readiness and p.retained_channels_advance_matching_generations and p.skipped_channels_leave_all_role_states_unchanged and p.prior_epoch_readers_retired_and_acks_drained and p.next_epoch_phase_agreement and p.progress_supported`.

- Precondition `pre_empty` (semantic): `p.all_roles_agree_empty_at_same_logical_scope and p.empty_outputs_and_side_effects_defined` — Every role observes the same empty scope; required zero outputs, statistics or external effects still have an owner.
- Precondition `pre_projection` (semantic): `p.retained_event_graph_matches_all_waiters and p.no_phantom_payload_readiness` — Every retained wait has its matching event and no empty signal authorizes reading unwritten payload.
- Precondition `pre_generation` (semantic): `p.retained_channels_advance_matching_generations and p.skipped_channels_leave_all_role_states_unchanged` — Control channels advance together and omitted payload channels preserve producer/consumer generations on every role.
- Precondition `pre_lifetime` (implementation): `p.prior_epoch_readers_retired_and_acks_drained and p.next_epoch_phase_agreement and p.progress_supported` — Old readers/acknowledgments cannot be confused with the next nonempty epoch, and every required signaling role can run.

## Local example and misuse

For a local instance of `sync`, identify the object `no_work_epoch_protocol` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flash-attention/delta/empty_work_pipeline_projection@1/template.yaml`. Original outline: `knowledge/studies/flash-attention/delta/empty_work_pipeline_projection@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Existing payload/control channels with explicit retained or unchanged generations.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `role_agreement` (test): All roles predicate the same empty epoch or stage and preserve required outputs/side effects.
- `event_balance` (proof): Every retained waiter has its matching current-generation event; skipped channels change no participant state and no unwritten payload is consumed.
- `transition_progress` (test): Empty-to-empty, nonempty-to-empty and empty-to-nonempty transitions maintain phase agreement and lifetime/progress.

Empty outputs and normalization statistics must retain the operator's defined convention.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- refines: per_slot_handshake (outside menu); States which channel generations participate on an empty logical epoch; per-slot handshake still supplies payload readiness and reuse on nonempty epochs.

Structured contract: [template.yaml](template.yaml).
