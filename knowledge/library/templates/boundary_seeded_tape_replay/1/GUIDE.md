# boundary_seeded_tape_replay@1: recurrent training retention boundary before local replay

## When to consider

When recurrent training tapes are large but nonlocal entry states are small, retain those states and reconstruct local tapes in backward. Full-tape retention saves replay work; retaining nothing repeats boundary communication too. Preserve exact stage/generation identity and keep backward adjoint exchange in every mode.

## Computation before and after

Retain recurrent boundary seeds to replay local training tapes without replaying forward nonlocal communication.

Forward resolves each fragment entry state S_in, computes local prepared factors and recurrent tapes at declared numeric stages, then outputs. full_tape keeps the complete backward-required local tapes plus configured boundary/transfer artifacts. entry_boundaries keeps S_in and required configured boundary descriptors; on backward regenerate local factors and local subsegment carries, then replay the forward state/tape walk from those saved entries, without repeating forward nonlocal summary exchange/owner merge. nothing retains no forward-derived artifact; backward first reproduces boundary resolution, including forward summaries/exchange/merges, then local factors/tapes. Recompute forward output only if this replay has an actual output consumer. In all cases run the same reverse boundary-adjoint communication and local gradient computation from the matching tapes. Keep matching normalization statistics/parameter/partition generations; publish each recreated artifact before use and release it after its last backward reader. An external recompute handle that retains S_in is an entry-boundary retention owner, not a zero-memory no-artifact mode.

Recover the original computation: Replace each replayed artifact by its matching forward artifact at the same sequence/fragment/state generation and numeric stage. Every backward consumer then receives the same admitted operands and reverse boundary adjoints. Boundary retention removes only the dependency subgraph already represented by S_in, not the reverse communication or local feedback itself.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| forward_retention | compile / software | ["full_tape", "entry_boundaries", "nothing"] | Keep full local tapes and boundary artifacts, retain boundary seeds but replay local work, or retain no forward-derived artifact and rerun boundary resolution. Inputs/autograd frames/workspace storage are separate. |

## Coupled constraints and scope

FlashInfer deterministic KCP GDN training on SM100/SM103/SM107, local64-token chunks and grouped Q/K heads. full_tape binds boundary_mode=saved,save_mode=full; entry_boundaries binds saved/light or a valid caller-owned recompute handle; nothing binds boundary replay/saved=None. Optional transfer retention changes exact retained/recomputed artifacts but not the dependency boundary. Native source bitwise replay claims remain numeric-stage obligations of the outline; exact logical tests do not validate distributed GPU execution. Arbitrary mutable external state, hidden RNG draws and unconditional checkpoint-output elimination are excluded.

Scope predicate: `(p.boundary_retained_bytes >= 0 and p.full_local_tape_bytes >= 0) and (p.forward_boundary_resolution_local_tape_and_backward_consumers_explicit) and (p.replay_inputs_parameters_partition_and_numeric_generation_immutable) and (p.retained_entry_states_complete_for_all_local_replay_segments) and (p.retained_or_recreated_tapes_match_original_backward_stage_recipe) and (forward_retention != 'nothing' or p.forward_boundary_replay_collective_order_and_seed_generation_reproducible) and (p.recomputed_output_stage_elided_only_when_no_required_consumer) and (p.recreated_artifacts_publish_before_use_and_live_through_last_backward_reader) and (p.reverse_boundary_adjoint_exchange_retained_in_all_forward_retention_modes) and (p.selected_retention_norm_grouped_heads_and_replay_path_implemented)`.

- Precondition `pre_forward` (implementation): `p.forward_boundary_resolution_local_tape_and_backward_consumers_explicit` — Forward resolves fragment-entry states through an ordered summary/exchange/merge phase, then computes deterministic local factors and recurrent tapes. Enumerate every backward consumer, its required state/factor stage and the separate reverse boundary-adjoint exchange.
- Precondition `pre_inputs` (implementation): `p.replay_inputs_parameters_partition_and_numeric_generation_immutable` — Original local inputs, parameter/gate/norm stages, document/fragment order and precision options remain unchanged or versioned through replay. Context/shape checks alone do not establish input content identity.
- Precondition `pre_boundary` (implementation): `p.retained_entry_states_complete_for_all_local_replay_segments` — When boundaries are retained, they include every nonlocal entry-state dependency; local subsegment starts are saved or reconstructible from them and local factors. A whole-rank zero seed cannot replace a crossing fragment entry state.
- Precondition `pre_tapes` (implementation): `p.retained_or_recreated_tapes_match_original_backward_stage_recipe` — Recreated before-chunk states, corrected values, factors, gate rows and normalization checkpoints match the forward stages required by backward under its declared equality/error contract. Matching real recurrence alone does not prove native BF16/TF32 tape identity.
- Precondition `pre_none` (implementation): `forward_retention != 'nothing' or p.forward_boundary_replay_collective_order_and_seed_generation_reproducible` — No-artifact mode reruns boundary resolution including required forward exchange/merge with the same rank/fragment order and progress contract; it cannot silently start from zero or omit network dependencies.
- Precondition `pre_output` (implementation): `p.recomputed_output_stage_elided_only_when_no_required_consumer` — Replay may omit the forward output program only when the specific backward/recompute has no consumer of that output. Torch checkpoint recompute that must provide outputs follows its own frame/liveness proof, not this retention choice alone.
- Precondition `pre_lifetime` (implementation): `p.recreated_artifacts_publish_before_use_and_live_through_last_backward_reader` — Saved boundaries, optional transfers and replayed tapes publish before their consumers and remain alive through the last reader. Original forward arenas cannot be reused as retained tapes without explicit storage ownership; after that reader release does not extend into unrelated backward work.
- Precondition `pre_reverse` (implementation): `p.reverse_boundary_adjoint_exchange_retained_in_all_forward_retention_modes` — Reusing forward boundary seeds removes their forward replay work only; backward still computes local adjoint offsets, exchanges/merges required boundary adjoints and performs seeded reverse local walks.
- Precondition `pre_backend` (implementation): `p.selected_retention_norm_grouped_heads_and_replay_path_implemented` — Backend implements selected retention for target/shape, grouped Q/K heads, optional normalization/raw-gate path and graph mode. Source package/context/precision compatibility checks remain part of the binding.
- Constraint `c_payload` (implementation): `p.boundary_retained_bytes >= 0 and p.full_local_tape_bytes >= 0` — Declared logical artifact bytes follow actual saved transfer/normalization/subsegment settings; no-artifact mode does not imply zero total memory.

## Local example and misuse

For a local instance of `store`, identify the object `recurrent_training_retention_boundary_before_local_replay` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/boundary_seeded_tape_replay@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/boundary_seeded_tape_replay@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Full local tapes plus boundaries, boundaries only, or no forward-derived saved artifact with larger replay work.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: retained_forward_boundaries
  level: dram
  scope: kernel
  bytes: p.boundary_retained_bytes
  when: forward_retention != 'nothing'
  note: Entry states and configured owner-transfer/subsegment/norm artifacts; inputs
    and autograd context are not counted here.
- id: retained_local_forward_tapes
  level: dram
  scope: kernel
  bytes: p.full_local_tape_bytes
  when: forward_retention == 'full_tape'
  note: Complete local factor/state/corrected-value/gate tapes at source stages. Replay
    transient workspace is separate.
overlap: []
```

## Obligations and numerical checks

- `boundary_sufficiency` (proof): Saved fragment entries capture every nonlocal forward dependency needed by local replay; partial/zero seeds cannot substitute for a crossing entry.
- `tape_and_gradient_equivalence` (test): Full, boundary-seeded local replay and complete boundary replay give identical admitted backward tapes/gradients under fixed inputs and arithmetic stages.
- `communication_contract` (test): Only no-artifact mode repeats forward boundary exchange; every mode retains necessary reverse adjoint exchange, order and producer progress.
- `artifact_liveness` (test): Saved/replayed tensors keep their stage/generation through all backward readers, and elided output work has no required consumer.

A replay with the wrong boundary seed or different normalization/rounding stages changes gradients even when shapes match. Forward seed reuse does not remove reverse-adjoint communication or prove native bitwise equality.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
