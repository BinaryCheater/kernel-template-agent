# runtime_partition_capacity@1: bounded live partition prefix with fixed scratch topology

## When to consider

When lengths change between replays, keep split workspace/topology fixed at a prepared capacity and derive a smaller useful prefix. Use the live count for work and merge loads, capacity for addresses, and physical participation for barriers. Inactive work does not automatically permit early return.

## Computation before and after

Vary useful split count within fixed physical capacity and collective topology.

For T=p.total_tiles, I=p.instruction_tile_group, R=requested_splits and C=capacity, derive L=max(I,ceil(T/(R*I))*I) and A=ceil(T/L). A=0 when T=0 and always A<=R<=C. Useful split s in[0,A) owns tiles[s*L,min((s+1)*L,T)); source row masks remain inside the tiles. Produce one complete merge state per useful split. Allocate/index partial rows with fixed capacity C and keep the prepared participant count unchanged. Reducers load only useful published s<A; inactive physical slots contribute no loads. Inactive ranks either complete local teardown with no outstanding collective consumer, or join every fixed collective using neutral state. Merge useful states under their declared frame. A count change inside C reuses storage and compiled topology; C/topology changes require new preparation and resource bindings.

Recover the original computation: Forget inactive physical capacity slots and concatenate useful aligned ranges; every live tile occurs exactly once and the declared partial-state merge recovers the complete reduction. Empty work follows its explicit output policy.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| capacity | compile / software | {"lo": 1, "hi": 2147483647} | Prepared upper bound C for split states and addressing; independent of the current useful count. |
| requested_splits | runtime / software | {"lo": 1, "hi": 2147483647} | Runtime requested fanout R<=C; geometry and instruction alignment derive actual useful A, which can be smaller. |
| inactive_protocol | compile / software | ["retire_local", "participate_neutral"] | Locally retire inactive producer CTAs only when no collective awaits them, or keep inactive ranks in the fixed collective with neutral state. |

## Coupled constraints and scope

Source task-scheduled FMHA/MLA split-KV on SM100/103/107. The FMHA runtime helper uses instruction-aligned spans; separate and collective reducers retain prepared capacity/topology. Capacity selection/bucketing and residency heuristics are upstream policy, not this rule. No cancellation or CLC protocol follows from the logical partition. Neutral-state format depends on raw versus normalized partial representation.

Scope predicate: `(requested_splits <= capacity and p.total_tiles >= 0 and p.instruction_tile_group >= 1) and (p.visible_tile_domain_complete_and_row_exact_masks_preserved and p.partial_state_merge_contract_admitted) and (p.all_producers_reducers_and_work_decoders_use_same_runtime_count_generation) and (p.scratch_row_strides_derive_from_capacity_and_all_addresses_fit and p.capacity_and_participant_topology_unchanged_during_launch) and (p.live_partial_state_published_before_merge and p.reducer_never_loads_unpublished_inactive_slots) and (inactive_protocol != 'retire_local' or p.retired_ranks_have_no_collective_consumers_and_complete_resource_teardown) and (inactive_protocol != 'participate_neutral' or p.inactive_ranks_supply_neutral_states_and_join_all_fixed_collectives) and (p.zero_work_output_policy_explicit_and_no_uninitialized_partial_read and p.replay_control_reset_and_workspace_exclusivity_valid) and p.count_alignment_and_division_arithmetic_valid`.

- Precondition `pre_partition` (semantic): `p.visible_tile_domain_complete_and_row_exact_masks_preserved and p.partial_state_merge_contract_admitted` — Visible source tiles partition completely, with exact logical masks; merge representation and rounding are specified.
- Precondition `pre_count` (semantic): `p.all_producers_reducers_and_work_decoders_use_same_runtime_count_generation and p.count_alignment_and_division_arithmetic_valid` — Count, division metadata and workload lengths belong to one immutable invocation generation. Count/alignment products and ceil numerators fit their integer types; precomputed exact-division metadata matches the divisor.
- Precondition `pre_address` (implementation): `p.scratch_row_strides_derive_from_capacity_and_all_addresses_fit and p.capacity_and_participant_topology_unchanged_during_launch` — Scratch addresses use C, never live A; collective ranks and resource allocations retain their prepared topology.
- Precondition `pre_publish` (implementation): `p.live_partial_state_published_before_merge and p.reducer_never_loads_unpublished_inactive_slots` — Only completed useful partials may be loaded; inactive caller scratch may be stale or uninitialized.
- Precondition `pre_retire` (implementation): `inactive_protocol != 'retire_local' or p.retired_ranks_have_no_collective_consumers_and_complete_resource_teardown` — An early-retired CTA cannot leave a barrier arrival, dependency release or TMEM retirement outstanding.
- Precondition `pre_neutral` (implementation): `inactive_protocol != 'participate_neutral' or p.inactive_ranks_supply_neutral_states_and_join_all_fixed_collectives` — Fixed collective participants remain present even when their logical contribution is empty.
- Precondition `pre_empty` (implementation): `p.zero_work_output_policy_explicit_and_no_uninitialized_partial_read and p.replay_control_reset_and_workspace_exclusivity_valid` — A=0 has declared output semantics without reading scratch; replay resets and concurrent workspace ownership remain valid.
- Constraint `c_counts` (semantic): `requested_splits <= capacity and p.total_tiles >= 0 and p.instruction_tile_group >= 1` — A nonnegative live tile count and positive instruction group admit bounded aligned splits.

## Local example and misuse

For a local instance of `reduce`, identify the object `bounded_live_partition_prefix_with_fixed_scratch_topology` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/runtime_partition_capacity@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/runtime_partition_capacity@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Capacity-strided partial states with a separately agreed live-prefix count.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `aligned_exact_cover` (test): Derived A<=R<=C and useful ranges partition[0,T) with no empty useful split.
- `capacity_addressing` (test): Live-count changes never alter fixed row strides or make reducers read inactive scratch.
- `inactive_protocol` (test): Inactive ranks preserve collective progress and resource/dependency teardown for the chosen physical topology.
- `metadata_and_replay` (test): Producers and reducers share the same count generation and controls reset before exclusive replay.

Changing live partitions changes reduction association. Masking an inactive value after loading it does not prove that stale NaN/uninitialized data is harmless; avoid the load or construct a declared neutral state.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
