# replicated_cluster_partials@1: replicated small tuple cluster allreduce

## When to consider

When several CTAs own disjoint columns of one row and each needs a tiny shared statistic before continuing, replicate each warp partial to every peer and reduce locally. This avoids choosing a statistic owner and broadcasting its result, at the cost of peers-squared tuple traffic and replicated scratch. Match every16-byte store to the receiver transaction barrier, give dependent reductions separate banks and retire the cluster before reuse.

## Computation before and after

All-to-all publication of tiny warp partials followed by redundant complete reduction in each CTA.

Allocate phases private per-CTA tables [peers*warps_per_cta,4] of FP32 and one fresh transaction barrier per phase. Initialize barriers and complete the cluster publication join. In each phase, each warp first reduces its own four components by the declared warp tree. Destination CTA leader expects16*peers*warps_per_cta bytes. For every destination d, one sender lane issues an asynchronous16-byte remote shared store of that tuple to slot rank*warps_per_cta+warp and completes bytes on destination d barrier. Each receiving CTA waits for its own barrier, then each consuming converged warp folds all live slots, zeroing unused lanes when sender count<32. The resulting tuple is available to each local output shard. Preserve dependent-phase checkpoints and use a different table/barrier for the next phase. After the last phase, a cluster retirement join precedes CTA exit or any table reuse. There is no elected global statistic owner and no second result-broadcast phase.

Recover the original computation: For each phase erase replicated transports and view each receiver result as the same declared fold of the original disjoint sender partials. Replicas enable local consumers but do not add contributions to the logical sum.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| peers | compile / software | [2, 4, 8] | CTAs sharing the reduction domain; native witness8, smaller groups require neutral inactive receiver lanes and supported launch. |
| warps_per_cta | compile / software | {"lo": 1, "hi": 8} | Sending warps per CTA; native witness4; total sender count must fit32 receiver lanes. |
| phases | compile / boundary | {"lo": 1, "hi": 8} | Finite dependent reductions in the fused graph, with separate tables/barriers; native dual-normalization witness2, not a free reassociation of stages. |

## Coupled constraints and scope

Native FlashInfer gated-residual-next-RMSNorm uses H8192,8 CTAs,4 warps per CTA,4 FP32 components and2 dedicated phases. CLUSTER_N8 gives exactly32 sender slots; scalar statistics use one live tuple component and three zeros. Smaller peer/warp settings are inferred bindings requiring explicit zero lanes. This is a finite cluster allreduce, not the global-counter repeated allreduce, row-owned attention merge or an indefinitely reused ring.

Scope predicate: `(peers * warps_per_cta <= 32) and (phases >= 1) and (p.local_partials_cover_disjoint_complete_domain and p.tuple_components_and_reduce_order_explicit) and (p.cluster_launch_and_remote_async_store_supported and p.all_ctas_resident_and_matching_phases) and (p.sender_slot_equals_rank_times_warps_plus_warp and p.one_identical_tuple_store_per_sender_per_destination) and (p.barriers_initialized_and_cluster_published_before_stores and p.expected_bytes_equals_16_times_sender_count and p.transaction_completion_orders_payload_reads) and (p.each_phase_has_dedicated_barrier_and_partial_bank and p.no_phase_bank_reuse_before_all_readers_retire) and (p.each_consuming_warp_reduces_each_live_slot_once and p.inactive_reduction_lanes_are_initialized_zero) and (p.all_remote_accesses_complete_before_cluster_exit_or_storage_reuse and p.final_cluster_retirement_join_matches) and (p.fp32_partial_and_tuple_reduction_rounding_admitted and p.dependent_phase_inputs_preserve_requested_checkpoints)`.

- Precondition `pre_partition` (implementation): `p.local_partials_cover_disjoint_complete_domain and p.tuple_components_and_reduce_order_explicit` — Each sender warp owns an exact disjoint contribution; four FP32 tuple components have declared combine/order semantics.
- Precondition `pre_cluster` (implementation): `p.cluster_launch_and_remote_async_store_supported and p.all_ctas_resident_and_matching_phases` — All peers share a supported resident cluster and enter the same finite dependent phases without divergent exits.
- Precondition `pre_slots` (implementation): `p.sender_slot_equals_rank_times_warps_plus_warp and p.one_identical_tuple_store_per_sender_per_destination` — Every source rank/warp has an exclusive slot in every destination; each destination receives one identical completed16-byte tuple per sender, including its own rank.
- Precondition `pre_visibility` (implementation): `p.barriers_initialized_and_cluster_published_before_stores and p.expected_bytes_equals_16_times_sender_count and p.transaction_completion_orders_payload_reads` — Each destination transaction barrier expects exactly16*peers*warps bytes and completes only after those payload stores become readable.
- Precondition `pre_banks` (implementation): `p.each_phase_has_dedicated_barrier_and_partial_bank and p.no_phase_bank_reuse_before_all_readers_retire` — The finite phase sequence uses distinct banks/barriers, initially phase0; this version does not recycle banks by parity.
- Precondition `pre_reads` (implementation): `p.each_consuming_warp_reduces_each_live_slot_once and p.inactive_reduction_lanes_are_initialized_zero` — Each consuming warp combines one tuple from every sender; unused warp lanes are neutral and never read outside the allocated slot table.
- Precondition `pre_lifetime` (implementation): `p.all_remote_accesses_complete_before_cluster_exit_or_storage_reuse and p.final_cluster_retirement_join_matches` — No peer exits or reuses distributed shared memory before every remote writer and reader retires.
- Precondition `pre_numeric` (implementation): `p.fp32_partial_and_tuple_reduction_rounding_admitted and p.dependent_phase_inputs_preserve_requested_checkpoints` — The local and tuple reduction trees meet the numeric contract; downstream dependent statistics use their specified rounded inputs rather than algebraically expanded substitutes.
- Constraint `c_slots` (implementation): `peers * warps_per_cta <= 32` — One receiver warp can fold every sender slot.
- Constraint `c_phases` (implementation): `phases >= 1` — At least one finite allreduce invocation with a dedicated bank.

## Local example and misuse

For a local instance of `reduce`, identify the object `replicated_small_tuple_cluster_allreduce` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/replicated_cluster_partials@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/replicated_cluster_partials@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Dedicated sender-indexed tuple table and transaction barrier per phase in each CTA.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: replicated_tuple_tables
  level: smem
  scope: cta
  bytes: 16 * peers * warps_per_cta * phases
- id: phase_barriers
  level: smem
  scope: cta
  bytes: 8 * phases
overlap: []
```

## Obligations and numerical checks

- `sender_destination_bijection` (exhaustive): Every receiver slot maps to exactly one rank/warp sender, with identical tuple data at all receivers.
- `transaction_census` (test): Exactly one16-byte completion per sender/destination matches each destination expected byte count; publication precedes reads.
- `phase_lifetime` (test): Dependent phases use distinct banks and matching checkpoints, and terminal retirement protects all remote accesses.
- `numeric_tree` (test): Local and receiver reduction trees plus subsequent normalization satisfy admitted finite-precision behavior.

Replicating a statistic must not multiply its logical contribution. Reassociation of FP32 sums and replacement of a rounded residual by expanded moment identities can change downstream normalization.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
