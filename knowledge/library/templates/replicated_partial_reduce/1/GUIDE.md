# replicated_partial_reduce@1: replicated scalar partial table

## When to consider

When every CTA needs the same row scalar before producing its own output slice, consider replicating a sender-indexed partial table and reducing it locally at each consumer warp. Choose peers and warps_per_row to match whole-warp ownership; preserve exact-once slots, completed stores and collective allocation lifetime. The pinned two-CTA/four-warp RMSNorm instance expects32 bytes per destination row and uses a fresh phase0 barrier. It differs from assigning final output shards: every consumer gets the complete result. Subwarp contributors and repeated calls need explicit alternative ownership/generation protocols.

## Computation before and after

Replicate sender-indexed scalar tables and independently reduce at every consumer warp.

Allocate per-CTA FP32 table [row, local_warp + warps_per_row * sender_cta]. For peers=1, each warp's lane0 stores its scalar and a CTA barrier publishes the table. For peers>1, lane d of every warp sends the uniform scalar to peer d at that sender-indexed slot, including the local peer. Each destination's fresh barrier has one registration arrival and expects4*rows*warps_per_row*peers bytes. Match transaction completions to that destination and phase0, then wait before reads. Every warp of a row folds lane-strided table entries from the identity and performs a full32-lane butterfly so all its lanes receive the complete row result. Keep all allocations live through collective reader retirement. Reusing the table or barrier requires a separate generation/retirement protocol.

Recover the original computation: Replace each replicated table fold by the admitted combination of all unique sender-CTA/local-warp partials for that row; erase redundant table copies and reader computations.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| peers | compile / boundary | [1, 2, 4, 8, 16] | CTAs sharing one row-reduction result; one selects the local shared-memory implementation. |
| warps_per_row | compile / boundary | {"lo": 1, "hi": 8} | Whole warps contributing scalar partials per row in each CTA. |
| combine_op | compile / software | ["add", "max"] | Admitted FP32 associative-style combination with the matching identity and explicit numerical policy. |

## Coupled constraints and scope

One-shot FP32 row-scalar combination with whole-warp contributors and replicated results in every participating CTA. Local mode uses shared stores and one CTA barrier; cluster mode uses four-byte remote stores and a fresh phase0 barrier. The RMSNorm source binds four warp partials per row and two CTAs at N32768. Subwarp rows, mixed-width values, a reused phase0 barrier or unequal participant shapes need another verified binding or variant. Native remote-store/proxy semantics, masks, numerical precision and progress remain implementation obligations.

Scope predicate: `p.warp_partial_uniform and p.partial_partition_exact and p.identity_matches_op and p.combine_order_admitted and p.sender_slots_injective and p.cooperative_shape_matches_launch and p.scratch_initialized and p.completion_before_reads and (peers == 1 or (p.cluster_backend_supported and p.registration_completion_order_legal)) and p.one_shot_generation and p.readers_retired_before_cta_exit`.

- Precondition `pre_values` (semantic): `p.warp_partial_uniform and p.partial_partition_exact and p.identity_matches_op and p.combine_order_admitted` — Each contributing warp holds one defined scalar replicated to its lanes and the combination preserves the admitted numerical contract.
- Precondition `pre_mapping` (semantic): `p.sender_slots_injective and p.cooperative_shape_matches_launch` — Sender CTA and local warp select unique row slots and peer ranks agree with the launch.
- Precondition `pre_ready` (implementation): `p.scratch_initialized and p.completion_before_reads and (peers == 1 or (p.cluster_backend_supported and p.registration_completion_order_legal))` — Local barrier or exact remote byte completion makes every stored scalar visible before reads.
- Precondition `pre_lifetime` (semantic): `p.one_shot_generation and p.readers_retired_before_cta_exit` — The pinned protocol uses one fresh phase0 generation and retains every peer's shared allocation until all readers finish.
- Constraint `c_shape` (implementation): `peers * warps_per_row > 1 and p.rows >= 1 and p.rows * warps_per_row == p.block_warps and p.block_warps <= 32` — Every whole warp contributes to exactly one row and there is more than one partial.
- Constraint `c_storage` (resource): `4 * p.rows * warps_per_row * peers + p.barrier_bytes <= p.scratch_capacity_bytes` — Every replicated CTA table and its selected barrier fit the allocated scratch.
- Constraint `c_barrier` (implementation): `(peers == 1 and p.barrier_bytes == 0) or (peers > 1 and p.barrier_bytes == 8)` — The selected one-shot path allocates exactly one eight-byte barrier only for cluster mode.

## Local example and misuse

For a local instance of `reduce`, identify the object `replicated_scalar_partial_table` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/replicated_partial_reduce@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/replicated_partial_reduce@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Per-CTA table with all sender partials and optional one-shot transaction barrier.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: replicated_scratch
  level: smem
  scope: cta
  bytes: 4 * p.rows * warps_per_row * peers + p.barrier_bytes
  note: Per-CTA allocation; all-to-all writes and repeated reader folds remain unpriced.
overlap: []
```

## Obligations and numerical checks

- `incidence` (exhaustive): Each destination receives every row/sender/warp scalar exactly once at an injective slot and registers exactly the delivered bytes.
- `result_replication` (exhaustive): Every reader warp combines each scalar exactly once and broadcasts the same full result within that row.
- `completion_lifetime` (test): Every read follows completed stores and every peer allocation survives until all remote/local reads retire.

FP32 partition/fold order can differ from the original sequential reduction; max also needs an admitted NaN policy.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- requires: cross_lane_reduction (outside menu); The final complete row table is folded across32 reader lanes.

Structured contract: [template.yaml](template.yaml).
