# multimem_peer_address_operation@2: multimem peer address operation

## When to consider

When matching peer addresses are registered for multicast, select an exact hardware reduction-load or replicated-store capability and retain payload publication/retirement. Version2 adds explicit pack/type/target records and min/max, with old FP32 vector sum/store as bindings.

## Computation before and after

A multicast virtual address applies one vector operation to corresponding bytes in each registered peer allocation.

Bind corresponding peer offsets through registered multicast mapping. For each supported aligned pack, replicate_store writes identical bits to all members; sum_load/min_load/max_load applies the exact capability-record hardware reduction over corresponding member values into one ordinary result. Bind PTX element type and accumulation modifier, not merely storage dtype. All peer input writes precede load; completion/retirement follows final access. Unsupported combinations are rejected. Scalar tails use separate legal capability records; they are never implied by full-vector support. Pre-scaling and final division remain caller stages, not silently folded into raw load.

Recover the original computation: Expand the multicast address to its member allocations: replicate_store copies identical source bits to every peer and sum_load adds each peer FP32 value once per coordinate.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| operation | compile / boundary | ["replicate_store", "sum_load", "min_load", "max_load"] | Source-supported hardware operation, constrained by exact type/pack capability record. |
| pack_bytes | compile / software | {"lo": 1, "hi": 16} | Exact supported instruction pack width; arbitrary widths in range are not legal without a capability record. |

## Coupled constraints and scope

Registered multicast peer operations with an exact operation/type/pack/target record. NCCL witnesses FP32 sum packs4/16, FP64 sum8, integer32/64 sum/min/max, half/bfloat16 vector and subpart paths, and architecture-specific FP8 packs. Not every combination in these families is legal; source specializations select actual menus.

Scope predicate: `(p.multimem_capability_record_matches_operation_type_pack_and_target and p.address_alignment_bytes >= pack_bytes and p.size_bytes % pack_bytes == 0) and (p.multicast_membership_matches_collective and p.peer_allocations_live and p.multimem_target_and_toolchain_supported) and (p.all_sources_ready_before_operation and p.all_readers_retired_before_source_reuse) and (p.reduction_numeric_contract_admitted or operation == 'replicate_store')`.

- Precondition `pre0` (semantic): `p.multicast_membership_matches_collective and p.peer_allocations_live and p.multimem_target_and_toolchain_supported` — Virtual address membership, peer lifetime and hardware gates are explicit.
- Precondition `pre1` (semantic): `p.all_sources_ready_before_operation and p.all_readers_retired_before_source_reuse` — Upstream stream publication and downstream retirement protect peer payload.
- Precondition `pre2` (semantic): `p.reduction_numeric_contract_admitted or operation == 'replicate_store'` — Hardware accumulation precision, min/max exceptional values, integer wrap and pre/post arithmetic are bound explicitly.
- Constraint `c_capability` (implementation): `p.multimem_capability_record_matches_operation_type_pack_and_target and p.address_alignment_bytes >= pack_bytes and p.size_bytes % pack_bytes == 0` — Capability record supplies legal PTX operation, element type, vector width and target/toolchain gate. Prefix/suffix must use a separate supported pack.

## Local example and misuse

For a local instance of `transfer`, identify the object `multimem_peer_address_operation` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/nccl/delta/multimem_peer_address_operation@2/template.yaml`. Original outline: `knowledge/studies/nccl/delta/multimem_peer_address_operation@2/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Source-bound physical buffer; exact byte count and capacity supplied by binding.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `membership` (test): Multicast registration denotes exactly the intended peer addresses for every vector.
- `payload_lifetime` (test): Ready signals follow input writes and retirement signals follow all relevant reads.
- `operation_values` (test): Vector store preserves bits or vector load performs the requested peer SUM under the numerical policy.

Address/storage transformation preserves values; arithmetic changes require separate obligations.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
