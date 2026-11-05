# multimem_peer_address_operation@1: multimem peer address operation

## When to consider

When peer buffers have corresponding offsets and a registered multicast address, replace software fanout stores or peer-load sums with the corresponding multimem address operation. Choose operation from the collective contract, keep stream ready/retired handshakes, and reject unregistered peers or scalar tails. Two peers containing [1,2,3,4] and [10,20,30,40] produce [11,22,33,44] in SUM mode.

## Computation before and after

A multicast virtual address applies one vector operation to corresponding bytes in each registered peer allocation.

Bind corresponding peer offsets through a multicast allocation. For each independent 16-byte vector, replicate_store loads its bits from one ordinary source and uses multimem.st.global.v4.f32 at the multicast destination, writing identical bits to every member. sum_load executes multimem.ld_reduce.global.add.v4.f32 at the multicast source and stores the resulting four FP32 sums to an ordinary destination. Peer ready and completion stream-memory operations bracket the source realization; a result at one rank versus all ranks is a caller consumer-set choice.

Recover the original computation: Expand the multicast address to its member allocations: replicate_store copies identical source bits to every peer and sum_load adds each peer FP32 value once per coordinate.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| operation | compile / boundary | ["replicate_store", "sum_load"] | Copy one vector to all registered peers, or sum corresponding FP32 vectors from those peers. |

## Coupled constraints and scope

NVIDIA SM90+ NVLink SHARP multicast allocations with explicit registered membership and full 16-byte vectors. SUM load is FP32 only; replicate store transports bit patterns. Distinct from CTA-cluster shared multicast and contraction instruction selection.

Scope predicate: `p.size_bytes % 16 == 0 and p.address_alignment_bytes >= 16 and p.multicast_membership_matches_collective and p.peer_allocations_live and p.sm90_multimem_supported and p.all_sources_ready_before_operation and p.all_readers_retired_before_source_reuse and p.sum_rounding_admitted or operation == 'replicate_store'`.

- Precondition `pre0` (semantic): `p.multicast_membership_matches_collective and p.peer_allocations_live and p.sm90_multimem_supported` — Virtual address membership, peer lifetime and hardware gates are explicit.
- Precondition `pre1` (semantic): `p.all_sources_ready_before_operation and p.all_readers_retired_before_source_reuse` — Upstream stream publication and downstream retirement protect peer payload.
- Precondition `pre2` (semantic): `p.sum_rounding_admitted or operation == 'replicate_store'` — SUM mode requires admitted peer reduction arithmetic.
- Constraint `c0` (semantic): `p.size_bytes % 16 == 0 and p.address_alignment_bytes >= 16` — Source realization uses full 16-byte vectors only.

## Local example and misuse

For a local instance of `transfer`, identify the object `multimem_peer_address_operation` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/fuser/delta/multimem_peer_address_operation@1/template.yaml`. Original outline: `knowledge/studies/fuser/delta/multimem_peer_address_operation@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

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
