# peer_read_owned_collective@1: peer read owned collective

## When to consider

When symmetric peer memory is directly addressable, partition output coordinates across ranks and let each owner read the matching coordinate from every peer, then write the completed result to its declared consumers. symmetric/all_reduce.cuh:15-260,333-425; flattenIx(lane,32,rank,P,block,B,warp,W). Same ownership core appears in symmetric reduce-scatter.

## Computation before and after

When symmetric peer memory is directly addressable, partition output coordinates across ranks and let each owner read the matching coordinate from every peer, then write the completed result to its declared consumers.

Flatten lane/rank/block/warp into disjoint global coordinate owners. Each owner loads its own coordinate and then cycles through all other peer inputs once; cast to declared accumulation type, reduce, finalize once, cast result. AllReduce writes same completed value to corresponding output at every rank; ReduceScatter writes only the declared local owner output, and AllGather is the replication-only specialization. Split aligned packed interior from scalar prefix/suffix without overlap. Optional async peer-to-shared loads wait transaction barriers before reduction; shared-to-peer stores wait source-read completion before scratch reuse, with closing peer barrier providing publication/retirement.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| pack_bytes | compile / software | {"lo": 1, "hi": 16} | Legal aligned load/store pack; scalar prefix/suffix cover unaligned tails. |
| peer_unroll | compile / software | {"lo": 1, "hi": 16} | Number of independent peer loads batched in registers. |

## Coupled constraints and scope

Directly accessible symmetric peers and resident progress. This is full peer-read fan-in per owned coordinate, not ring reduce-scatter. Resource effects: Spreads arithmetic/stores across rank owners; peer unroll increases live registers, async staging adds shared storage.

Scope predicate: `p.coordinate_owner_partition_complete and p.corresponding_peer_offsets and p.peer_memory_accessible and p.peer_inputs_published_before_remote_read and p.outputs_complete_before_consumers and p.no_inplace_overwrite_of_unread_input and p.accumulator_casts_and_peer_order_admitted and p.target_pack_and_async_completion_legal`.

- Precondition `pre_0` (semantic): `p.coordinate_owner_partition_complete and p.corresponding_peer_offsets and p.peer_memory_accessible` — Every output coordinate has exactly one owner and all rank inputs have corresponding offsets.
- Precondition `pre_1` (semantic): `p.peer_inputs_published_before_remote_read and p.outputs_complete_before_consumers and p.no_inplace_overwrite_of_unread_input` — Opening/closing barriers and in-place lifetime ordering cover every read.
- Precondition `pre_2` (semantic): `p.accumulator_casts_and_peer_order_admitted and p.target_pack_and_async_completion_legal` — Accumulator precision, cyclic rank order, casts and optional async shared staging are explicit.

## Local example and misuse

For a local instance of `map`, identify the object `peer_read_owned_collective` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/nccl/delta/peer_read_owned_collective@1/template.yaml`. Original outline: `knowledge/studies/nccl/delta/peer_read_owned_collective@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Spreads arithmetic/stores across rank owners; peer unroll increases live registers, async staging adds shared storage.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `cover` (proof): Every owned coordinate reads each peer exactly once; all coordinates have one owner.
- `consumers` (proof): Completed value is stored once to every declared consumer, without overwriting an unread input.
- `numeric` (proof): Accumulator type, casts and peer reduction order match selected numerical contract; changing backend need not be bitwise equal.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
