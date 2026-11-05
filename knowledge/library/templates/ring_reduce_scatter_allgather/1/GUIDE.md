# ring_reduce_scatter_allgather@1: ring reduce scatter allgather

## When to consider

When all ranks need the same reduced vector, circulate disjoint chunks through a reduce-scatter ring, then forward completed chunks without reducing them again. P=4, ring index0 sends chunk3, reduces/forwards chunks2 then1, finishes chunk0, forwards chunks3 then2 and receives chunk1. Reapplying sum during the forwarding phase double-counts contributions.

## Computation before and after

When all ranks need the same reduced vector, circulate disjoint chunks through a reduce-scatter ring, then forward completed chunks without reducing them again.

At ring index r, first send chunk (r+P-1)%P. For j=2..P-1 receive/reduce/send chunk (r+P-j)%P. Then receive/reduce/store/send chunk r, applying the reduction post-operation exactly once. Forward completed chunks (r+P-j)%P for j=1..P-2 with no new arithmetic, and finally receive chunk (r+1)%P. Repeat disjoint loops and channels. Forget chunk partition to obtain the all-reduced vector at every rank.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| chunk_elements | compile / software | {"lo": 1, "hi": 1073741824} | Elements per rank chunk; channel/loop partition and protocol determine source binding. |

## Coupled constraints and scope

NVIDIA CUDA NCCL ring AllReduce with at least two ranks. Protocol SIMPLE/LL/LL128 has a separate transport realization. Noncommutative reductions require an explicitly compatible rank order; floating sums follow source ordering and rounding. Resource effects: Per channel and full loop, 2*(P-1) communication steps per rank, each moving at most chunk_elements. Reduction arithmetic occurs only during first P-1 stages, forwarding second stage requires no additional reductions.

Scope predicate: `p.ranks >= 2 and p.ring_permutation_valid and p.channels_partition_elements and p.tail_masked and p.reduction_reassociation_permitted and p.transport_correct`.

- Precondition `pre_0` (semantic): `p.ranks >= 2 and p.ring_permutation_valid` — Each rank has one predecessor and successor in a single complete ring.
- Precondition `pre_1` (semantic): `p.channels_partition_elements and p.tail_masked` — Channels and chunk loops partition the vector and invalid tail elements do not access storage.
- Precondition `pre_2` (semantic): `p.reduction_reassociation_permitted and p.transport_correct` — Transport preserves values and numerical contract permits the ring reduction order.

## Local example and misuse

For a local instance of `map`, identify the object `ring_reduce_scatter_allgather` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/nccl/delta/ring_reduce_scatter_allgather@1/template.yaml`. Original outline: `knowledge/studies/nccl/delta/ring_reduce_scatter_allgather@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Per channel and full loop, 2*(P-1) communication steps per rank, each moving at most chunk_elements. Reduction arithmetic occurs only during first P-1 stages, forwarding second stage requires no additional reductions.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `contributions` (proof): Before a chunk completes, it accumulates exactly one original contribution per rank.
- `completed_owner` (proof): Chunk r completes at ring index r before allgather forwarding begins for that chunk.
- `coverage` (proof): Every valid vector element belongs to exactly one channel/loop/chunk and reaches every output rank.
- `post_operation` (proof): Final normalization/post-op occurs once after all contributions, never at each hop.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
