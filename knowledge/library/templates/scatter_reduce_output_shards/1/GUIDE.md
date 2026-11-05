# scatter_reduce_output_shards@1: peer owned reduction shards

## When to consider

When K is split across cooperating peers that hold the same output fragment, consider distributing final output shards among those peers and sending each owner its required contributions. Preserve one defined FP32 contribution per peer and one final writer per element; require equal fragment maps, E divisible by peers and read acknowledgements before mailbox reuse. The pinned DSMEM binding uses128 threads and owner-first FP32 addition, but empty source K partitions need an identity write or exclusion and other accumulation types require their own numerical variant.

## Computation before and after

Scatter partial output shards to their owners and reduce each owner locally before disjoint final stores.

For peer count S and per-thread fragment extent E, peer r owns indices r*(E/S)..(r+1)*(E/S)-1. Each sender s keeps its own shard locally and sends receiver r's shard to r's slot s-(s>r), with thread index innermost in memory. Receiver r arms FULL for (S-1)*threads*(E/S)*element_bytes, waits for all incoming shard bytes and combines its local shard first, then received slots in ascending sender order with self omitted. After all its readers finish, r acknowledges each sender's EMPTY event. A sender waits for its previous S-1 acknowledgements before reusing outgoing mailbox slots. Only peer r stores r's combined shard.

Recover the original computation: Gather the disjoint owner output shards and replace each by the reduction of all defined peer partials at its original fragment coordinate, retaining the declared floating-point order.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| peers | compile / boundary | {"lo": 2, "hi": 16} | Number of defined input partials and disjoint output shards in the cooperative group. |

## Coupled constraints and scope

Defined same-layout FP32 partial output fragments distributed across a cooperative peer group, with equal per-thread output shards and sender-specific remote shared mailboxes. No arithmetic atomics or one-CTA finalizer is introduced. FP32 combines follow owner-first then ascending-other-sender order; global output casting/epilogue operations remain separate. Remote store cost and latency are unpriced; mailbox storage is explicit.

Scope predicate: `p.local_partials_defined and p.accumulator_fp32 and p.numeric_reassociation_allowed and p.fragment_map_equal_and_injective and p.async_peer_store_supported and p.full_bytes_and_empty_acks_correct and p.mailbox_phase_ordered`.

- Precondition `pre_partials` (semantic): `p.local_partials_defined and p.accumulator_fp32 and p.numeric_reassociation_allowed` — Defined FP32 partials and the numerical contract admit the specified owner-first combine order; empty K contributors supply an identity or are excluded.
- Precondition `pre_layout` (implementation): `p.fragment_map_equal_and_injective and p.async_peer_store_supported` — Peers agree on fragment coordinates and the target can deliver shard bytes to the selected peer mailbox.
- Precondition `pre_generation` (semantic): `p.full_bytes_and_empty_acks_correct and p.mailbox_phase_ordered` — Full signals follow all expected bytes and no sender overwrites a generation before every receiver has read it.
- Constraint `c_partition` (semantic): `peers == p.partial_count and p.elements_per_thread % peers == 0` — Each thread's fragment splits into equal peer-owned shards.
- Constraint `c_capacity` (resource): `(peers - 1) * p.threads * (p.elements_per_thread // peers) * p.element_bytes + p.control_bytes <= p.mailbox_capacity_bytes` — Sender-specific receive slots and control state fit the mailbox budget.

## Local example and misuse

For a local instance of `reduce`, identify the object `peer_owned_reduction_shards` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/scatter_reduce_output_shards@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/scatter_reduce_output_shards@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Sender-specific mailbox slots reused only after recipient acknowledgements.. Storage: One receive shard per other sender plus full/empty control state.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: mailbox
  level: smem
  scope: cta
  bytes: (peers - 1) * p.threads * (p.elements_per_thread // peers) * p.element_bytes
    + p.control_bytes
overlap: []
```

## Obligations and numerical checks

- `exact_contributors` (exhaustive): Each owned element combines exactly one partial from every peer with no sender-slot collision.
- `exact_output_owner` (exhaustive): Every original output fragment coordinate is stored by exactly its owning peer and maps injectively to the final output.
- `generation_lifetime` (test): Full waits follow all shard bytes and empty acknowledgements follow every mailbox reader before the sender can overwrite the next generation.
- `defined_numeric_partials` (test): Every peer contributes a defined partial including identity values for empty upstream partitions when admitted; the actual combine order meets the numerical contract.

Owner-first ordering and upstream K partitioning alter floating-point grouping; numerical acceptance is a required premise.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- requires: [reduction_partition@1](../../reduction_partition/1/GUIDE.md), [reduction_partition@2](../../reduction_partition/2/GUIDE.md); Upstream K partitions must produce a defined local partial for every peer.

Structured contract: [template.yaml](template.yaml).
