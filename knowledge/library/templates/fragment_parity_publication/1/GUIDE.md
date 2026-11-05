# fragment_parity_publication@1: fragment readiness mask

## When to consider

Use when a consumer can read independent parts of a producer's output before the whole output finishes. Bind num_fragments and group_size from the two layouts, then publish each completed fragment with a release-XOR and acquire just the required group. With four fragments and group_size2, bits0/1 permit the first read while bits2/3 are pending. Keep reuse after all reads; parity alone does not prevent overwrite.

## Computation before and after

Release-toggle per-fragment parity and acquire only the current read group's bits.

Assign fragment i to bit i of one unsigned 64-bit word initialized to zero. A producer completes its payload writes then release-XORs bit i exactly once. Generation g expects all owned bits one when g is even and zero when g is odd. For read group j form a mask over bits j*group_size through (j+1)*group_size-1. Acquire-load the word until the observed XOR expected value has no bits in that group, then read only that group's current-generation payload. Other groups may still be incomplete. Protect payload lifetime and generation reuse separately.

Recover the original computation: Replace each bit toggle and grouped parity wait with completed-producer events for exactly the fragments read by that consumer.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| num_fragments | compile / boundary | {"lo": 1, "hi": 64} | Number of disjoint independently published payload fragments in one buffer generation. |
| group_size | compile / boundary | {"lo": 1, "hi": 64} | Number of consecutive producer fragments required by one consumer read group. |

## Coupled constraints and scope

Device-local fragment publication with one through 64 distinct fragment bits and complete contiguous equal-sized consumer groups. Producer ordering is free subject to payload ownership. Parity is only a readiness representation, not a lifetime or capacity proof. Duplicate toggles, untracked overwrites and generation lapping invalidate the protocol; partial final groups need another version or binding.

Scope predicate: `p.exactly_one_publication_per_fragment and p.no_generation_lap and p.release_acquire_supported`.

- Precondition `pre_once` (semantic): `p.exactly_one_publication_per_fragment and p.initial_mask_zero` — Every fragment toggles exactly once per generation and generation zero starts from a zero mask.
- Precondition `pre_lifetime` (semantic): `p.no_generation_lap and p.payload_reuse_after_reads` — A fragment cannot enter its next generation until all current consumers finish reading its payload.
- Precondition `pre_memory` (implementation): `p.release_acquire_supported and p.producers_can_progress` — Payload publication has the required memory ordering and awaited producers can execute.
- Constraint `c_shape` (semantic): `num_fragments == p.producer_fragments and group_size == p.fragments_per_read` — Bit ownership and read groups agree with the producer and consumer layouts.
- Constraint `c_groups` (implementation): `group_size <= num_fragments and num_fragments % group_size == 0` — This version covers complete equal contiguous read groups.

## Local example and misuse

For a local instance of `sync`, identify the object `fragment_readiness_mask` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/deepgemm/delta/fragment_parity_publication@1/template.yaml`. Original outline: `knowledge/studies/deepgemm/delta/fragment_parity_publication@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One 64-bit mask per reusable payload slot plus consumer-local pending bits.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `bit_ownership` (exhaustive): Producer fragments have disjoint bits and every consumer group mask contains exactly its required fragments.
- `parity_generation` (exhaustive): Under exact-once and no-lap premises a matching bit is from the requested generation rather than an earlier or later generation.
- `publication_visibility` (test): A matching acquire observes every required fragment's completed payload stores.
- `read_lifetime` (test): Every payload read finishes before its fragment is overwritten or toggled for the next generation.
- `progress` (test): Scheduling and buffer capacity allow all awaited producers and consumers to run without circular waits.

No arithmetic reassociation is introduced; upstream quantization and downstream accumulation retain their own contracts.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
